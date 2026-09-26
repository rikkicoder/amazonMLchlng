#!/usr/bin/env python3
"""Business Entity Resolution -- end-to-end pipeline (data -> blocking -> matching -> output TSVs).

Stages (each writes its result under --work and is skipped when that result exists, so a crash resumes):
  load      TSVs -> parquet; ground truth, fixed random priority and fold per training S1
  dict      learn the native-script -> Latin token dictionary from training pairs (S1 priority >= 0.6)
  norm      L4 normalisation of every record (parallel) + per-country IDF tables
  finetune  contrastive fine-tuning of multilingual-e5-small (S1 priority 0.50-0.53, disjoint from A/B)
  embed     encode every record ("name , address") with the fine-tuned encoder (fp16, GPU, resumable)
  block     exact GPU top-k inside each country: S1->pool top-10  U  pool->S1 top-3, + context features
  feats     pairwise name/address/number features for training candidates (A and B slices)
  train     cheap + full LightGBM on A; on B: cheap-filter threshold, stage-2 stacking, final threshold, F0.5
  predict   test: pair features -> cheap filter (= candidate_pairs.tsv) -> full model -> stage 2 -> threshold
            + one-to-one -> matching_results.tsv; then runs utils/validate_submission.py if present

    python src/run_pipeline.py                    # everything
    python src/run_pipeline.py --stages predict   # only the test predictions (models already trained)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from ber import config as C
from ber.util import Tee, exists, load_json, log, save_json

STAGES = ["load", "dict", "norm", "finetune", "embed", "block", "feats", "train", "predict"]
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))       # student_resource/ when run in place


def default(p, fallback):
    return p if os.path.exists(p) else fallback


# ----------------------------------------------------------------------------- stages
def st_load(a):
    from ber.data import build_split
    for split in ("train", "test"):
        build_split(a.data_dir, a.work, split)


def st_dict(a):
    from ber.prep import learn_translit_dict
    learn_translit_dict(a.work)


def st_norm(a):
    from ber.prep import learn_translit_dict, normalize_split
    dct = learn_translit_dict(a.work)
    for split in ("train", "test"):
        normalize_split(a.work, split, dct, a.norm_workers)


def ft_dir(a):
    return a.ft_model or os.path.join(a.work, "models", "ft_joint")


def st_finetune(a):
    from ber.embed import finetune
    out = ft_dir(a)
    if os.path.exists(os.path.join(out, "config.json")):
        log(f"fine-tuned encoder found at {out} -- skipping")
        return
    finetune(a.work, out)


def st_embed(a):
    from ber.embed import embed_split
    for split in ("test", "train"):
        embed_split(a.work, split, ft_dir(a))


def train_rows(s1):
    return np.where((s1.prio.values >= C.TRAIN_A[0]) & (s1.prio.values < C.TRAIN_B[1]))[0]


def st_block(a):
    from ber.data import cache_paths
    from ber.knn import add_context, block
    os.makedirs(os.path.join(a.work, "block"), exist_ok=True)
    for split in ("test", "train"):
        out = os.path.join(a.work, "block", f"{split}_cands.parquet")
        if os.path.exists(out):
            continue
        p1, pp = cache_paths(a.work, split)
        s1 = pd.read_parquet(p1, columns=["country", "prio"] if split == "train" else ["country"])
        pool = pd.read_parquet(pp, columns=["country", "src"])
        E1 = np.load(os.path.join(a.work, "emb", f"{split}_s1.npy"), mmap_mode="r")
        Ep = np.load(os.path.join(a.work, "emb", f"{split}_pool.npy"), mmap_mode="r")
        rows = train_rows(s1) if split == "train" else None
        present = None
        if split == "train":
            lo, hi = C.TRAIN_HIDDEN_S1
            present = ~((s1.prio.values >= lo) & (s1.prio.values < hi))
            log(f"hiding {(~present).sum():,} training S1 ({(~present).mean():.1%}) from the search: their pool "
                f"records become ownerless like the test set's extra ones")
        t0 = time.time()
        log(f"blocking {split}: forward top-{C.K_FWD} for {len(s1) if rows is None else len(rows):,} S1, "
            f"reverse top-{C.M_REV} for {len(pool):,} pool records")
        c, q_top1, p_top1 = block(E1, Ep, s1.country.values, pool.country.values, C.K_FWD, C.M_REV, rows, present)
        c = add_context(c, q_top1, p_top1, pool.src.values)
        c = c.sort_values(["qi", "pj"]).reset_index(drop=True)
        c.to_parquet(out, index=False)
        log(f"blocking {split}: {len(c):,} candidate pairs ({len(c) / max(c.qi.nunique(), 1):.1f} per S1) "
            f"in {time.time() - t0:.0f}s")


def split_arrays(a, split):
    from ber.data import load_split
    from ber.prep import load_norm
    s1, pool = load_split(a.work, split)
    ns, npl, idf = load_norm(a.work, split)
    S = dict(nname=ns.nname.values, naddr=ns.naddr.values, name=s1.name.values, addr=s1.addr.values,
             country=s1.country.values)
    P = dict(nname=npl.nname.values, naddr=npl.naddr.values, name=pool.name.values, addr=pool.addr.values,
             country=pool.country.values)
    return s1, pool, S, P, idf


def tri_tables(a, split, S, P):
    """Per-country character-trigram IDF for the split (cached)."""
    import pickle
    from ber.feats import build_tri_idf
    path = os.path.join(a.work, "cache", f"{split}_tri_idf.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    log(f"{split}: building character-trigram IDF tables")
    tri = build_tri_idf(np.concatenate([S["nname"], P["nname"]]), np.concatenate([S["naddr"], P["naddr"]]),
                        np.concatenate([S["country"], P["country"]]))
    with open(path, "wb") as f:
        pickle.dump(tri, f)
    return tri


def st_feats(a):
    from ber.data import owner_index
    from ber.feats import compute
    out = os.path.join(a.work, "feats", "train.parquet")
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    s1, pool, S, P, idf = split_arrays(a, "train")
    c = pd.read_parquet(os.path.join(a.work, "block", "train_cands.parquet"))
    log(f"pair features for {len(c):,} training candidate pairs")
    for f, v in compute(S, P, c.qi.values, c.pj.values, idf, a.feat_workers).items():
        c[f] = v
    tri = tri_tables(a, "train", S, P)
    for f, v in compute(S, P, c.qi.values, c.pj.values, idf, a.feat_workers, tri=tri).items():
        c[f] = v
    c["label"] = (owner_index(s1, pool)[c.pj.values] == c.qi.values).astype(np.int8)
    c.to_parquet(out, index=False)


def pool_nums(pool, pjs):
    from ber.text import numbers
    return {int(j): numbers(pool.addr.values[j]) for j in np.unique(pjs)}


def stage2_frame(sv, pool, pn, pa, emb, use_s2x):
    from ber.model import stage2_extra, stage2_features
    d = stage2_features(sv, pn, pa, pool_nums(pool, sv.pj.values), pool.src.values)
    if use_s2x:
        d = stage2_extra(d, emb, pool.src.values)
    return d


def st_train(a):
    from ber.data import load_split, owner_index
    from ber.model import (RULES, S2_COLS, S2X_COLS, apply_rule, blocking_stats, fit_lgb, macro_f05, predict,
                           tune_rule)
    from ber.prep import load_norm
    mdir = os.path.join(a.work, "models")
    if exists(*(os.path.join(mdir, f) for f in ("cheap.lgb", "full.lgb", "stage2.lgb", "decision.json"))):
        log("models found -- skipping train")
        return
    s1, pool = load_split(a.work, "train")
    _, npl, _ = load_norm(a.work, "train")
    emb = np.load(os.path.join(a.work, "emb", "train_pool.npy"), mmap_mode="r")
    own, k = owner_index(s1, pool), s1.k.values.astype(np.int64)
    prio, fold, cty = s1.prio.values, s1.fold.values, s1.country.values
    F = pd.read_parquet(os.path.join(a.work, "feats", "train.parquet"))
    inA = (prio >= C.TRAIN_A[0]) & (prio < C.TRAIN_A[1])
    inB = (prio >= C.TRAIN_B[0]) & (prio < C.TRAIN_B[1])
    tune, hold = inB & (fold < 3), inB & (fold >= 3)
    FA, FB = F[inA[F.qi.values]], F[inB[F.qi.values]].copy()
    del F
    cols1 = C.FULL + (C.PAIR_V2 if C.USE_V2 else [])
    rep = {"A_s1": int(inA.sum()), "B_s1": int(inB.sum()), "A_pairs": len(FA), "B_pairs": len(FB), "cols1": cols1}
    log(f"A: {rep['A_s1']:,} S1 / {len(FA):,} pairs (pos {FA.label.mean():.2%}); B: {rep['B_s1']:,} S1 / "
        f"{len(FB):,} pairs; full matcher uses {len(cols1)} features")

    log("fitting cheap re-ranker and full matcher on A")
    cheap = fit_lgb(FA[C.CHEAP].values.astype(np.float32), FA.label.values, FA.qi.values, C.LGB_PARAMS, a.n_jobs)
    full = fit_lgb(FA[cols1].values.astype(np.float32), FA.label.values, FA.qi.values, C.LGB_FULL, a.n_jobs)
    del FA
    cheap.save_model(os.path.join(mdir, "cheap.lgb"), num_iteration=cheap.best_iteration)
    full.save_model(os.path.join(mdir, "full.lgb"), num_iteration=full.best_iteration)

    rep["blocking_B"] = blocking_stats(FB.qi.values, FB.pj.values, own, k, inB)
    log(f"B blocking (test-like density): {rep['blocking_B']}")
    FB["pc"] = predict(cheap, FB[C.CHEAP].values.astype(np.float32))
    curve, t_cheap = [], C.CHEAP_THRESHOLDS[0]
    for t in C.CHEAP_THRESHOLDS:
        s = FB[FB.pc >= t]
        st = blocking_stats(s.qi.values, s.pj.values, own, k, inB)
        curve.append({"t": t, **st})
        if st["pair_recall"] >= rep["blocking_B"]["pair_recall"] - C.CHEAP_MAX_RECALL_LOSS:
            t_cheap = t
    rep["cheap_curve_B"], rep["t_cheap"] = curve, t_cheap
    sv = FB[FB.pc >= t_cheap].copy()
    del FB
    rep["candidates_B"] = blocking_stats(sv.qi.values, sv.pj.values, own, k, inB)
    log(f"cheap filter p >= {t_cheap}: {rep['candidates_B']}")

    sv["p"] = predict(full, sv[cols1].values.astype(np.float32))

    def choose(d, tag):
        """Pick the decision rule on folds 0-2, report every rule on folds 3-4."""
        from ber.model import prep_rules
        res = {}
        s = prep_rules(d)
        for r in C.RULES_TRY:
            params, f_tune = tune_rule(d, own, k, tune, r, prepared=s)
            kept = apply_rule(d, r, params, prepared=s)
            res[r] = {"params": list(params), "f05_tune": f_tune, **macro_f05(kept.qi.values, kept.pj.values, own, k,
                                                                               hold, cty)}
            log(f"  {tag} {r}{tuple(params)}: tune {f_tune:.4f}  holdout {res[r]['f05']:.4f}")
        best = max(res, key=lambda r: res[r]["f05_tune"])
        return best, res

    r1, res1 = choose(sv[["qi", "pj", "p"]], "stage 1")
    rep["stage1_B"] = res1
    log("stage 2 features + 5-fold stacking on B")
    d = stage2_frame(sv, pool, npl.nname.values, npl.naddr.values, emb, C.USE_S2X)
    cols2 = S2_COLS + cols1 + ([c for c in S2X_COLS if c in d] if C.USE_S2X else [])
    X, y, q = d[cols2].values.astype(np.float32), d.label.values, d.qi.values
    fq = fold[q]
    p2 = np.zeros(len(d), np.float32)
    for f in range(5):
        te = fq == f
        m = fit_lgb(X[~te], y[~te], q[~te], C.LGB_STAGE2, a.n_jobs)
        p2[te] = predict(m, X[te])
    d2 = d[["qi", "pj"]].assign(p=p2)
    r2, res2 = choose(d2, "stage 2")
    rep["stage2_B"] = res2
    use_s2 = res2[r2]["f05_tune"] >= res1[r1]["f05_tune"]
    rule, dd = (r2, d2) if use_s2 else (r1, sv[["qi", "pj", "p"]])
    params, f_all = tune_rule(dd, own, k, inB, rule)          # final parameters from all of B
    est = (res2 if use_s2 else res1)[rule]["f05"]
    rep.update(use_stage2=bool(use_s2), rule=rule, params=list(params), f05_B_all=f_all, estimated_f05=est)
    m2 = fit_lgb(X, y, q, C.LGB_STAGE2, a.n_jobs)
    m2.save_model(os.path.join(mdir, "stage2.lgb"), num_iteration=m2.best_iteration)
    save_json({"t_cheap": t_cheap, "use_stage2": bool(use_s2), "rule": rule, "params": list(params),
               "cols1": cols1, "cols2": cols2, "use_s2x": bool(C.USE_S2X)}, os.path.join(mdir, "decision.json"))
    save_json(rep, os.path.join(a.work, "train_report.json"))
    log(f"TRAIN DONE. estimated F0.5 on test-like B holdout = {est:.4f} ({'stage 2' if use_s2 else 'stage 1'}, "
        f"rule {rule}{tuple(params)}); t_cheap={t_cheap}")


def write_lists(path, s1_ids, qi, ids, header):
    lists = pd.Series(ids).groupby(qi).agg(",".join) if len(qi) else pd.Series(dtype=str)
    col = pd.Series(s1_ids).index.map(lambda i: lists.get(i, ""))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for sid, v in zip(s1_ids, col):
            f.write(f"{sid}\t{v}\n")


def st_predict(a):
    import lightgbm as lgb
    from ber.feats import compute
    from ber.model import apply_rule
    mdir = os.path.join(a.work, "models")
    cheap = lgb.Booster(model_file=os.path.join(mdir, "cheap.lgb"))
    full = lgb.Booster(model_file=os.path.join(mdir, "full.lgb"))
    m2 = lgb.Booster(model_file=os.path.join(mdir, "stage2.lgb"))
    dec = load_json(os.path.join(mdir, "decision.json"))
    cols1 = dec.get("cols1", C.FULL)
    s1, pool, S, P, idf = split_arrays(a, "test")
    c = pd.read_parquet(os.path.join(a.work, "block", "test_cands.parquet"))
    log(f"test: {len(c):,} blocking pairs for {c.qi.nunique():,}/{len(s1):,} S1")
    cdir = os.path.join(a.work, "feats", "test_all")     # base features of ALL blocking pairs, cached per chunk
    os.makedirs(cdir, exist_ok=True)
    parts = []
    step = a.predict_chunk
    bounds = np.searchsorted(c.qi.values, np.arange(0, len(s1) + step, step))
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        if hi <= lo:
            continue
        path = os.path.join(cdir, f"part_{i:03d}_{lo}_{hi}.parquet")
        if os.path.exists(path):
            cc = pd.read_parquet(path)
        else:
            cc = c.iloc[lo:hi].copy()
            for f, v in compute(S, P, cc.qi.values, cc.pj.values, idf, a.feat_workers).items():
                cc[f] = v
            cc.to_parquet(path, index=False)
        cc["pc"] = cheap.predict(cc[C.CHEAP].values.astype(np.float32)).astype(np.float32)
        parts.append(cc[cc.pc >= dec["t_cheap"]])
        del cc
        log(f"  test pairs {hi:,}/{len(c):,}: kept {sum(len(p) for p in parts):,} after the cheap filter")
    sv = pd.concat(parts, ignore_index=True)
    del parts, c
    log(f"candidate set: {len(sv):,} pairs, {len(sv) / len(s1):.2f} per S1")
    if any(f in cols1 for f in C.PAIR_V2):
        tri = tri_tables(a, "test", S, P)
        for f, v in compute(S, P, sv.qi.values, sv.pj.values, idf, a.feat_workers, tri=tri).items():
            sv[f] = v
    del S
    import gc
    gc.collect()
    sv["p"] = full.predict(sv[cols1].values.astype(np.float32)).astype(np.float32)
    if dec.get("use_stage2", True):
        # stage 2 in S1 chunks: its per-S1 features only look at that S1's candidates, the per-pool-record ones
        # are computed once for everything first -> same result as one pass, a fraction of the peak memory
        from ber.model import add_pj_features
        emb = np.load(os.path.join(a.work, "emb", "test_pool.npy"), mmap_mode="r")
        sv = add_pj_features(sv).sort_values("qi", kind="stable").reset_index(drop=True)
        qv = sv.qi.values
        cuts = [0]
        for s in np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]]):
            if s - cuts[-1] >= 1_500_000:
                cuts.append(int(s))
        cuts.append(len(sv))
        outs = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            part = stage2_frame(sv.iloc[lo:hi].copy(), pool, P["nname"], P["naddr"], emb, dec.get("use_s2x", False))
            outs.append(part[["qi", "pj"]].assign(
                p=m2.predict(part[dec["cols2"]].values.astype(np.float32)).astype(np.float32)))
            del part
            gc.collect()
            log(f"  stage 2 {hi:,}/{len(sv):,}")
        d = pd.concat(outs, ignore_index=True)
    else:
        d = sv[["qi", "pj", "p"]]
    kept = apply_rule(d, dec.get("rule", "threshold"), dec.get("params", [dec.get("t_final", 0.5)]))
    kept = kept.sort_values(["qi", "p"], ascending=[True, False])
    sv = sv.sort_values(["qi", "cos"], ascending=[True, False])
    os.makedirs(a.out, exist_ok=True)
    ids1, idsp = s1.entity_id.values, pool.entity_id.values
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), ids1, sv.qi.values, idsp[sv.pj.values],
                "candidate_entity_ids")
    write_lists(os.path.join(a.out, "matching_results.tsv"), ids1, kept.qi.values, idsp[kept.pj.values],
                "matched_entity_ids")
    n_m = np.bincount(kept.qi.values, minlength=len(s1))
    log(f"wrote {a.out}: {len(kept):,} matches ({len(kept) / len(s1):.2f} per S1); S1 with >=1 match "
        f"{(n_m > 0).mean():.1%}; by country {pd.Series(n_m > 0).groupby(s1.country.values).mean().round(3).to_dict()}")
    val = os.path.join(a.data_dir, "..", "utils", "validate_submission.py")
    if os.path.exists(val):
        import subprocess
        r = subprocess.run([sys.executable, val, "--matching", os.path.join(a.out, "matching_results.tsv"),
                            "--candidate", os.path.join(a.out, "candidate_pairs.tsv"),
                            "--test-dir", os.path.join(a.data_dir, "test")])
        log(f"validator exit code {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=default(os.path.join(ROOT, "dataset"), "dataset"))
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    ap.add_argument("--stages", default="all", help="comma list of " + ",".join(STAGES) + " (default all)")
    ap.add_argument("--ft-model", default=None, help="use this fine-tuned encoder directory instead of training one")
    ap.add_argument("--n-jobs", type=int, default=24, help="CPU processes/threads for features and LightGBM")
    ap.add_argument("--norm-workers", type=int, default=16)
    ap.add_argument("--feat-workers", type=int, default=20,
                    help="pair-feature processes (each holds a copy of the IDF tables, ~1 GB)")
    ap.add_argument("--predict-chunk", type=int, default=300_000, help="test S1 rows per feature batch")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.work, "models"), exist_ok=True)
    todo = [s for s in STAGES if a.stages == "all" or s in a.stages.split(",")]
    if len(todo) > 1:
        # one fresh process per stage: memory held by a finished stage (models, 10M-row text lists, the CUDA
        # caching allocator) is returned to the OS before the next stage starts
        import subprocess
        argv = [x for i, x in enumerate(sys.argv[1:]) if x != "--stages" and (i == 0 or sys.argv[i] != "--stages")]
        for s in todo:
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--stages", s] + argv)
            if r.returncode != 0:
                sys.exit(f"stage {s} failed (exit {r.returncode}); fix and re-run -- finished stages are cached")
        return
    sys.stdout = Tee(os.path.join(a.work, "pipeline.log"))
    s = todo[0]
    log(f"stage={s} data={a.data_dir} work={a.work} out={a.out}")
    t0 = time.time()
    log(f"=== {s} ===")
    globals()[f"st_{s}"](a)
    log(f"=== {s} done in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
