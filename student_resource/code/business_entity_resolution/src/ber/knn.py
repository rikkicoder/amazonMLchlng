"""Blocking: exact nearest neighbours on the GPU inside each country, forward (S1 -> pool) and reverse (pool -> S1).

Scores are computed in fp16 on tensor cores to find the top (k + extra) per query, then re-scored exactly in fp32
and re-ranked, so the result equals an exact fp32 search up to ties. No approximate index is involved."""
import numpy as np
import pandas as pd

from .util import log


def topk_gpu(Q, P, k, extra=8, q_chunk=1024, p_block=250_000):
    """Q (n x d), P (m x d): L2-normalised float16 arrays. Returns (idx n x k into P, -1 = none; scores fp32)."""
    import torch
    n, m = len(Q), len(P)
    k = min(k, m)
    idx = np.full((n, max(k, 0)), -1, np.int64)
    sc = np.zeros((n, max(k, 0)), np.float32)
    if n == 0 or m == 0:
        return idx, sc
    kk = min(k + extra, m)
    dev = "cuda"
    Pg = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float16)).to(dev)
    blocks = [(s, min(s + p_block, m)) for s in range(0, m, p_block)]
    for s in range(0, n, q_chunk):
        q = torch.from_numpy(np.ascontiguousarray(Q[s:s + q_chunk], dtype=np.float16)).to(dev)
        vs, js = [], []
        for a, b in blocks:
            v, j = torch.topk(q @ Pg[a:b].T, min(kk, b - a), dim=1)
            vs.append(v)
            js.append(j + a)
        v, o = torch.topk(torch.cat(vs, 1), kk, dim=1)
        j = torch.gather(torch.cat(js, 1), 1, o)
        exact = torch.einsum("nd,nkd->nk", q.float(), Pg[j].float())       # exact fp32 re-scoring
        v, o = torch.topk(exact, k, dim=1)
        idx[s:s + len(q)] = torch.gather(j, 1, o).cpu().numpy()
        sc[s:s + len(q)] = v.cpu().numpy()
    del Pg
    torch.cuda.empty_cache()
    return idx, sc


def block(E1, Ep, s1_country, pool_country, k_fwd, m_rev, fwd_rows=None, rev_mask=None):
    """Union of forward top-k_fwd and reverse top-m_rev neighbours inside each country.
    fwd_rows: S1 rows that need candidates (default all).
    rev_mask: S1 rows that exist as reverse-search targets (default all). Training excludes a slice of S1 so that
    their S2/S3 records become ownerless, matching the test set's higher share of pool records without an S1.
    Returns (cands[qi, pj, cos, fwd_rank, rev_rank], q_top1, p_top1)."""
    n1, n2 = len(s1_country), len(pool_country)
    need = np.ones(n1, bool) if fwd_rows is None else np.isin(np.arange(n1), fwd_rows)
    present = np.ones(n1, bool) if rev_mask is None else np.asarray(rev_mask, bool)
    q_top1 = np.zeros(n1, np.float32)
    p_top1 = np.zeros(n2, np.float32)
    fwd, rev = [], []
    for c in pd.unique(s1_country):
        qa = np.where((s1_country == c) & present)[0]
        qn = qa[need[qa]]
        pi = np.where(pool_country == c)[0]
        if len(pi) == 0 or len(qn) == 0:
            continue
        log(f"  [{c}] forward {len(qn):,} x {len(pi):,}")
        fi, fs = topk_gpu(E1[qn], Ep[pi], k_fwd)
        q_top1[qn] = fs[:, 0]
        ok = fi >= 0
        fwd.append(pd.DataFrame({"qi": np.repeat(qn, fi.shape[1])[ok.ravel()], "pj": pi[fi[ok]],
                                 "fwd_rank": np.tile(np.arange(fi.shape[1]), len(qn))[ok.ravel()].astype(np.float32),
                                 "cos": fs[ok]}))
        log(f"  [{c}] reverse {len(pi):,} x {len(qa):,}")
        ri, rs = topk_gpu(Ep[pi], E1[qa], m_rev)
        p_top1[pi] = rs[:, 0]
        ok = ri >= 0
        d = pd.DataFrame({"pj": np.repeat(pi, ri.shape[1])[ok.ravel()], "qi": qa[ri[ok]],
                          "rev_rank": np.tile(np.arange(ri.shape[1]), len(pi))[ok.ravel()].astype(np.float32),
                          "cos_r": rs[ok]})
        rev.append(d[need[d.qi.values]])
    f = pd.concat(fwd, ignore_index=True) if fwd else pd.DataFrame(columns=["qi", "pj", "fwd_rank", "cos"])
    r = pd.concat(rev, ignore_index=True) if rev else pd.DataFrame(columns=["pj", "qi", "rev_rank", "cos_r"])
    c = f.merge(r, on=["qi", "pj"], how="outer")
    c["cos"] = c["cos"].fillna(c["cos_r"]).astype(np.float32)
    c = c.drop(columns="cos_r").astype({"qi": np.int64, "pj": np.int64})
    return c[["qi", "pj", "cos", "fwd_rank", "rev_rank"]].reset_index(drop=True), q_top1, p_top1


def add_context(c, q_top1, p_top1, pool_src):
    c["gap_q"] = (q_top1[c.qi.values] - c.cos.values).astype(np.float32)
    c["rev_margin"] = (p_top1[c.pj.values] - c.cos.values).astype(np.float32)
    c["mutual"] = ((c.fwd_rank == 0) & (c.rev_rank == 0)).astype(np.float32)
    c["n_cand_q"] = c.groupby("qi").pj.transform("size").astype(np.float32)
    c["is_s3"] = (pool_src[c.pj.values] == "S3").astype(np.float32)
    return c
