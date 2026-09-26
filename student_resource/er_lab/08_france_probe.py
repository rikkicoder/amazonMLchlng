#!/usr/bin/env python3
"""08 -- France probe (label-free): does our pipeline behave on the unseen country like on US / India?

There are no labels for the test set, so we use label-free signals, computed per test country on a sample of
S1 records searched against the FULL test pool of that country (HashingVectorizer TF-IDF: no vocabulary needed,
works at any scale -- the answer to "how do we do TF-IDF without knowing the vocabulary"):
  * top-1 cosine and top1-top2 gap distributions
  * house-number agreement between the S1 and its top-1 / top-5 neighbours  (pseudo-precision)
  * how many neighbours look like matches (cos >= 0.5)  vs the ~3.5 matches per S1 seen in train
Compared across normalisation levels (L1 vs L4 -- L4 French canonical forms are untested by labels).
Also prints a few France S1 records with their nearest neighbours so we can eyeball the noise.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from lab.common import base_parser, log_result, setup, step
from lab.data import load_cached
from lab.knn import topk_sparse
from lab.text import load_dict, normalize_series, numbers


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=3000, help="S1 records sampled per country")
    ap.add_argument("--levels", default="1,4")
    ap.add_argument("--countries", default="", help="default: all test countries")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()
    setup(args, "08_france_probe")
    s1 = load_cached(args, "test_s1")
    pool = pd.concat([load_cached(args, "test_s2"), load_cached(args, "test_s3")], ignore_index=True)
    dct = load_dict(os.path.join(args.work, "models", "translit_dict.json"))
    countries = args.countries.split(",") if args.countries else sorted(s1.country.unique())
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 21, alternate_sign=False,
                           norm=None, dtype=np.float32)
    for cty in countries:
        q = s1[s1.country == cty].sample(min(args.n, int((s1.country == cty).sum())), random_state=args.seed)
        p = pool[pool.country == cty].reset_index(drop=True)
        qn_nums = [numbers(a) for a in q.addr.values]
        pn_nums = [numbers(a) for a in p.addr.values]
        for lvl in (int(x) for x in args.levels.split(",")):
            t0 = time.time()
            qt = [a + " " + b for a, b in zip(normalize_series(q.name.values, q.country.values, lvl, "name", dct),
                                              normalize_series(q.addr.values, q.country.values, lvl, "addr", dct))]
            pt = [a + " " + b for a, b in zip(normalize_series(p.name.values, p.country.values, lvl, "name", dct),
                                              normalize_series(p.addr.values, p.country.values, lvl, "addr", dct))]
            Hp = hv.transform(pt)
            tf = TfidfTransformer(sublinear_tf=True).fit(Hp[np.random.default_rng(0).choice(Hp.shape[0],
                                                                                            min(500_000, Hp.shape[0]),
                                                                                            replace=False)])
            P = tf.transform(Hp).astype(np.float32)
            Q = tf.transform(hv.transform(qt)).astype(np.float32)
            idx, sc = topk_sparse(Q, P, 10, args.n_jobs)
            secs = time.time() - t0
            agree1, agree5, has_num = [], [], []
            for i in range(len(q)):
                a = qn_nums[i]
                if not a:
                    continue
                has_num.append(1)
                j = idx[i, 0]
                agree1.append(float(j >= 0 and bool(a & pn_nums[j])))
                top = [jj for jj in idx[i, :5] if jj >= 0]
                agree5.append(np.mean([bool(a & pn_nums[jj]) for jj in top]) if top else 0.0)
            gap = sc[:, 0] - sc[:, 1]
            qs = lambda x: [round(float(v), 3) for v in np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])]
            log_result(args, "08_country_probe", f"{cty}|L{lvl}", pool=len(p), seconds=secs,
                       top1_cos_q=qs(sc[:, 0]), gap_q=qs(gap),
                       num_agree_top1=float(np.mean(agree1)) if agree1 else None,
                       num_agree_top5=float(np.mean(agree5)) if agree5 else None,
                       mean_neighbours_cos_ge_05=float((sc >= 0.5).sum(1).mean()),
                       mean_neighbours_cos_ge_07=float((sc >= 0.7).sum(1).mean()),
                       s1_with_numbers=float(len(has_num) / len(q)))
            if lvl == max(int(x) for x in args.levels.split(",")) and args.show:
                print(f"\n--- {cty}: sample S1 records and their top-4 neighbours (L{lvl}) ---")
                for i in range(min(args.show, len(q))):
                    r = q.iloc[i]
                    print(f"  S1 {r.entity_id} {r['name'][:55]} | {r.addr[:90]}")
                    for jj, s in zip(idx[i, :4], sc[i, :4]):
                        if jj >= 0:
                            pr = p.iloc[jj]
                            print(f"      {s:.3f} {pr.entity_id} {pr['name'][:55]} | {pr.addr[:90]}")


if __name__ == "__main__":
    main()
