"""Mining: frequent itemsets with Apriori, FP-Growth and ECLAT, then class association rules
X -> API_MINGGU_DEPAN ("fire next week").

Three algorithms, one answer. All three return the same frequent itemsets; they differ only in how they
search, which shows up in run time and memory:
- Apriori (mlxtend): breadth-first. Level k candidates are built from frequent (k-1)-itemsets and counted by
  scanning the whole transaction table.
- FP-Growth (mlxtend): compresses the transactions into an FP-tree once, then grows itemsets from conditional
  trees without generating candidates.
- ECLAT (own code below): vertical layout. Each item keeps the set of transactions that contain it (a bitset);
  the support of an itemset is the size of the intersection of its items' bitsets. Depth-first search.

Rules are mined on 2012 to 2019 only. Two rule sets answer the business questions:
- "fire":  all transactions, consequent API_MINGGU_DEPAN (questions 1 to 3).
- "quiet": only cells with no hotspot for 30 days (SENYAP_30HARI), the same consequent, i.e. new ignitions
           (question 4). Lift here is measured against the fire rate of quiet cells, not of all cells.

Rule measures (A = antecedent, F = API_MINGGU_DEPAN, N = number of transactions in the set):
  support(A -> F)    = count(A and F) / N
  confidence(A -> F) = count(A and F) / count(A)          = P(F | A)
  lift(A -> F)       = confidence / P(F)                   (> 1: fire more likely than an average week)
  improvement        = confidence - max confidence of every rule whose antecedent is a proper subset of A
                       (the empty antecedent gives P(F)). A rule with improvement <= MIN_IMPROVEMENT only
                       repeats a shorter rule with extra items, so it is dropped.

Output: results/benchmark.csv, results/benchmark.png, results/algorithm_check.csv,
        results/rules_fire.csv, results/rules_quiet.csv, results/mining_summary.json
Run:    python mine.py            (add --quick to skip the slow benchmark repeats)
"""
import gc, json, sys, time, tracemalloc
from itertools import combinations
from math import ceil
import numpy as np, pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from config import RESULTS, MINE_YEARS
from items import ITEMS, ITEM_TEXT, TARGET, load_transactions

MAX_LEN = 5                 # itemset size incl. the consequent, so at most 4 conditions per rule
MIN_LIFT = 1.5              # data mining success criterion (BOD)
MIN_IMPROVEMENT = 0.01      # a rule must beat each of its shorter sub-rules by at least 1 point of confidence
SETTINGS = {                # rule set -> minimum support used for the final rules
    "fire": 0.01,           # BOD: at least 1% of transactions
    "quiet": 0.005,         # smaller subset and a rarer event (about 7% fire), so 0.5% of the subset
}
BENCH_SUPPORTS = [0.05, 0.02, 0.01, 0.005]
QUIET_CONSTANT = ["SENYAP_30HARI", "TERBAKAR_7HARI", "TERBAKAR_8_30HARI", "HOTSPOT_7HARI_TINGGI"]


# ----------------------------------------------------------------------------------------------- algorithms
def run_apriori(X, min_sup, max_len=MAX_LEN):
    # low_memory: count candidates one by one instead of in one big array. Same itemsets, slower, but the
    # default needed about 1.5 GB on only 14,000 rows at 0.5% support, which would not fit for ~230,000 rows.
    return apriori(X, min_support=min_sup, use_colnames=True, max_len=max_len, low_memory=True)


def run_fpgrowth(X, min_sup, max_len=MAX_LEN):
    return fpgrowth(X, min_support=min_sup, use_colnames=True, max_len=max_len)


def run_eclat(X, min_sup, max_len=MAX_LEN):
    """ECLAT with tid-sets stored as Python integers used as bitsets (bit r set = transaction r has the item).
    Intersection is a bitwise AND, support is the number of set bits."""
    n = len(X)
    min_count = ceil(min_sup * n - 1e-9)
    tid = {}
    for c in X.columns:
        bits = np.packbits(X[c].to_numpy(dtype=bool), bitorder="little")
        v = int.from_bytes(bits.tobytes(), "little")
        k = v.bit_count()
        if k >= min_count:
            tid[c] = (v, k)
    out_sets, out_sup = [], []

    def dfs(prefix, cands):
        # cands: (item, bitset, count) that extend prefix and are frequent together with it
        for a, (item, bits, cnt) in enumerate(cands):
            new = prefix + (item,)
            out_sets.append(frozenset(new)); out_sup.append(cnt / n)
            if len(new) < max_len:
                nxt = []
                for item2, bits2, _ in cands[a + 1:]:
                    b = bits & bits2
                    k = b.bit_count()
                    if k >= min_count:
                        nxt.append((item2, b, k))
                if nxt:
                    dfs(new, nxt)

    start = sorted(tid, key=lambda c: tid[c][1])  # rarest first keeps the tid-sets small
    dfs((), [(c, *tid[c]) for c in start])
    return pd.DataFrame({"support": out_sup, "itemsets": out_sets})


ALGOS = {"Apriori": run_apriori, "FP-Growth": run_fpgrowth, "ECLAT": run_eclat}


def as_dict(freq):
    return dict(zip(freq.itemsets.map(frozenset), freq.support.astype(float)))


# ------------------------------------------------------------------------------------------ rule generation
def class_rules(freq, target=TARGET):
    """All rules A -> target from a dict {itemset: support}. Every subset of a frequent itemset is frequent,
    so the antecedent and all sub-rules can be looked up in the same dict."""
    base = freq[frozenset([target])]
    rows = []
    for s, sup in freq.items():
        if target not in s or len(s) < 2:
            continue
        a = s - {target}
        conf = sup / freq[a]
        best_sub = base  # empty antecedent
        for k in range(1, len(a)):
            for sub in combinations(a, k):
                sub = frozenset(sub)
                best_sub = max(best_sub, freq[sub | {target}] / freq[sub])
        rows.append({"antecedent": " & ".join(sorted(a)), "n_items": len(a), "support": sup,
                     "antecedent_support": freq[a], "confidence": conf, "lift": conf / base,
                     "improvement": conf - best_sub})
    r = pd.DataFrame(rows)
    return r.sort_values(["lift", "support"], ascending=False).reset_index(drop=True), base


def sentence(antecedent, conf, lift):
    cond = ", ".join(ITEM_TEXT[i] for i in antecedent.split(" & "))
    return f"Jika {cond}, maka peluang ada hotspot minggu depan {conf:.0%} ({lift:.1f} kali rata-rata)."


def transactions_for(tx, which):
    """Item table (bool) of the mining years for a rule set."""
    t = tx[tx.year.isin(MINE_YEARS)]
    cols = list(ITEMS)
    if which == "quiet":
        t = t[t.SENYAP_30HARI]
        cols = [c for c in cols if c not in QUIET_CONSTANT]
    return t[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- benchmark
def benchmark(X, supports=BENCH_SUPPORTS, repeats=3):
    rows, checks = [], []
    for s in supports:
        results = {}
        for name, fn in ALGOS.items():
            times = []
            for _ in range(repeats):
                gc.collect()
                t0 = time.perf_counter()
                f = fn(X, s)
                times.append(time.perf_counter() - t0)
            gc.collect()
            tracemalloc.start()
            fn(X, s)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            results[name] = as_dict(f)
            rows.append({"min_support": s, "algorithm": name, "seconds_median": float(np.median(times)),
                         "seconds_min": float(np.min(times)), "peak_memory_mb": peak / 2**20,
                         "n_itemsets": len(f), "n_rules_to_target": int(sum(TARGET in k and len(k) > 1 for k in results[name]))})
            print(f"  min_support {s:.3f} {name:10s} {np.median(times):7.2f} s  {peak / 2**20:7.1f} MB  {len(f):6d} itemsets", flush=True)
        ref = results["Apriori"]
        for name in ("FP-Growth", "ECLAT"):
            other = results[name]
            same_sets = set(ref) == set(other)
            max_diff = max((abs(ref[k] - other[k]) for k in ref if k in other), default=0.0)
            checks.append({"min_support": s, "compared": f"Apriori vs {name}", "same_itemsets": same_sets,
                           "max_support_difference": max_diff, "n_itemsets": len(other)})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def plot_benchmark(b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, g in b.groupby("algorithm", sort=False):
        ax[0].plot(g.min_support * 100, g.seconds_median, marker="o", label=name)
        ax[1].plot(g.min_support * 100, g.peak_memory_mb, marker="o", label=name)
    for a, lab in zip(ax, ["Run time (s, median of repeats)", "Peak memory (MB, tracemalloc)"]):
        a.set_xscale("log"); a.invert_xaxis(); a.set_xlabel("Minimum support (%)"); a.set_ylabel(lab)
        a.set_xticks([5, 2, 1, 0.5]); a.set_xticklabels(["5", "2", "1", "0.5"]); a.grid(alpha=.3)
    ax[0].legend()
    fig.suptitle("Apriori vs FP-Growth vs ECLAT on 2012-2019 transactions (max 5 items)")
    fig.tight_layout(); fig.savefig(RESULTS / "benchmark.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------------------------------- main
def main():
    quick = "--quick" in sys.argv
    RESULTS.mkdir(exist_ok=True)
    tx = load_transactions()
    summary = {}

    X = transactions_for(tx, "fire")
    print(f"benchmark on {len(X):,} transactions x {X.shape[1]} items (2012-2019)")
    b, chk = benchmark(X, repeats=1 if quick else 3)
    b.to_csv(RESULTS / "benchmark.csv", index=False)
    chk.to_csv(RESULTS / "algorithm_check.csv", index=False)
    plot_benchmark(b)
    print(chk.to_string(index=False))
    assert chk.same_itemsets.all() and (chk.max_support_difference < 1e-9).all(), "algorithms disagree"

    for which, min_sup in SETTINGS.items():
        X = transactions_for(tx, which)
        per_algo = {}
        for name, fn in ALGOS.items():
            rules, base = class_rules(as_dict(fn(X, min_sup)))
            per_algo[name] = rules
        same_rules = all(set(per_algo[n].antecedent) == set(per_algo["Apriori"].antecedent) for n in per_algo)
        rules = per_algo["FP-Growth"]  # identical for all three (checked above)
        keep = (rules.lift >= MIN_LIFT) & (rules.improvement >= MIN_IMPROVEMENT)
        out = rules[keep].reset_index(drop=True)
        out.insert(0, "rule_id", [f"{which[0].upper()}{k + 1:03d}" for k in range(len(out))])
        out["sentence"] = [sentence(a, c, l) for a, c, l in zip(out.antecedent, out.confidence, out.lift)]
        out.to_csv(RESULTS / f"rules_{which}.csv", index=False)
        summary[which] = {"transactions": len(X), "items": X.shape[1], "min_support": min_sup, "base_rate": base,
                          "rules_to_target": len(rules), "rules_lift_ge_1_5": int((rules.lift >= MIN_LIFT).sum()),
                          "rules_kept_after_improvement": len(out), "same_rules_all_algorithms": bool(same_rules)}
        print(f"\n[{which}] {len(X):,} transactions, base rate {base:.3f}: {len(rules)} rules -> "
              f"{summary[which]['rules_lift_ge_1_5']} with lift >= {MIN_LIFT} -> {len(out)} after improvement pruning"
              f" | same rules in all algorithms: {same_rules}")
        print(out.head(15)[["rule_id", "antecedent", "support", "confidence", "lift", "improvement"]].round(3).to_string(index=False))
    (RESULTS / "mining_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
