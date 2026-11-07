"""Evaluation of the mined rules on years they were not mined from.

Why holdout years and not significance tests: neighbouring cells and consecutive weeks are not independent,
so with about 286,000 rows almost any rule gets a tiny p-value. Instead each rule mined on 2012 to 2019 is
re-measured on 2020 to 2026, and per year. Uncertainty comes from a block bootstrap that resamples whole
weeks (all cells of a week stay together), so the dependence between cells in the same week is kept.

What is evaluated
1. Each rule: support, confidence and lift on the mining years and on the holdout years 2020-2026 (also split
   2020-2022 and 2023-2026), lift per year, a 95% block-bootstrap interval of the holdout lift, and lift in
   El Nino versus other years.
2. Success criteria from the BOD: train lift >= 1.5, train support >= the minimum support, holdout lift
   stays >= 1.5 (or at least close to the train lift).
3. Rule card: up to 10 non-overlapping rules. A cell-week is flagged when any card rule matches. Precision,
   recall and share of flagged cell-weeks on the holdout years, against simple baselines and random.
4. Comparison with a decision tree trained on the same items (its leaf paths are also if-then rules).

Output: results/eval_rules_{fire,quiet}.csv, results/lift_per_year_{fire,quiet}.csv, results/rule_card_{...}.csv,
        results/card_vs_baselines.csv, results/tree_rules_{...}.csv, results/tree_{...}.txt,
        results/lift_train_vs_holdout.png, results/lift_per_year_card.png, results/summary.md
Run:    python evaluate_rules.py
"""
import json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from config import RESULTS, MINE_YEARS, EL_NINO_YEARS, SEED
from items import ITEMS, ITEM_TEXT, TARGET, load_transactions
from mine import MIN_LIFT, SETTINGS, QUIET_CONSTANT

HOLDOUT = list(range(2020, 2027))
CARD_SIZE = 10
N_BOOT = 1000
SPLITS = {"2012-2019": MINE_YEARS, "2020-2026": HOLDOUT, "2020-2022": [2020, 2021, 2022], "2023-2026": [2023, 2024, 2025, 2026]}


def rule_mask(tx, antecedent):
    m = np.ones(len(tx), bool)
    for it in antecedent.split(" & "):
        m &= tx[it].to_numpy()
    return m


def measures(m, y):
    """support, confidence, lift, coverage of one rule mask m on labels y."""
    n, base = len(y), y.mean()
    cov = m.sum()
    both = (m & y).sum()
    conf = both / cov if cov else np.nan
    return {"support": both / n, "coverage": cov / n, "confidence": conf, "lift": conf / base if base else np.nan,
            "n_covered": int(cov)}


def block_bootstrap(masks, y, week_codes, n_boot=N_BOOT, seed=SEED):
    """95% intervals of confidence and lift for many rules at once, resampling whole weeks."""
    W = week_codes.max() + 1
    A = np.stack([np.bincount(week_codes, weights=m, minlength=W) for m in masks], 1)            # covered per week
    B = np.stack([np.bincount(week_codes, weights=m & y, minlength=W) for m in masks], 1)        # covered & fire
    N = np.bincount(week_codes, minlength=W).astype(float)
    P = np.bincount(week_codes, weights=y, minlength=W)
    rng = np.random.default_rng(seed)
    C = rng.multinomial(W, np.full(W, 1 / W), size=n_boot).astype(float)  # how often each week is drawn
    conf = (C @ B) / np.maximum(C @ A, 1)
    lift = conf / ((C @ P) / (C @ N))[:, None]
    q = lambda x, p: np.nanpercentile(x, p, axis=0)
    return q(conf, 2.5), q(conf, 97.5), q(lift, 2.5), q(lift, 97.5)


def evaluate_set(tx, rules, which):
    pop = tx[tx.SENYAP_30HARI] if which == "quiet" else tx
    pop = pop.reset_index(drop=True)
    y = pop.y.to_numpy().astype(bool)
    masks = [rule_mask(pop, a) for a in rules.antecedent]
    rows, per_year = [], []
    years = sorted(pop.year.unique())
    year_idx = {yr: (pop.year == yr).to_numpy() for yr in years}
    split_idx = {k: pop.year.isin(v).to_numpy() for k, v in SPLITS.items()}
    # sensitivity check: holdout rows built only from archived (SP, "final") hotspots, i.e. without Aug-Oct 2026 NRT
    split_idx["2020-2026_final"] = split_idx["2020-2026"] & (pop.quality == "final").to_numpy()
    nino = pop.year.isin(EL_NINO_YEARS).to_numpy()
    for r, m in zip(rules.itertuples(), masks):
        row = {"rule_id": r.rule_id, "antecedent": r.antecedent, "n_items": r.n_items}
        for k, idx in split_idx.items():
            for name, v in measures(m[idx], y[idx]).items():
                row[f"{name}_{k}"] = v
        row["lift_el_nino_years"] = measures(m[nino], y[nino])["lift"]
        row["lift_other_years"] = measures(m[~nino], y[~nino])["lift"]
        ly = {yr: measures(m[year_idx[yr]], y[year_idx[yr]])["lift"] for yr in years}
        per_year.append({"rule_id": r.rule_id, "antecedent": r.antecedent, **ly})
        hold = [ly[yr] for yr in HOLDOUT if yr in ly and not np.isnan(ly[yr])]
        row["holdout_years_lift_ge_1"] = int(sum(v >= 1 for v in hold))
        row["holdout_years_lift_ge_1_5"] = int(sum(v >= MIN_LIFT for v in hold))
        row["holdout_years_measured"] = len(hold)
        rows.append(row)
    ev = pd.DataFrame(rows)

    hidx = split_idx["2020-2026"]
    wk = pd.factorize(pop.week[hidx])[0]
    c_lo, c_hi, l_lo, l_hi = block_bootstrap([m[hidx] for m in masks], y[hidx], wk)
    ev["confidence_2020-2026_ci_low"], ev["confidence_2020-2026_ci_high"] = c_lo, c_hi
    ev["lift_2020-2026_ci_low"], ev["lift_2020-2026_ci_high"] = l_lo, l_hi
    ev["lift_ratio_holdout_vs_train"] = ev["lift_2020-2026"] / ev["lift_2012-2019"]
    ev["pass_train"] = (ev["lift_2012-2019"] >= MIN_LIFT) & (ev["support_2012-2019"] >= SETTINGS[which])
    ev["pass_holdout"] = (ev["lift_2020-2026"] >= MIN_LIFT) & (ev["lift_2020-2026_ci_low"] > 1)
    ev["stable"] = ev.pass_train & ev.pass_holdout & (ev.holdout_years_lift_ge_1 >= ev.holdout_years_measured - 1)
    return ev, pd.DataFrame(per_year), pop, masks


# --------------------------------------------------------------------------------------------------- card
def pick_card(ev, masks, train_idx, size=CARD_SIZE, max_jaccard=0.7):
    """Greedy: highest train lift first, skip a rule whose antecedent contains a chosen one or whose covered
    cell-weeks overlap a chosen rule's by more than max_jaccard (it would flag almost the same places)."""
    order = ev[ev.pass_train].sort_values(["lift_2012-2019", "support_2012-2019"], ascending=False).index
    chosen = []
    for k in order:
        a = set(ev.antecedent[k].split(" & "))
        mk = masks[k][train_idx]
        ok = True
        for c in chosen:
            if set(ev.antecedent[c].split(" & ")) <= a:
                ok = False; break
            mc = masks[c][train_idx]
            if (mk & mc).sum() / max((mk | mc).sum(), 1) > max_jaccard:
                ok = False; break
        if ok:
            chosen.append(k)
        if len(chosen) == size:
            break
    return chosen


def flag_scores(flag, y):
    tp = (flag & y).sum()
    return {"flagged_share": flag.mean(), "precision": tp / flag.sum() if flag.sum() else np.nan,
            "recall": tp / y.sum(), "lift": (tp / flag.sum()) / y.mean() if flag.sum() else np.nan}


# --------------------------------------------------------------------------------------------------- tree
def tree_rules(pop, train_idx, cols, max_depth=4):
    """Decision tree on the same boolean items. Every leaf is a rule: a path of 'item' or 'NOT item' conditions."""
    Xtr, ytr = pop.loc[train_idx, cols].to_numpy(), pop.y[train_idx].to_numpy()
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(int(0.01 * len(ytr)), 1), random_state=SEED)
    clf.fit(Xtr, ytr)
    t = clf.tree_
    paths = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            paths.append((node, conds)); return
        f = cols[t.feature[node]]
        walk(t.children_left[node], conds + [f"NOT {f}"])   # bool feature <= 0.5 -> false
        walk(t.children_right[node], conds + [f])
    walk(0, [])
    leaf_all = clf.apply(pop[cols].to_numpy())
    return clf, paths, leaf_all, export_text(clf, feature_names=cols)


def evaluate_tree(pop, which, cols):
    tr = pop.year.isin(MINE_YEARS).to_numpy()
    ho = pop.year.isin(HOLDOUT).to_numpy()
    y = pop.y.to_numpy().astype(bool)
    clf, paths, leaf, text = tree_rules(pop, tr, cols)
    rows = []
    for node, conds in paths:
        m = leaf == node
        a, b = measures(m[tr], y[tr]), measures(m[ho], y[ho])
        rows.append({"leaf": int(node), "path": " & ".join(conds), "n_conditions": len(conds),
                     "n_positive_conditions": sum(not c.startswith("NOT ") for c in conds),
                     "support_2012-2019": a["support"], "coverage_2012-2019": a["coverage"],
                     "confidence_2012-2019": a["confidence"], "lift_2012-2019": a["lift"],
                     "confidence_2020-2026": b["confidence"], "lift_2020-2026": b["lift"],
                     "coverage_2020-2026": b["coverage"]})
    tr_rules = pd.DataFrame(rows).sort_values("lift_2012-2019", ascending=False).reset_index(drop=True)
    tr_rules.to_csv(RESULTS / f"tree_rules_{which}.csv", index=False)
    (RESULTS / f"tree_{which}.txt").write_text(text)
    fire_leaves = tr_rules.loc[tr_rules["lift_2012-2019"] >= MIN_LIFT, "leaf"].to_numpy()
    flag = np.isin(leaf, fire_leaves)
    used = sorted({c.replace("NOT ", "") for p in tr_rules.path for c in p.split(" & ")})
    return tr_rules, flag, used


# --------------------------------------------------------------------------------------------------- plots
def plots(evs, per_year_card):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (which, ev) in zip(axes, evs.items()):
        c = np.where(ev.stable, "tab:green", "tab:red")
        ax.scatter(ev["lift_2012-2019"], ev["lift_2020-2026"], s=12, c=c, alpha=.6)
        lim = [0, max(ev["lift_2012-2019"].max(), ev["lift_2020-2026"].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="same lift")
        ax.axhline(MIN_LIFT, color="grey", lw=1, ls=":", label="lift 1.5")
        ax.set_xlabel("Lift 2012-2019 (mining)"); ax.set_ylabel("Lift 2020-2026 (holdout)")
        ax.set_title(f"{which} rules: green = stable, red = not"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "lift_train_vs_holdout.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(len(per_year_card), 1, figsize=(12, 3.2 * len(per_year_card)))
    for ax, (which, py) in zip(np.atleast_1d(axes), per_year_card.items()):
        years = [c for c in py.columns if isinstance(c, (int, np.integer))]
        M = py[years].to_numpy(dtype=float)
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(4, np.nanmax(M)))
        ax.set_xticks(range(len(years))); ax.set_xticklabels(years)
        ax.set_yticks(range(len(py))); ax.set_yticklabels(py.rule_id)
        ax.axvline(years.index(2020) - .5, color="k", lw=2)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if not np.isnan(M[a, b]):
                    ax.text(b, a, f"{M[a, b]:.1f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Lift per year of the {which} rule card (left of the black line: mining years)")
        fig.colorbar(im, ax=ax, label="lift")
    fig.tight_layout(); fig.savefig(RESULTS / "lift_per_year_card.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------------------------------- main
def md_table(df, floatfmt=3):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in d.itertuples(index=False))


def main():
    tx = load_transactions()
    evs, cards, py_card, comp, tree_info = {}, {}, {}, [], {}
    for which in SETTINGS:
        rules = pd.read_csv(RESULTS / f"rules_{which}.csv")
        ev, per_year, pop, masks = evaluate_set(tx, rules, which)
        ev = ev.merge(rules[["rule_id", "improvement", "sentence"]], on="rule_id")
        ev.to_csv(RESULTS / f"eval_rules_{which}.csv", index=False)
        per_year.to_csv(RESULTS / f"lift_per_year_{which}.csv", index=False)
        evs[which] = ev

        tr = pop.year.isin(MINE_YEARS).to_numpy()
        ho = pop.year.isin(HOLDOUT).to_numpy()
        y = pop.y.to_numpy().astype(bool)
        chosen = pick_card(ev, masks, tr)
        card = ev.loc[chosen].reset_index(drop=True)
        card["sentence_holdout"] = [
            f"Jika {', '.join(ITEM_TEXT[i] for i in a.split(' & '))}, maka peluang ada hotspot minggu depan "
            f"{c:.0%} ({l:.1f} kali rata-rata{' sel senyap' if which == 'quiet' else ''}; diuji 2020-2026)."
            for a, c, l in zip(card.antecedent, card["confidence_2020-2026"], card["lift_2020-2026"])]
        card.to_csv(RESULTS / f"rule_card_{which}.csv", index=False)
        cards[which] = card
        py_card[which] = per_year.set_index("rule_id").loc[card.rule_id].reset_index()

        card_flag = np.zeros(len(pop), bool)
        for k in chosen:
            card_flag |= masks[k]
        cols = [c for c in ITEMS if c != TARGET and not (which == "quiet" and c in QUIET_CONSTANT)]
        tr_rules, tree_flag, used = evaluate_tree(pop, which, cols)
        tree_info[which] = {"n_leaves": len(tr_rules), "n_fire_leaves": int((tr_rules["lift_2012-2019"] >= MIN_LIFT).sum()),
                            "items_used": used, "card_items": sorted({i for a in card.antecedent for i in a.split(" & ")})}
        best = ev[ev.pass_train].sort_values("lift_2012-2019", ascending=False).iloc[0]
        base_items = ["TERBAKAR_7HARI", "PERNAH_TERBAKAR_BULAN_INI"] if which == "fire" else ["PERNAH_TERBAKAR_BULAN_INI", "TETANGGA_TERBAKAR_7HARI"]
        methods = {f"ARM rule card ({len(chosen)} rules)": card_flag,
                   f"Decision tree, leaves with lift >= {MIN_LIFT}": tree_flag,
                   f"Best single ARM rule ({best.rule_id})": masks[best.name]}
        for b in base_items:
            methods[f"Baseline: {b} only"] = pop[b].to_numpy()
        for name, f in methods.items():
            comp.append({"rule_set": which, "method": name, **flag_scores(f[ho], y[ho])})
        comp.append({"rule_set": which, "method": "Random (same share as card)", "flagged_share": card_flag[ho].mean(),
                     "precision": y[ho].mean(), "recall": card_flag[ho].mean(), "lift": 1.0})
    comp = pd.DataFrame(comp)
    comp.to_csv(RESULTS / "card_vs_baselines.csv", index=False)
    plots(evs, py_card)

    # ------------------------------------------------------------------ printed and markdown summary
    mining = json.loads((RESULTS / "mining_summary.json").read_text())
    bench = pd.read_csv(RESULTS / "benchmark.csv")
    check = pd.read_csv(RESULTS / "algorithm_check.csv")
    overview = pd.read_csv(RESULTS / "data_overview.csv")
    md = ["# Results summary (generated by evaluate_rules.py)\n", "## Data\n", md_table(overview)]
    md += ["\n## Algorithm comparison (2012-2019 transactions)\n",
           md_table(bench[["min_support", "algorithm", "seconds_median", "peak_memory_mb", "n_itemsets", "n_rules_to_target"]], 2),
           "\n", md_table(check, 6)]
    for which, ev in evs.items():
        m = mining[which]
        md += [f"\n## Rule set `{which}`\n",
               f"- Transactions (2012-2019): {m['transactions']:,}; base rate of fire next week: {m['base_rate']:.3f}\n",
               f"- Rules X -> {TARGET}: {m['rules_to_target']}; with lift >= {MIN_LIFT}: {m['rules_lift_ge_1_5']}; "
               f"kept after improvement pruning: {m['rules_kept_after_improvement']}; same in all algorithms: {m['same_rules_all_algorithms']}\n",
               f"- Pass train criteria: {int(ev.pass_train.sum())}; pass holdout: {int(ev.pass_holdout.sum())}; stable: {int(ev.stable.sum())} "
               f"({ev.stable.mean():.0%})\n",
               f"- Median lift 2012-2019: {ev['lift_2012-2019'].median():.2f}; median lift 2020-2026: {ev['lift_2020-2026'].median():.2f}\n",
               "\n### Rule card\n",
               md_table(cards[which][["rule_id", "antecedent", "support_2012-2019", "confidence_2012-2019", "lift_2012-2019",
                                      "confidence_2020-2026", "lift_2020-2026", "lift_2020-2026_ci_low", "lift_2020-2026_ci_high",
                                      "holdout_years_lift_ge_1", "holdout_years_measured", "lift_el_nino_years", "lift_other_years", "stable"]], 2),
               "\n", "\n".join(f"- **{r}**: {s}" for r, s in zip(cards[which].rule_id, cards[which].sentence_holdout)), "\n",
               "\n### Lift per year of the card rules\n", md_table(py_card[which].drop(columns="antecedent"), 2),
               f"\n### Decision tree (depth 4) on the same items\n- Leaves: {tree_info[which]['n_leaves']}, "
               f"leaves with lift >= {MIN_LIFT}: {tree_info[which]['n_fire_leaves']}\n- Items used by the tree: {', '.join(tree_info[which]['items_used'])}\n"
               f"- Items used by the rule card: {', '.join(tree_info[which]['card_items'])}\n"]
    md += ["\n## Rule card versus baselines on 2020-2026\n", md_table(comp, 3)]
    (RESULTS / "summary.md").write_text("".join(md), encoding="utf-8")
    (RESULTS / "tree_summary.json").write_text(json.dumps(tree_info, indent=2))
    print("".join(md))


if __name__ == "__main__":
    main()
