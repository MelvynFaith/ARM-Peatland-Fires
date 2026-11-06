# Research: making association rule mining useful for peatland fires

This document records the week 10 research for the Data Mining course project. It answers one question: how can association rule mining (ARM) give a useful output on Indonesian peatland fire data, with real novelty? It ends with the research design for weeks 11 and 12.

Status: proposal. No rules have been mined yet. The numbers in section 2 come from the earlier competition project (SIGAP Gambut, private repo `MelvynFaith/Larper-IRSDC2026`) and from a quick test of 14 hand-picked rules. They are not results of this project.

## 1. What already exists

### 1.1 ARM on Indonesian hotspot data

| Study | Data and area | Method | What it lacks for operational use |
|---|---|---|---|
| Sitanggang et al., Rokan Hilir, Riau ([spatial multidimensional rules](https://www.researchgate.net/publication/276041487_Spatial_Multidimensional_Association_Rules_Mining_in_Forest_Fire_Data)) | Hotspots plus land cover, peat, roads and settlements in one regency | Apriori, 324 rules | Rules describe where hotspots co-occur with static land attributes. No lead time, no weather dynamics, no validation on later years. |
| [FP-Growth and ECLAT on forest fire data](https://ieeexplore.ieee.org/document/7506520) (Riau) | Hotspot attributes | FP-Growth and ECLAT, min support 30%, min confidence 80% | The 30% minimum support keeps only very common patterns, so rare but important conditions (new ignitions) cannot appear. |
| [Sequence patterns on peatland hotspots](https://www.researchgate.net/publication/308753646_Mining_Sequence_Pattern_on_Hotspot_Data_to_Identify_Fire_Spot_in_Peatland) (Sumatra, 2014 to 2015) | Daily hotspot sequences plus weather | SPADE, then association rules | Finds that hotspots on 2 to 5 consecutive days usually mean a real fire. Covers two seasons only, and the goal is confirming fires, not anticipating them. |
| [GA-optimised Apriori on online news](https://journal-isi.org/index.php/isi/article/view/1656) | 742 news transactions about forest and land fires | Apriori with a genetic algorithm | Mines reported causes from text, not measured conditions. |
| Student theses, for example FP-Growth on Landsat burn index in Katingan ([Mercu Buana, 2021](https://repository.mercubuana.ac.id/82812)) | One district | FP-Growth used as a classifier | Single area, no comparison with a baseline. |

The common pattern: one region, a few seasons, rules about co-occurrence at the same time, and no test on years the rules were not mined from.

### 1.2 Related work outside ARM

- Fire drivers in Kalimantan peat are usually studied with machine learning or correlation, for example [XGBoost on the ex-Mega Rice Project](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9286596/). That work notes that the conditions that start fires are still poorly understood, and that human ignition is hard to model.
- Rain and fire: severe burning becomes likely below about 4 mm of rain per day over the season ([NASA summary of Field et al.](https://www.nasa.gov/centers-and-facilities/goddard/severe-2015-indonesian-fire-season-linked-to-el-ni%c3%b1o-drought/)). A [field experiment in Sumatra](https://publish.csiro.au/WF/fulltext/WF21135) found that peat fires cannot be put out below a critical water-table height. These are seasonal or site-level findings, not weekly rules for a given place.
- Time in ARM: a [survey of temporal association rule mining](https://simidat.ujaen.es/en/bibcite/reference/399) reports that most ARM studies ignore time, so nobody knows whether their rules still hold. [Dynamic association rules](https://link.springer.com/chapter/10.1007/11596448_102) track the support and confidence of each rule per period.
- Comparing groups: contrast set mining and emerging patterns find itemsets whose support changes between groups ([Kralj Novak, Lavrač and Webb, JMLR 2009](https://jmlr.org/papers/v10/kralj-novak09a.html)). We found no use of these on El Niño versus normal fire seasons.
- Explaining model errors with rules: slice discovery finds subgroups where a model fails ([Slice Finder and related work](https://arxiv.org/pdf/2211.04476)). Frequent itemset mining is rarely used for this on environmental models.

## 2. What the earlier project tells us

The competition project built a clean weekly dataset: 413 peat cells (0.25°, about 28 km) in Sumatra and Kalimantan, 286,622 cell-weeks from April 2012 to October 2026. It uses NASA FIRMS VIIRS hotspots, CHIRPS rainfall, the Peat-ML peat map and the NOAA ONI El Niño index. Three findings matter for this project.

1. **Fire history dominates.** About 79% of the LightGBM model's importance comes from recent fire activity. A rule such as "burned last week, so burns next week" is already strong, and a hotspot map already shows it. ARM rules that only restate fire history add nothing.
2. **New ignitions are the blind spot.** About 26% of next-week fire cells had no hotspot in the previous 30 days. The model's ignition watch list caught about 20% of those fire cells, with a precision of 23%. Ranking by seasonal history (how often this cell burned in this month in past years) did as well, 23%. Machine learning adds nothing here, so this is where readable rules could add value.
3. **Quick ARM check (14 hand-picked rules).** Rules were measured on 2012 to 2019 and checked on 2020 to 2026:

   | Rule (then fire next week) | Confidence 2012-19 | Lift 2012-19 | Confidence 2020-26 | Lift 2020-26 |
   |---|---|---|---|---|
   | Burned in last 7 days | 61% | 2.3 | 47% | 3.5 |
   | Burned and long dry spell | 74% | 2.8 | 62% | 4.7 |
   | Burned, El Niño and fire season | 80% | 3.0 | 79% | 5.9 |
   | Quiet 30 days, neighbour burning and dry | 18% | 0.7 | 13% | 1.0 |

   The last rule's lift was measured against the base rate of all cells (about 21%), not against quiet cells (about 7 to 8%). Against quiet cells it is roughly 2. So ignition rules are not hopeless, but they need proper mining.

## 3. The gap and the proposed contribution

Existing ARM work on Indonesian fires answers "what conditions occur together with hotspots?". An agency needs answers to decisions made a week ahead:

- **Where to patrol to prevent a new fire** in an area that is quiet now.
- **When a burning area will go out**, so a crew can be moved elsewhere.

Neither the hotspot map, BMKG's weather-only fire danger indices, nor the competition model answers these well with readable reasons.

### Proposed title

**Association Rule Mining of Conditions Preceding New Ignitions and Fire Extinction in Indonesian Peatlands Using Apriori and FP-Growth**

Indonesian version for the IMRAD article: *Penambangan Aturan Asosiasi Kondisi Pendahulu Api Baru dan Padamnya Kebakaran di Lahan Gambut Indonesia Menggunakan Apriori dan FP-Growth*.

### Novelty, stated honestly

The algorithms (Apriori, FP-Growth, ECLAT) are standard. The novelty is in what is mined and how the rules are checked. In the literature we found, no ARM study on Indonesian peat fires does any of the following:

1. **Rules with lead time.** Every rule uses conditions up to today and predicts the next 7 days. Prior work mines conditions and hotspots at the same time.
2. **Rules split by decision.** Two separate rule sets, each mined only on the cells where that decision applies:
   - ignition rules on quiet cells (no hotspot in 30 days), with consequent `API_BARU` (new fire next week);
   - extinction rules on burning cells (hotspot in the last 7 days), with consequent `PADAM` (no hotspot next week).
   This removes fire history as the dominant item, which otherwise floods the rule list.
3. **Rules tested on later years.** Rules are mined on 2012 to 2019 and judged on 2023 to 2026, years they never saw. Each rule also gets a lift value per year, so its stability can be shown, not assumed.
4. **Rules that must beat a free baseline.** A rule is kept only if it is better than what an officer already knows. For ignition, the baseline is seasonal history for that cell and month. For extinction, it is a single rain rule. This uses the improvement measure: a rule must beat every shorter rule inside it.
5. **El Niño versus normal years, as a contrast.** Emerging-pattern growth rate and lift difference show which rules only hold in El Niño seasons. This tells agencies which guidance to switch on when NOAA declares El Niño.
6. **Honest uncertainty.** Neighbouring cells and weeks are not independent, so p-values would mislead with 286,000 rows. Confidence intervals come from a block bootstrap over weeks, as in the earlier project.

### The useful output

A **rule card** (*Kartu Aturan Siaga*): at most about 10 ignition rules and 10 extinction rules, each written as a plain sentence an officer can check every week from free data (SiPongi or FIRMS hotspots and rainfall). Each rule shows:

- its confidence and lift, with a 95% interval, on the test years;
- how much better it is than the baseline;
- how many cells a week it usually flags, by month;
- whether it holds in El Niño years, normal years, or both.

Example of the intended form (not a real result yet):

> If a peat cell has had no hotspot for 30 days, a neighbouring cell burned in the last 7 days, it has had 14 or more dry days, and it is August, then a new fire there next week is about 3 times more likely than in an average quiet cell.

As a demonstration, the card is applied to the latest week of data to produce a list of flagged cells. This shows the rules work as a tool, not only as a table.

## 4. Business Objective Determination (updated)

**Background.** Peatland fires in Sumatra and Kalimantan return every dry season. The 2015 crisis cost Indonesia about US$16.1 billion (World Bank, 2016). Crews are limited. Existing tools show fires that are already burning (SiPongi, NASA FIRMS) or rate fire danger from weather alone (BMKG fire weather indices).

**Business objectives.**

1. Help agencies (BPBD, Manggala Agni) decide where to send prevention patrols, by giving readable conditions that come before a new fire in a currently quiet peat area.
2. Help them decide when a burning area is likely to go out, so crews can be moved.

**Business success criteria.**

- Each rule can be read as one sentence and checked from free weekly data, without a computer model.
- Rules flag a manageable number of cells per week.
- Rules beat the baseline an officer already uses.

**Data mining goals.**

1. Mine class association rules with consequent `API_BARU` on quiet cell-weeks, and with consequent `PADAM` on burning cell-weeks.
2. Compare Apriori, FP-Growth and ECLAT: the rules they find (expected to be identical) and their run time and memory at several minimum-support levels.
3. Measure each rule's lift and stability on years it was not mined from, and in El Niño versus normal years.

**Data mining success criteria (proposed, to confirm with the lecturer).**

- Holdout lift of at least 1.5 relative to the subgroup base rate, and support of at least 0.5% of the subgroup.
- Holdout confidence of the rule card above the baseline at the same number of flagged cells.
- Lift stays above 1 in at least most test years.

## 5. Research design for weeks 11 and 12

### 5.1 Data

Reuse the competition data pipeline (download and `build_dataset.py`) without changing its cleaning rules: vegetation fires only, Indonesian cells only, satellite-outage weeks dropped. The data folder is about 450 MB and is not stored in git. Downloading needs a free FIRMS map key and about an hour.

The dataset has no temperature, humidity or wind. Optional additions, only if time allows:

- ERA5 relative humidity and temperature (Copernicus, free account).
- A static landscape layer such as distance to roads or land cover. This connects to Sitanggang's work, since most ignitions are human.

### 5.2 Transactions and items

One transaction is one peat cell in one week. Numeric features become items. Bin edges are set on training years only, so the test years cannot leak in. They use quantiles where no physical threshold exists, and literature thresholds where one does (for example about 4 mm a day from Field et al., or [100 mm a month](https://mdpi-res.com/d_attachment/fire/fire-09-00024/article_deploy/fire-09-00024.pdf) of rain).

| Group | Example items |
|---|---|
| Rain | `HUJAN30_SANGAT_RENDAH`, `HUJAN7_NAIK_TAJAM`, `KERING_14HARI+` |
| Fire history | `TETANGGA_TERBAKAR_7HARI`, `PERNAH_TERBAKAR_BULAN_INI` (seasonal history), `HOTSPOT_7HARI_TINGGI` |
| Peat | `GAMBUT_TINGGI` (Peat-ML probability) |
| Season and climate | `BULAN_AGU`, `MUSIM_KEMARAU`, `EL_NINO` |
| Place | `PULAU_KALIMANTAN`, `PROV_KALTENG` |
| Consequent | `API_BARU` (quiet subset) or `PADAM` (burning subset) |

### 5.3 Splits

- Mining: 2012 to 2019 (includes the 2015 El Niño).
- Validation, for choosing bins and thresholds: 2020 to 2022.
- Test, used once at the end: 2023 to 2026 (includes the 2023 and 2026 El Niño seasons).

### 5.4 Mining

- Apriori and FP-Growth (`mlxtend`), plus ECLAT (a small implementation or `pyECLAT`).
- Only rules whose consequent is `API_BARU` or `PADAM`.
- Pruning:
  - minimum lift;
  - the improvement rule (each rule must beat all its shorter sub-rules);
  - removing rules that cover the same cells as a stronger rule.

### 5.5 Evaluation

| Level | Measure |
|---|---|
| Algorithms | Same rule set from all three; run time and peak memory at minimum support 5%, 2%, 1%, 0.5%. |
| Each rule | Support, confidence, lift relative to the subgroup, and improvement on mining and test years. Block-bootstrap 95% intervals and lift per year. |
| Rule card | Precision, recall and flagged cells per week on test years, compared with random, seasonal history and the single best rule. For ignition, also compared with the competition model's watch list. |
| El Niño contrast | Growth rate and lift difference of each rule between El Niño and normal fire seasons. |

### 5.6 Possible extra experiments (week 14)

- A rule-based classifier (CBA-style) built from the card, compared with a decision tree on the same items. Its paths are also if-then rules.
- ARM on the competition model's errors (false alarms and missed fires), to describe when the bulletin should not be trusted.
- Adding humidity and temperature from ERA5 and measuring whether the ignition rules improve.

## 6. Risks

- **Ignition rules may stay weak.** Most new fires are lit by people, and the data has no human variables. A weak result is still a valid finding if it is measured and reported honestly. The extinction rules are likely to be stronger, since rain directly drives them.
- **Rules can restate known facts.** Rules such as "dry and El Niño means fire" are already known. The value is in putting numbers on the combinations and testing them on later years.
- **Rules are associations, not causes.** The report must say "comes before" or "is associated with", never "causes".
- **Overlap with the competition project.** The data pipeline is reused, so the report should say so and follow the course rules on reuse and AI assistance.

## 7. Timeline mapping

| Week | Deliverable |
|---|---|
| 10 (done) | Data, data characteristics, BOD, algorithms (this document) |
| 11 to 12 | Code: data pipeline, item building, mining with three algorithms, evaluation, rule card |
| 13 | Report 1 |
| 14 | Extra experiments (section 5.6), if requested |
| 15 | IMRAD article in Indonesian |
| 16 | Presentation |
