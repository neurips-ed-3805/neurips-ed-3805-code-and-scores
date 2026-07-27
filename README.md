# neurips-ed-3805-code-and-scores

This repo has the LLM-judge scores from our paper along with the evaluation pipeline code.

## 1. Install

You need Python 3.9 or newer. Install the libraries the code uses:

```bash
pip install -r requirements.txt
```

## 2. Figures

Since some results are already saved in the `reports/` folder, they can be used to generate the figures in the work.

```bash
# Figures — saved as PDFs in the figures/ folder
python figures/plot_frontier_improvements.py     # Figure 1
python figures/plot_canonical_vs_best.py         # Figure 2
python figures/plot_flu_combined.py              # Figure 3
python figures/plot_task_subjectivity.py         # Figure A.1
python figures/plot_correlation_vs_width.py      # Figure A.2

 
# To generate Tables 4 and 5, run
python range_selection/compute_scale_selection_tables.py
```


## 3. Results from raw scores

First unzip the scores (about 1.3 GB zipped; a few GB once unzipped):

```bash
cd scores && bash extract_all.sh && cd ..
```

Then:

```bash
python meta_eval/compute_standard_correlations.py   # -> reports/standard_range_correlations.csv    Pearson/Spearman/Kendall, 11 judges x 5 standard ranges  (Table A.2)

python meta_eval/compute_correlations.py            # -> reports/all_range_correlations.csv          Spearman/Kendall, 8 judges x 20 ranges  (the per-judge 20-range appendix tables)

python meta_eval/compute_canonical_vs_best.py       # -> reports/canonical_vs_best.csv               canonical vs best standard range per judge/dataset  

python figures/compute_cross_range_correlations.py  # -> reports/cross_range_correlation_summary.csv  (data for Figure 3)

python flips/compute_flips_cohens_d.py              # -> reports/cohens_d_instability.csv            bilateral flips + Cohen's-d magnitude swings (Table 1)

python range_selection/scale_selection.py           # -> reports/scale_selection/scale_selection_results.json (Table 4)

python range_selection/scale_selection_ablation.py  # -> reports/scale_selection/scale_selection_ablation_results.json   calibration-size ablation  (Table 5)

python range_selection/compute_scale_selection_tables.py   # -> reports/scale_selection.csv (Table 4) + reports/scale_selection_ablation.csv (Table 5)

bash   non_oracle_gains/run_all_judges.sh            # -> reports/non_oracle_gains/per_judge/<judge>.{json,csv}   calibration-only range-selection sweep, 11 judges 

python non_oracle_gains/build_summary_tables.py      # -> reports/non_oracle_gains/calibration_gains_{by_metric,by_dataset,overall}.csv

python temperature_effects/compute_correlations.py   # -> reports/temperature_effects/<variant>/<dataset>/correlations.csv   temperature-0 + default-temperature judge variants  
```

The figure commands from step 2 can be run on these results.

## 4. BigGen-Bench 

```bash
cd BigGen-Bench/scores && bash extract_all.sh && cd ../..
python BigGen-Bench/compute_correlations.py
python BigGen-Bench/compute_capability_gains.py
```

See `BigGen-Bench/README.md` for more.

## 5. Contents

| Directory | Description |
|:----------|:------------|
| `scores/` | Raw judge scores (zipped), along with the unzip script. |
| `reports/` | Saved CSV files used to generate the tables and figures. |
| `figures/` | Figure-generation scripts and their corresponding outputs. |
| `meta_eval/` | Meta-evaluation code for building the correlation tables from the judge scores. |
| `flips/` | Code for bilateral flips and effect-size swing analyses. |
| `range_selection/` | Implementation of our range selection protocol. |
| `non_oracle_gains/` | Analysis of gains when the scoring range is selected using a small labeled (calibration) set. |
| `temperature_effects/` | Results and analysis of the same judges evaluated at different temperatures. |
| `BigGen-Bench/` | Analysis of judge performance on the BigGen-Bench benchmark, covering a variety of tasks beyond NLG. |

## CSV file contents

| File | Description |
|---|---|
| `all_range_correlations.csv` | Spearman and Kendall correlation between judge and human scores for the 8 non-frontier judges across 4 datasets and all 20 scoring ranges. Used for the per-judge 20-range appendix tables and Figure A.2. |
| `standard_range_correlations.csv` | Pearson, Spearman, and Kendall correlation for all 11 judges across 4 datasets and the 5 standard ranges, with the canonical range flagged. Used for Table A.2 (and Figures 1, 2, A.1). |
| `canonical_vs_best.csv` | Per judge and dataset: the canonical range's correlation vs the best standard range's correlation (Spearman and Kendall). Used for Table A.1. |
| `cross_range_correlation_summary.csv` | Per-instance-averaged Spearman/Pearson agreement between every pair of the 5 standard ranges, for the 3 frontier judges. Used for Figure 3. |
| `cohens_d_instability.csv` | Per judge and dataset: the max and mean Cohen's-d magnitude swing across the 5 standard ranges, and the number of bilateral flips. Used for Table 1. |
| `scale_selection.csv` | Per judge/dataset/metric at 20% calibration: the majority-vote selected range, its full-data pairwise agreement, the best−worst agreement spread, the regret, and the agreement gain over the canonical range for standard ranges. Used for Table 4. |
| `scale_selection_ablation.csv` | Per judge and dataset: selection regret (in percentage points) at 5/10/20/50% calibration fractions. Used for Table 5. |
| `dist_stats.csv` | Per judge/dataset/metric/range score-distribution statistics (e.g. normalized mean) |
| `non_oracle_gains/calibration_gains_by_metric.csv` | Per judge/dataset/metric and calibration fraction (10/20/30/40/50%): held-out-test Spearman correlation (with human scores) of the calibration-selected range vs the canonical range, their difference, and the test-set oracle regret. |
| `non_oracle_gains/calibration_gains_by_dataset.csv` / `_overall.csv` | The same quantities, averaged over metrics (per dataset) and over datasets (overall). |
