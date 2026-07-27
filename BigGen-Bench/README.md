# BiGGen-Bench

Self-contained BiGGen-Bench meta-evaluation.

## Setup and run
```bash
cd scores && bash extract_all.sh && cd ..
python compute_correlations.py       # -> reports/<judge>/correlations.csv  (all ranges)
python compute_capability_gains.py   # -> reports/capability_gains.csv + capability_gains_matrix_{spearman,kendall}.csv
```

The per-capability gain (`compute_capability_gains.py`) is, for each judge and capability, `max(over the 5 standard ranges) - canonical([1,5])` of the correlation with the human scores. The Spearman matrix is the per-capability summary; a `+0.000` cell means canonical was already the best standard range for that capability.

## Notes
- BiGGen-Bench correlations are calculated similar to the WritingPrompts_A dataset since there are very few responses for each entry.

- Gemini 3.1 Pro has the 5 standard ranges only; the other 7 non-frontier judges were run on all 20 ranges.

- Each score file carries a per-item `capability` (task, essentially) label (grounding, instruction_following, planning, reasoning, refinement, safety, theory_of_mind, and tool_usage).
