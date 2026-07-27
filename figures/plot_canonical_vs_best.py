import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_RN   = _os.path.join(_ROOT, "reports")   # standard_range_correlations.csv, dist_stats.csv
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (5.5, 3.5), 
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
    'pdf.fonttype': 42, 
    'ps.fonttype': 42,
    'hatch.linewidth': 0.4 
})


df = pd.read_csv(_os.path.join(_RN,'standard_range_correlations.csv'))

name_map = {
    'gemma2_9b': 'Gemma 2 9B',
    'gemma3_27b': 'Gemma 3 27B',
    'qwen3_4b': 'Qwen 3 4B',
    'mistral_7b': 'Mistral 7B',
    'llama31_8b': 'Llama 3.1 8B',
    'llama31_70b': 'Llama 3.1 70B',
    'gemini2_flash': 'Gemini 2 Flash',
    'gemini3_flash': 'Gemini 3 Flash',
    'gemini31_pro': 'Gemini 3.1 Pro',
    'gpt54': 'GPT 5.4',
    'opus47': 'Claude Opus 4.7'
}

models = df['model'].unique()
display_models = [name_map.get(m, m) for m in models]

canon_means = []
best_fixed_means = []
oracle_means = []

for m in models:
    m_df = df[df['model'] == m]
    
    # 1. Canonical Range
    canon_df = m_df[m_df['is_canon'] == True]
    canon_means.append(canon_df['spearman'].mean())
    
    # 2. Best Std (Best Fixed Alternative Range)
    fixed_means = m_df.groupby(['rmin', 'rmax'])['spearman'].mean()
    best_fixed_means.append(fixed_means.max())
    
    # 3. Oracle: best range per task across ALL ranges (including canonical)
    oracle_best = m_df.groupby(['dataset', 'metric', 'rmin', 'rmax'])['spearman'].mean()
    oracle_best = oracle_best.groupby(['dataset', 'metric']).max()
    oracle_means.append(oracle_best.mean())


x = np.arange(len(models))
width = 0.25 

fig, ax = plt.subplots()
colors = ['#7BA1D2', '#F6A87C', '#85C796']

rects1 = ax.bar(x - width, canon_means, width, label='Canonical Range', 
                color=colors[0], edgecolor='black', linewidth=0.8, hatch='\\\\')
rects2 = ax.bar(x, best_fixed_means, width, label='Best Std', 
                color=colors[1], edgecolor='black', linewidth=0.8, hatch='..')
rects3 = ax.bar(x + width, oracle_means, width, label='Oracle', 
                color=colors[2], edgecolor='black', linewidth=0.8, hatch='-')

ax.set_ylabel("Spearman ($\\bar{\\rho}$)")
ax.set_xticks(x)
ax.set_xticklabels(display_models, rotation=40, ha='right')

y_max_overall = max(max(canon_means), max(best_fixed_means), max(oracle_means))
ax.set_ylim(0.0, y_max_overall + 0.12)

ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=True, columnspacing=1.0, handletextpad=0.4)

ax.grid(axis='y', linestyle='--', alpha=0.6)

for i, rect in enumerate(rects3):
    diff = oracle_means[i] - canon_means[i]
    sign = "+" if diff >= 0 else ""
    
    y_max = max(canon_means[i], best_fixed_means[i], oracle_means[i])
    
    ax.annotate(f"{sign}{diff:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, y_max),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7, rotation=30)

plt.tight_layout()
output_path = _os.path.join(_HERE,'canonical_vs_best.pdf')
plt.savefig(output_path, format='pdf', bbox_inches='tight')