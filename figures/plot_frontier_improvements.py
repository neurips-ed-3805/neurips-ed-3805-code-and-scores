import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_RN   = _os.path.join(_ROOT, "reports")   # standard_range_correlations.csv, dist_stats.csv
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

df = pd.read_csv(_os.path.join(_RN,'standard_range_correlations.csv'))

name_map = {
    'gemini31_pro': 'Gemini 3.1 Pro',
    'gpt54':        'GPT 5.4',
    'opus47':       'Claude Opus 4.7',
}
target_models = list(name_map.keys())
df = df[df['model'].isin(target_models)].copy()
df['model_clean'] = df['model'].map(name_map)

delta_records = []
for (model_clean, dataset, metric), group in df.groupby(['model_clean', 'dataset', 'metric']):
    canon_row = group[group['is_canon'] == True]
    if canon_row.empty:
        continue
    canon_sp = canon_row['spearman'].values[0]
    for _, row in group[group['is_canon'] == False].iterrows():
        delta_records.append({
            'Model':          model_clean,
            'Dataset':        dataset,
            'Metric':         metric,
            'Delta_Spearman': row['spearman'] - canon_sp,
        })

delta_df     = pd.DataFrame(delta_records)
opportunity  = delta_df.groupby(['Model', 'Dataset', 'Metric'])['Delta_Spearman'].max().reset_index()

desired_dataset_order = ['summeval', 'newsroom', 'tc', 'wp_a']
dataset_labels        = {'summeval': 'SummEval', 'newsroom': 'Newsroom',
                         'tc': 'TopicalChat', 'wp_a': 'WP-A'}
metric_labels         = {'coh': 'Coh', 'con': 'Con', 'flu': 'Flu', 'rel': 'Rel',
                         'inf': 'Inf', 'eng': 'Eng', 'gro': 'Gro', 'nat': 'Nat',
                         'enj': 'Enj', 'gra': 'Gra'}

opportunity['ds_order'] = opportunity['Dataset'].apply(
    lambda d: desired_dataset_order.index(d) if d in desired_dataset_order else 99)
opportunity = opportunity.sort_values(['ds_order', 'Metric'])

opportunity['col'] = list(zip(opportunity['Dataset'], opportunity['Metric']))
col_order = list(dict.fromkeys(opportunity['col']))  

pivot_df = opportunity.pivot(index='Model', columns='col', values='Delta_Spearman')
pivot_df = pivot_df[col_order]  

model_order = [name_map[m] for m in target_models if name_map[m] in pivot_df.index]
pivot_df = pivot_df.reindex(model_order)

num_cols = len(pivot_df.columns)   
num_rows = len(pivot_df.index)     

cell_w, cell_h = 1.30, 1.05        
cbar_w         = 1.6               
header_h       = 0.60              

fig_w = num_cols * cell_w + cbar_w + 1.5
fig_h = num_rows * cell_h + header_h + 0.9

fig = plt.figure(figsize=(fig_w, fig_h))

top_frac    = header_h / fig_h
hm_bottom   = 0.18
hm_top      = 1.0 - top_frac - 0.04
hm_left     = 0.14
hm_right    = 0.88

ax = fig.add_axes([hm_left, hm_bottom, hm_right - hm_left, hm_top - hm_bottom])

vmax = max(abs(pivot_df.values[np.isfinite(pivot_df.values)].max()),
           abs(pivot_df.values[np.isfinite(pivot_df.values)].min()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

sns.heatmap(
    pivot_df,
    cmap='RdYlGn',
    norm=norm,
    annot=True,
    fmt='+.2f',
    annot_kws={'size': 13, 'weight': 'bold'},
    linewidths=0.4,
    linecolor='#cccccc',
    cbar_kws={
        'label':  r'Max $\Delta\rho$ ($\rho_{\mathrm{std}} - \rho_{\mathrm{canon}}$)',
        'shrink': 0.85,
        'pad':    0.02,
    },
    ax=ax,
)

metric_tick_labels = [metric_labels.get(m, m.capitalize()) for (_, m) in col_order]
ax.set_xticks(np.arange(num_cols) + 0.5)
ax.set_xticklabels(metric_tick_labels, rotation=0, ha='center', fontsize=13)
ax.xaxis.tick_bottom()

ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=14)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title('')

dataset_col_counts = []
for ds in desired_dataset_order:
    count = sum(1 for (d, _) in col_order if d == ds)
    if count:
        dataset_col_counts.append((ds, count))

sep_positions = []
running = 0
for ds, cnt in dataset_col_counts[:-1]:
    running += cnt
    sep_positions.append(running)

for xpos in sep_positions:
    ax.axvline(x=xpos, color='#444444', linewidth=1.5, zorder=5)

fig.canvas.draw()  
ds_colors = {
    'summeval': '#d4e6f1',
    'newsroom': '#d5f5e3',
    'tc':       '#fdebd0',
    'wp_a':     '#e8daef',
}

ax_bbox   = ax.get_position()          
plot_w    = ax_bbox.width              
plot_x0   = ax_bbox.x0
header_bottom = ax_bbox.y1 + 0.005
header_height = top_frac - 0.01

ax_top = fig.add_axes([plot_x0, header_bottom, plot_w, header_height])
ax_top.set_xlim(0, num_cols)
ax_top.set_ylim(0, 1)
ax_top.axis('off')

running = 0
for ds, cnt in dataset_col_counts:
    mid = running + cnt / 2.0
    color = ds_colors.get(ds, '#eeeeee')
    rect = mpatches.FancyBboxPatch(
        (running + 0.05, 0.08), cnt - 0.10, 0.84,
        boxstyle='round,pad=0.02',
        facecolor=color, edgecolor='#888888', linewidth=0.8,
        transform=ax_top.transData, zorder=2,
    )
    ax_top.add_patch(rect)
    ax_top.text(mid, 0.50, dataset_labels.get(ds, ds),
                ha='center', va='center', fontsize=13, fontweight='bold',
                transform=ax_top.transData, zorder=3)
    running += cnt

plt.savefig(_os.path.join(_HERE,'frontier_improvements.pdf'), dpi=300, format='pdf', bbox_inches='tight')