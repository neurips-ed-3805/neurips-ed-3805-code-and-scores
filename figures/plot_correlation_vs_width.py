import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_RN   = _os.path.join(_ROOT, "reports")   # standard_range_correlations.csv, dist_stats.csv
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'pdf.fonttype': 42,  
    'ps.fonttype': 42,
    'lines.linewidth': 1.0,
})

CSV_PATH = _os.path.join(_RN,'all_range_correlations.csv')
df = pd.read_csv(CSV_PATH)

rename_map = {
    'judge': 'model',
    'range_min': 'rmin',
    'range_max': 'rmax',
    'spearman_rho': 'spearman',
}
df = df.rename(columns=rename_map)

frontier_models = {'gemini31_pro', 'gpt54', 'opus47'}
df = df[~df['model'].isin(frontier_models)].copy()

name_map = {
    'gemma2_9b': 'Gemma 2 9B',
    'gemma3_27b': 'Gemma 3 27B',
    'qwen3_4b': 'Qwen 3 4B',
    'mistral_7b': 'Mistral 7B',
    'llama31_8b': 'Llama 3.1 8B',
    'llama31_70b': 'Llama 3.1 70B',
    'gemini2_flash': 'Gemini 2 Flash',
    'gemini3_flash': 'Gemini 3 Flash',
}
df['model_clean'] = df['model'].map(lambda x: name_map.get(x, x))

# Ranges: (0-1, 0-5, 0-10, 0-100, 0-1000)
target_widths = [1, 5, 10, 100, 1000]
df = df[(df['rmin'] == 0) & (df['rmax'].isin(target_widths))].copy()

df['range_label'] = df['rmax'].map(lambda w: f'0-{w}')

model_order = [name_map[m] for m in name_map.keys() if m in df['model'].unique()]
width_order = target_widths

available_widths = sorted(df['rmax'].unique())
missing_widths = [w for w in target_widths if w not in available_widths]
print(f'Available widths in filtered data: {available_widths}')
if missing_widths:
    print(f'Missing requested widths (no data): {missing_widths}')

markers_list = ['o', 's', '^', 'D', 'v', 'p', 'X', '*']
model_markers = {m: markers_list[i % len(markers_list)] for i, m in enumerate(model_order)}


overall_view = df.groupby(['model_clean', 'rmax'], as_index=False)['spearman'].mean()


def setup_x_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(width_order)
    ax.set_xticklabels([f'0-{w}' for w in width_order])
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.grid(axis='x', linestyle=':', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def overall_plot():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    palette = sns.color_palette('tab10', n_colors=max(len(model_order), 3))

    for i, judge in enumerate(model_order):
        sub = overall_view[overall_view['model_clean'] == judge].sort_values('rmax')
        if sub.empty:
            continue
        ax.plot(sub['rmax'], sub['spearman'], marker=model_markers[judge], markersize=4.5,
                linewidth=1.2, alpha=0.8, label=judge, color=palette[i % len(palette)],
                markeredgecolor='white', markeredgewidth=0.4)

    setup_x_axis(ax)
    ax.set_ylabel(r'Spearman ($\bar{\rho}$)')
    ax.set_xlabel('Score Range')
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
    
    fig.tight_layout()
    fig.savefig(_os.path.join(_HERE,'correlation_vs_width.pdf'), dpi=300, format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    overall_plot()