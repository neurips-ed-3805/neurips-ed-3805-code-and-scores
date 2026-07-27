import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_RN   = _os.path.join(_ROOT, "reports")   # standard_range_correlations.csv, dist_stats.csv
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

frontier = {'gemini31_pro', 'gpt54', 'opus47'}
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
    'gpt54': 'GPT-5.4',
    'opus47': 'Claude Opus 4.7',
}

df_dist = pd.read_csv(_os.path.join(_RN,'dist_stats.csv'))

df_dist = df_dist[~df_dist['model'].isin(frontier)].copy()

df_dist['Model'] = df_dist['model'].map(name_map)

def build_deltas(df, target_col):
    records = []
    for (model, dataset, metric), g in df.groupby(['Model', 'dataset', 'metric']):
        canon = g[g['is_canon'] == True]
        if canon.empty:
            continue
        canon_val = canon[target_col].iloc[0]
        non_canon = g[g['is_canon'] == False]

        for _, row in non_canon.iterrows():
            records.append({
                'Model': model,
                'Dataset': dataset,
                'Metric': metric,
                'Delta': row[target_col] - canon_val,
                'AbsDelta': abs(row[target_col] - canon_val),
            })
    return pd.DataFrame(records)

score_delta = build_deltas(df_dist, 'norm_mean')

preferred_dataset_order = ['summeval', 'newsroom', 'tc', 'wp_a']
available_datasets = set(score_delta['Dataset'].unique())
dataset_order = [d for d in preferred_dataset_order if d in available_datasets]
dataset_order += sorted(list(available_datasets - set(dataset_order)))

model_order = [
    name_map['gemma2_9b'],
    name_map['qwen3_4b'],
    name_map['mistral_7b'],
    name_map['llama31_8b'],
]
model_order = [m for m in model_order if m in set(score_delta['Model'])]

palette_vals = sns.color_palette('tab10', n_colors=max(len(model_order), 3))
palette = {m: palette_vals[i % len(palette_vals)] for i, m in enumerate(model_order)}


def draw_single_plot(data, ylabel, title, out_pdf):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))

    sns.boxplot(
        data=data,
        x='Dataset', y='AbsDelta', hue='Model',
        order=dataset_order, hue_order=model_order,
        palette=palette, showfliers=False,
        width=0.72, linewidth=0.9, ax=ax,
        boxprops={'alpha': 0.35},
    )

    sns.stripplot(
        data=data,
        x='Dataset', y='AbsDelta', hue='Model',
        order=dataset_order, hue_order=model_order,
        palette=palette, dodge=True,
        alpha=0.45, size=2.8, linewidth=0, ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    uniq_h = []
    uniq_l = []
    seen = set()
    for h, l in zip(handles, labels):
        if l in model_order and l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)

    if ax.legend_:
        ax.legend_.remove()

    ax.legend(
        uniq_h, uniq_l,
        loc='upper center', bbox_to_anchor=(0.5, 1.20),
        ncol=4, frameon=True, columnspacing=0.9, handletextpad=0.4
    )

    ax.set_title(title)
    ax.set_xlabel('Dataset')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)


draw_single_plot(
    score_delta,
    r'$|\Delta\,\mu_{norm}|$ vs canonical',
    'Normalized-score fragility by dataset',
    _os.path.join(_HERE,'task_subjectivity.pdf'),
)