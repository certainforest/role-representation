"""
logit lens :)

needs: 
- `model(**inputs, output_hidden_states=True, return_dict=True)` returns `.hidden_states`
- `model.lm_head` maps hidden -> vocab logits

sample use: 
    text = '''The first letter of the alphabet is: '''
    out, tokens = get_model_outputs(model, tokenizer, text, device="cuda")
    df = logit_lens_to_df(model, out, tokens, tokenizer, k=5)
    plot_logit_lens(df, target="A")
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def get_model_outputs(model, tokenizer, text, device = 'cuda'):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    tokens = [tokenizer.decode(t) for t in inputs['input_ids'][0]]
    return out, tokens


def logit_lens_to_df(model, out, tokens, tokenizer, k=5):
    rows = []
    with torch.no_grad():
        for layer_idx, layer_hs in enumerate(out.hidden_states):
            normed = model.model.norm(layer_hs)
            logits = model.lm_head(normed)
            probs = logits.softmax(-1)[0, -1]
            top = probs.topk(k)
            for rank in range(k):
                rows.append({
                    'layer':         layer_idx,
                    'rank':          rank,
                    'decoded_token': tokenizer.decode(top.indices[rank]),
                    'prob':          top.values[rank].item(),
                })
    return pd.DataFrame(rows)


def plot_logit_lens(logit_df, target):
    pivot_text = logit_df.pivot(index='layer', columns='rank', values='decoded_token')
    pivot_prob = logit_df.pivot(index='layer', columns='rank', values='prob')

    fig, ax = plt.subplots(figsize=(10, len(pivot_prob) * 0.35))
    sns.heatmap(pivot_prob, ax=ax, cmap='RdYlGn', cbar_kws={'label': 'Probability'},
                linewidths=0.3, linecolor='#eeeeee', vmin=0, vmax=1)
    for (i, j), val in np.ndenumerate(pivot_text.values):
        prob = pivot_prob.values[i, j]
        text_color = 'white' if (prob > 0.7 or prob < 0.2) else '#222222'
        label = f"{str(val).strip()[:6]} ({prob:.2f})"
        ax.text(j + 0.5, i + 0.5, label,
                ha='center', va='center', fontsize=8,
                fontfamily='DejaVu Sans', color=text_color,
                fontweight='bold')
    ax.set_title(f"Answer is: {target}", fontfamily='DejaVu Sans')
    ax.set_xlabel('Rank', fontfamily='DejaVu Sans')
    ax.set_ylabel('Layer', fontfamily='DejaVu Sans')
    plt.tight_layout()
    plt.show()
