from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def get_embeddings(transcript, model_name="allenai/OLMo-7B-0724-Instruct-hf"):
    """Get embeddings for transcript using specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    
    embeddings = []
    speakers = []
    
    for speaker, text in transcript:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        emb = hidden.mean(dim=1).squeeze()
        embeddings.append(emb)
        speakers.append(speaker)
    
    return torch.stack(embeddings).numpy(), speakers


def plot_pca_2d(embeddings, speakers, title="Utterance embeddings by speaker", save_path=None):
    """2D PCA visualization."""
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    
    unique_speakers = list(set(speakers))
    colors = {spk: f"C{i}" for i, spk in enumerate(unique_speakers)}
    
    plt.figure(figsize=(10, 8))
    for spk in unique_speakers:
        mask = [s == spk for s in speakers]
        pts = emb_2d[mask]
        plt.scatter(pts[:, 0], pts[:, 1], c=colors[spk], label=spk, s=50, alpha=0.7)
    
    plt.legend()
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    
    return pca, emb_2d


def plot_pca_3d(embeddings, speakers, title="Utterance embeddings by speaker", save_path=None, interactive=True):
    """3D PCA visualization. Set interactive=True for plotly, False for matplotlib."""
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(embeddings)
    
    unique_speakers = list(set(speakers))
    
    if interactive:
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame({
            'PC1': emb_3d[:, 0],
            'PC2': emb_3d[:, 1],
            'PC3': emb_3d[:, 2],
            'speaker': speakers
        })
        
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='speaker', title=title)
        fig.update_traces(marker=dict(size=4))
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
        fig.show()
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {spk: f"C{i}" for i, spk in enumerate(unique_speakers)}
        for spk in unique_speakers:
            mask = [s == spk for s in speakers]
            pts = emb_3d[mask]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors[spk], label=spk, s=30, alpha=0.7)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.legend()
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    return pca, emb_3d


def compute_silhouette(embeddings, speakers):
    """Compute silhouette score for speaker separability."""
    unique_speakers = list(set(speakers))
    speaker_ids = [unique_speakers.index(s) for s in speakers]
    
    if len(unique_speakers) > 1 and len(speakers) > 2:
        score = silhouette_score(embeddings, speaker_ids)
        print(f"Silhouette score (speaker separability): {score:.3f}")
        return score
    return None


# === Usage ===
if __name__ == "__main__":
    from parse_txt import parse_transcript, parse_transcript_sentences, parse_debate_transcript_sentences, parse_debate_transcript_blocks
    source = "generic"  # or "debate"
    section = "sentences"  # or "blocks"
    dimension = 3
    file_name = "Fireworks_Lin_Qiao.txt"
    result_file = f"{file_name}_{section}_{dimension}d.png"
    with open(file_name, "r") as f:
        raw = f.read()
    
    if source == "debate":
        if section == "blocks":
            transcript = parse_debate_transcript_blocks(raw)
        else:   
            transcript = parse_debate_transcript_sentences(raw)
    else:
        if section == "blocks":
            transcript = parse_transcript(raw)
        else:
            transcript = parse_transcript_sentences(raw)
    embeddings, speakers = get_embeddings(transcript)
    
    # Choose one:
    if dimension == 2:
        plot_pca_2d(embeddings, speakers, save_path=result_file)
    else:
        plot_pca_3d(embeddings, speakers, interactive=False, save_path=result_file)
    
    compute_silhouette(embeddings, speakers)