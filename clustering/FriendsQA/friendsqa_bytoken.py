import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict


def parse_friendsqa_by_scene(filepath: str, by_sentence: bool = False) -> dict[str, list[tuple[str, str]]]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    def split_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    scenes = {}
    for dialogue in data['data']:
        scene_id = dialogue['title']
        scenes[scene_id] = []
        
        for para in dialogue['paragraphs']:
            for utt in para.get('utterances:', []) or para.get('utterances', []):
                speaker = utt['speakers'][0]
                text = utt['utterance'].strip()
                
                if not text or speaker == '#NOTE#':
                    continue
                
                if by_sentence:
                    for sentence in split_sentences(text):
                        scenes[scene_id].append((speaker, sentence))
                else:
                    scenes[scene_id].append((speaker, text))
    
    return scenes


def filter_by_sentence_count(transcript: list[tuple[str, str]], min_sentences: int = 10) -> list[tuple[str, str]]:
    """Filter out speakers who say fewer than min_sentences total."""
    sentence_counts = defaultdict(int)
    for speaker, text in transcript:
        sentence_counts[speaker] += text.count('.')
    
    valid_speakers = {spk for spk, count in sentence_counts.items() if count >= min_sentences}
    filtered = [(spk, txt) for spk, txt in transcript if spk in valid_speakers]
    
    removed = set(sentence_counts.keys()) - valid_speakers
    if removed:
        print(f"  Filtered out {len(removed)} speakers with <{min_sentences} sentences: {removed}")
    
    return filtered


def get_embeddings_token_level(transcript, tokenizer, model):
    """
    Get TOKEN-LEVEL embeddings, not sentence-level.
    Returns embeddings for each token, with corresponding speaker labels.
    """
    all_embeddings = []
    all_speakers = []
    all_tokens = []
    
    for speaker, text in transcript:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state  # (1, seq_len, hidden_dim)
        
        # Get individual token embeddings (exclude special tokens if desired)
        token_embeddings = hidden.squeeze(0).cpu()  # (seq_len, hidden_dim)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
        
        # Skip special tokens (BOS, EOS, PAD)
        for i, (emb, tok) in enumerate(zip(token_embeddings, tokens)):
            if tok in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
                continue
            if tok is None:
                continue
                
            all_embeddings.append(emb)
            all_speakers.append(speaker)
            all_tokens.append(tok)
    
    return torch.stack(all_embeddings).numpy(), all_speakers, all_tokens


def get_embeddings_sentence_level(transcript, tokenizer, model):
    """
    Original sentence-level embeddings (for comparison).
    """
    embeddings = []
    speakers = []
    
    for speaker, text in transcript:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        emb = hidden.mean(dim=1).squeeze().cpu()  # Average over tokens
        embeddings.append(emb)
        speakers.append(speaker)
    
    return torch.stack(embeddings).numpy(), speakers


def plot_pca_2d_token_level(embeddings, speakers, tokens, title="", save_path=None, show_tokens=False):
    """
    PCA plot for token-level embeddings.
    """
    if len(embeddings) < 3:
        print(f"Skipping {title}: too few tokens ({len(embeddings)})")
        return None
    
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    
    unique_speakers = list(set(speakers))
    colors = {spk: f"C{i % 10}" for i, spk in enumerate(unique_speakers)}
    
    plt.figure(figsize=(14, 10))
    
    for spk in unique_speakers:
        mask = [s == spk for s in speakers]
        pts = emb_2d[mask]
        plt.scatter(pts[:, 0], pts[:, 1], c=colors[spk], label=spk, s=20, alpha=0.5)
    
    # Optionally annotate some tokens
    if show_tokens:
        for i in range(0, len(tokens), max(1, len(tokens) // 50)):  # Show ~50 labels
            plt.annotate(tokens[i], emb_2d[i], fontsize=6, alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(f"{title}\n({len(embeddings)} tokens)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    
    return pca


def plot_scenes_token_level(scenes, scene_ids, tokenizer, model, aggregate=1, min_sentences=10, save_dir="."):
    """
    Plot scenes with TOKEN-LEVEL PCA.
    """
    for i in range(0, len(scene_ids), aggregate):
        batch_ids = scene_ids[i:i + aggregate]
        
        # Combine utterances for this batch
        transcript = []
        for sid in batch_ids:
            transcript.extend(scenes[sid])
        
        # Filter by sentence count
        transcript = filter_by_sentence_count(transcript, min_sentences=min_sentences)
        
        if len(transcript) < 3:
            print(f"Skipping batch {i}: too few utterances after filtering")
            continue
        
        # Title and filename
        if aggregate == 1:
            title = batch_ids[0]
            save_path = f"{save_dir}/friends_{batch_ids[0]}_tokens.png"
        else:
            title = f"Friends scenes {i+1}-{i+len(batch_ids)} ({len(batch_ids)} scenes)"
            save_path = f"{save_dir}/friends_batch_{i+1}-{i+len(batch_ids)}_tokens.png"
        
        unique_spk = set(s for s, _ in transcript)
        print(f"\n{title}: {len(transcript)} utterances, {len(unique_spk)} speakers")
        
        # TOKEN-LEVEL embeddings
        embeddings, speakers, tokens = get_embeddings_token_level(transcript, tokenizer, model)
        print(f"  Total tokens: {len(embeddings)}")
        
        plot_pca_2d_token_level(
            embeddings, speakers, tokens, 
            title=title, save_path=save_path, show_tokens=False
        )
        
        # Silhouette score
        unique_speakers = list(set(speakers))
        if len(unique_speakers) > 1:
            speaker_ids = [unique_speakers.index(s) for s in speakers]
            score = silhouette_score(embeddings, speaker_ids)
            print(f"  Silhouette (token-level): {score:.3f}")


def compare_token_vs_sentence_pca(transcript, tokenizer, model, title="", save_dir="."):
    """
    Side-by-side comparison of token-level vs sentence-level PCA.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- Sentence-level ---
    emb_sent, speakers_sent = get_embeddings_sentence_level(transcript, tokenizer, model)
    pca_sent = PCA(n_components=2)
    emb_sent_2d = pca_sent.fit_transform(emb_sent)
    
    unique_speakers = list(set(speakers_sent))
    colors = {spk: f"C{i % 10}" for i, spk in enumerate(unique_speakers)}
    
    ax = axes[0]
    for spk in unique_speakers:
        mask = [s == spk for s in speakers_sent]
        pts = emb_sent_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors[spk], label=spk, s=50, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca_sent.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_sent.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"SENTENCE-level PCA\n({len(emb_sent)} sentences)")
    ax.legend(fontsize=8)
    
    # Silhouette
    if len(unique_speakers) > 1:
        speaker_ids = [unique_speakers.index(s) for s in speakers_sent]
        score_sent = silhouette_score(emb_sent, speaker_ids)
        ax.text(0.02, 0.98, f"Silhouette: {score_sent:.3f}", transform=ax.transAxes, 
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # --- Token-level ---
    emb_tok, speakers_tok, tokens = get_embeddings_token_level(transcript, tokenizer, model)
    pca_tok = PCA(n_components=2)
    emb_tok_2d = pca_tok.fit_transform(emb_tok)
    
    ax = axes[1]
    for spk in unique_speakers:
        mask = [s == spk for s in speakers_tok]
        pts = emb_tok_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors[spk], label=spk, s=10, alpha=0.4)
    ax.set_xlabel(f"PC1 ({pca_tok.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_tok.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"TOKEN-level PCA\n({len(emb_tok)} tokens)")
    ax.legend(fontsize=8)
    
    # Silhouette
    if len(unique_speakers) > 1:
        speaker_ids = [unique_speakers.index(s) for s in speakers_tok]
        score_tok = silhouette_score(emb_tok, speaker_ids)
        ax.text(0.02, 0.98, f"Silhouette: {score_tok:.3f}", transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_token_vs_sentence.png", dpi=150)
    plt.close()
    
    print(f"\nComparison saved. Sentence silhouette: {score_sent:.3f}, Token silhouette: {score_tok:.3f}")


if __name__ == "__main__":
    # Load model once
    model_name = "allenai/OLMo-7B-0724-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # Parse scenes
    scenes = parse_friendsqa_by_scene("dat/friendsqa_trn.json", by_sentence=False)
    scene_ids = list(scenes.keys())[:50]
    
    print(f"Processing {len(scene_ids)} scenes...")
    
    # Combine all 50 scenes for comparison
    transcript = []
    for sid in scene_ids:
        transcript.extend(scenes[sid])
    transcript = filter_by_sentence_count(transcript, min_sentences=10)
    
    # Compare token vs sentence level
    compare_token_vs_sentence_pca(
        transcript, tokenizer, model,
        title="Friends QA: Token vs Sentence PCA",
        save_dir="plots"
    )
    
    # Token-level plots
    plot_scenes_token_level(
        scenes, 
        scene_ids, 
        tokenizer, 
        model, 
        aggregate=50,
        min_sentences=10,
        save_dir="plots"
    )