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
    """Filter out speakers who say fewer than min_sentences total (counted by periods/dots)."""
    # Count total sentences per speaker
    sentence_counts = defaultdict(int)
    for speaker, text in transcript:
        sentence_counts[speaker] += text.count('.')
    
    # Filter speakers
    valid_speakers = {spk for spk, count in sentence_counts.items() if count >= min_sentences}
    filtered = [(spk, txt) for spk, txt in transcript if spk in valid_speakers]
    
    # Report filtering
    removed = set(sentence_counts.keys()) - valid_speakers
    if removed:
        print(f"  Filtered out {len(removed)} speakers with <{min_sentences} sentences: {removed}")
    
    return filtered


def get_embeddings(transcript, tokenizer, model):
    embeddings = []
    speakers = []
    
    for speaker, text in transcript:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        emb = hidden.mean(dim=1).squeeze().cpu()
        embeddings.append(emb)
        speakers.append(speaker)
    
    return torch.stack(embeddings).numpy(), speakers


def plot_pca_2d(embeddings, speakers, title="", save_path=None):
    if len(embeddings) < 3:
        print(f"Skipping {title}: too few utterances ({len(embeddings)})")
        return None
    
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    
    unique_speakers = list(set(speakers))
    colors = {spk: f"C{i % 10}" for i, spk in enumerate(unique_speakers)}
    
    plt.figure(figsize=(12, 10))
    for spk in unique_speakers:
        mask = [s == spk for s in speakers]
        pts = emb_2d[mask]
        plt.scatter(pts[:, 0], pts[:, 1], c=colors[spk], label=spk, s=50, alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    
    return pca


def plot_scenes(scenes, scene_ids, tokenizer, model, aggregate=1, min_sentences=10, save_dir="."):
    """
    Plot scenes with flexible aggregation and sentence count filtering.
    
    aggregate=1:     each scene separate
    aggregate=N:     every N scenes combined
    min_sentences:   filter speakers with fewer than this many total sentences (dots)
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
            save_path = f"{save_dir}/friends_{batch_ids[0]}.png"
        else:
            title = f"Friends scenes {i+1}-{i+len(batch_ids)} ({len(batch_ids)} scenes)"
            save_path = f"{save_dir}/friends_batch_{i+1}-{i+len(batch_ids)}.png"
        
        unique_spk = set(s for s, _ in transcript)
        print(f"\n{title}: {len(transcript)} utterances, {len(unique_spk)} speakers")
        
        embeddings, speakers = get_embeddings(transcript, tokenizer, model)
        plot_pca_2d(embeddings, speakers, title=title, save_path=save_path)
        
        # Silhouette score
        unique_speakers = list(set(speakers))
        if len(unique_speakers) > 1:
            speaker_ids = [unique_speakers.index(s) for s in speakers]
            score = silhouette_score(embeddings, speaker_ids)
            print(f"  Silhouette: {score:.3f}")


if __name__ == "__main__":
    # Load model once
    model_name = "allenai/OLMo-7B-0724-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # Parse scenes
    scenes = parse_friendsqa_by_scene("dat/friendsqa_trn.json", by_sentence=False)
    scene_ids = list(scenes.keys())[:50]  # 50 scenes
    
    print(f"Processing {len(scene_ids)} scenes...")
    
    # Aggregated plot of all 50 scenes, filtering speakers with <10 sentences
    plot_scenes(
        scenes, 
        scene_ids, 
        tokenizer, 
        model, 
        aggregate=50,        # All 50 scenes in one plot
        min_sentences=10,    # Filter speakers with <10 sentences (dots)
        save_dir="plots"
    )