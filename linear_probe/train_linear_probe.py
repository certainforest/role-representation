"""
Linear probe with CHUNKED activation collection — 3 speakers.

Same as train_linear_probe_chunk.py but configured for a three-person
transcript (Alice, Bob, Charlie). Uses a single 3-class linear probe,
NOT three separate probes.

Train/test split is chronological by turns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ══════════════════════════════════════════════════════════════════════
# CONFIGURE HERE
# ══════════════════════════════════════════════════════════════════════
SPEAKERS = ["OBAMA", "MCCAIN", "LEHRER"]
TRANSCRIPT_PATHS = [
    "2008debate_1.txt",
    "2008debate_2.txt",
    "2008debate_3.txt",
]
# SPEAKERS = ["Alice", "Bob"]
# TRANSCRIPT_PATHS = [
#     "transcript_long_continue_transcript1_serious_career.txt"
# ]

# Optional: only probe a subset of speakers. Set to None to probe all.
# Tokens from excluded speakers are still seen by the model but ignored by the probe.
# PROBE_SPEAKERS = ["OBAMA", "MCCAIN"]
PROBE_SPEAKERS = None  # probe all speakers

MODEL_NAME = "gpt2"
# MODEL_NAME = "meta-llama/Llama-3.1-8B"

MAX_CHUNK_TOKENS = 8192  # cap chunk size to avoid OOM on large-context models

COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]


# ══════════════════════════════════════════════════════════════════════
# 1. TRANSCRIPT PARSING
# ══════════════════════════════════════════════════════════════════════

def parse_transcripts(paths, speakers):
    """Parse one or more transcripts. Returns list of (speaker_id, content) turns
    and a parallel list of transcript indices (which file each turn came from)."""
    speaker_to_id = {s: i for i, s in enumerate(speakers)}
    turns = []
    turn_transcript_ids = []  # which transcript each turn belongs to

    for ti, path in enumerate(paths):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for speaker in speakers:
                    if line.startswith(f"{speaker}:"):
                        content = line[len(speaker) + 1:].strip()
                        turns.append((speaker_to_id[speaker], content))
                        turn_transcript_ids.append(ti)
                        break

    return turns, speaker_to_id, turn_transcript_ids


def turns_to_text(turns):
    """Convert list of (speaker_id, content) turns to quoted text with
    per-character speaker labels."""
    full_text = ""
    speaker_per_char = []
    turn_char_boundaries = []

    for speaker_id, content in turns:
        text = f'"{content}"\n'
        char_start = len(full_text)
        full_text += text
        char_end = len(full_text)
        speaker_per_char += [speaker_id] * len(text)
        turn_char_boundaries.append((char_start, char_end, speaker_id))

    return full_text, speaker_per_char, turn_char_boundaries


def tokenize_and_label(tokenizer, text, speaker_per_char):
    """Tokenize text and map character-level speaker labels to tokens."""
    encoding = tokenizer(text, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    speaker_per_token = []
    for start, end in offsets:
        if start == end:
            speaker_per_token.append(-1)
        else:
            mid = (start + end) // 2
            if mid < len(speaker_per_char):
                speaker_per_token.append(speaker_per_char[mid])
            else:
                speaker_per_token.append(-1)

    return token_ids, speaker_per_token


# ══════════════════════════════════════════════════════════════════════
# 2. CHUNKED ACTIVATION COLLECTION
# ══════════════════════════════════════════════════════════════════════

def build_synthetic_prefix(speakers):
    """Build a short synthetic prefix: one 'I am X.' line per speaker."""
    id_to_name = {i: s for i, s in enumerate(speakers)}
    return [(spk_id, f"I am {id_to_name[spk_id]}.") for spk_id in range(len(speakers))]


def collect_activations_chunked(model, tokenizer, all_turns, speakers,
                                turn_transcript_ids=None):
    """Collect activations by splitting turns into context-sized chunks.

    Each chunk is prefixed with a short synthetic intro so the model has
    speaker context. Only activations for non-prefix tokens are kept.
    Returns: activations dict {layer: np.array(total_tokens, hidden_dim)},
             speaker_per_token list, token_transcript_ids list.
    """
    max_len = min(
        getattr(tokenizer, "model_max_length", 1024),
        MAX_CHUNK_TOKENS,
    )

    prefix_turns = build_synthetic_prefix(speakers)
    body_turns = all_turns  # all actual turns are body

    # Tokenize prefix to know its length
    prefix_text, prefix_chars, _ = turns_to_text(prefix_turns)
    prefix_ids, _ = tokenize_and_label(tokenizer, prefix_text, prefix_chars)
    prefix_len = len(prefix_ids)
    chunk_budget = max_len - prefix_len

    print(f"  Synthetic prefix: {[f'I am {s}.' for s in speakers]}")
    print(f"  Model max_len: {max_len}")
    print(f"  Prefix tokens: {prefix_len}")
    print(f"  Budget per chunk: {chunk_budget} tokens")

    # Build parallel list of transcript IDs per turn (default to 0 if not provided)
    if turn_transcript_ids is None:
        turn_transcript_ids = [0] * len(body_turns)

    # Group body turns into chunks that fit in budget
    chunks = []       # each chunk: list of (speaker_id, content) turns
    chunk_tids = []   # each chunk: list of transcript IDs for those turns
    current_chunk = []
    current_tids = []
    current_len = 0

    for turn, tid in zip(body_turns, turn_transcript_ids):
        turn_text, turn_chars, _ = turns_to_text([turn])
        turn_ids, _ = tokenize_and_label(tokenizer, turn_text, turn_chars)
        turn_len = len(turn_ids)

        if current_len + turn_len > chunk_budget and current_chunk:
            chunks.append(current_chunk)
            chunk_tids.append(current_tids)
            current_chunk = []
            current_tids = []
            current_len = 0

        current_chunk.append(turn)
        current_tids.append(tid)
        current_len += turn_len

    if current_chunk:
        chunks.append(current_chunk)
        chunk_tids.append(current_tids)

    print(f"  Split into {len(chunks)} chunks")

    # Process each chunk
    all_activations = {}
    all_speaker_labels = []
    all_token_tids = []  # transcript ID per token

    for ci, (chunk_turns, ctids) in enumerate(zip(chunks, chunk_tids)):
        combined_turns = prefix_turns + chunk_turns
        combined_text, combined_chars, _ = turns_to_text(combined_turns)
        combined_ids, combined_speakers = tokenize_and_label(
            tokenizer, combined_text, combined_chars
        )

        inputs = tokenizer(combined_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx, hidden in enumerate(outputs.hidden_states):
            act = hidden[0, prefix_len:, :].float().cpu().numpy()
            if layer_idx not in all_activations:
                all_activations[layer_idx] = []
            all_activations[layer_idx].append(act)

        chunk_labels = combined_speakers[prefix_len:]
        all_speaker_labels.extend(chunk_labels)

        # Map each token in this chunk back to its transcript ID
        # by expanding turn-level tids to token-level
        for turn, tid in zip(chunk_turns, ctids):
            turn_text, turn_chars, _ = turns_to_text([turn])
            turn_ids, _ = tokenize_and_label(tokenizer, turn_text, turn_chars)
            all_token_tids.extend([tid] * len(turn_ids))

        if (ci + 1) % 5 == 0 or ci == 0:
            print(f"    Chunk {ci+1}/{len(chunks)}: "
                  f"{len(combined_ids)} tokens ({prefix_len} prefix + "
                  f"{len(combined_ids) - prefix_len} content)")

    # Also process the prefix turns themselves
    prefix_text_full, prefix_chars_full, _ = turns_to_text(prefix_turns)
    prefix_ids_full, prefix_speakers = tokenize_and_label(
        tokenizer, prefix_text_full, prefix_chars_full
    )

    inputs = tokenizer(prefix_text_full, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for layer_idx, hidden in enumerate(outputs.hidden_states):
        act = hidden[0].float().cpu().numpy()
        all_activations[layer_idx].insert(0, act)

    all_speaker_labels = prefix_speakers + all_speaker_labels
    all_token_tids = [-1] * len(prefix_ids_full) + all_token_tids  # prefix has no transcript

    activations = {}
    for layer_idx in all_activations:
        activations[layer_idx] = np.concatenate(all_activations[layer_idx], axis=0)

    print(f"  Total activation tokens: {activations[0].shape[0]}")
    print(f"  Total speaker labels: {len(all_speaker_labels)}")

    return activations, all_speaker_labels, all_token_tids


# ══════════════════════════════════════════════════════════════════════
# 3. LINEAR PROBE
# ══════════════════════════════════════════════════════════════════════

class RoleProbe(nn.Module):
    def __init__(self, hidden_dim, num_roles):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_roles)

    def forward(self, x):
        return self.classifier(x)


def prepare_data_splits(speaker_per_token, token_transcript_ids,
                        test_split=0.2, seed=42):
    """Per-transcript chronological split: last 20% of each transcript = test.

    This ensures every transcript contributes to both train and test,
    and temporal ordering is preserved within each transcript.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Group valid token indices by transcript
    transcript_groups = {}
    for i, (spk, tid) in enumerate(zip(speaker_per_token, token_transcript_ids)):
        if spk >= 0 and tid >= 0:
            transcript_groups.setdefault(tid, []).append(i)

    train_indices = []
    test_indices = []

    for tid in sorted(transcript_groups.keys()):
        indices = transcript_groups[tid]  # already in order
        n_test = int(len(indices) * test_split)
        n_train = len(indices) - n_test
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
        print(f"    Transcript {tid}: {n_train} train, {n_test} test tokens")

    print(f"  Total: {len(train_indices)} train, {len(test_indices)} test tokens")

    train_labels = torch.tensor([speaker_per_token[i] for i in train_indices],
                                dtype=torch.long)
    test_labels = torch.tensor([speaker_per_token[i] for i in test_indices],
                               dtype=torch.long)

    return train_indices, test_indices, train_labels, test_labels


def compute_confusion_matrix(preds, labels, num_roles):
    """Build a num_roles x num_roles confusion matrix (row=true, col=pred)."""
    cm = np.zeros((num_roles, num_roles), dtype=int)
    for t, p in zip(labels.numpy(), preds.numpy()):
        cm[t, p] += 1
    return cm


def train_probe(acts_layer, train_idx, test_idx, train_labels, test_labels,
                num_roles, epochs=50, lr=1e-3, weight_decay=1e-4,
                batch_size=64, verbose=True):
    """Train linear probe on one layer. Returns (best_acc, best_epoch, confusion_matrix)."""
    hidden_dim = acts_layer.shape[1]
    X_train = torch.tensor(acts_layer[train_idx], dtype=torch.float32)
    X_test = torch.tensor(acts_layer[test_idx], dtype=torch.float32)

    probe = RoleProbe(hidden_dim, num_roles)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    best_epoch = 0
    best_cm = None

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_train))
        correct, total = 0, 0

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = probe(X_train[idx])
            loss = criterion(logits, train_labels[idx])
            loss.backward()
            optimizer.step()

            _, pred = torch.max(logits, 1)
            correct += (pred == train_labels[idx]).sum().item()
            total += idx.size(0)

        train_acc = correct / total

        probe.eval()
        with torch.no_grad():
            logits_test = probe(X_test)
            _, pred_test = torch.max(logits_test, 1)
            test_acc = (pred_test == test_labels).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_cm = compute_confusion_matrix(pred_test, test_labels, num_roles)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}: Train={train_acc:.3f}, Test={test_acc:.3f}")

    return best_test_acc, best_epoch, best_cm


def train_probe_random_labels(acts_layer, train_idx, test_idx, num_roles,
                              seed=99, epochs=50, lr=1e-3, weight_decay=1e-4,
                              batch_size=64):
    """Probe with random labels — sanity check."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train = torch.tensor(acts_layer[train_idx], dtype=torch.float32)
    X_test = torch.tensor(acts_layer[test_idx], dtype=torch.float32)
    rand_train = torch.randint(0, num_roles, (len(train_idx),))
    rand_test = torch.randint(0, num_roles, (len(test_idx),))

    probe = RoleProbe(acts_layer.shape[1], num_roles)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(probe(X_train[idx]), rand_train[idx])
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            _, pred = torch.max(probe(X_test), 1)
            acc = (pred == rand_test).float().mean().item()
        if acc > best_acc:
            best_acc = acc

    return best_acc


# ══════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ══════════════════════════════════════════════════════════════════════

def plot_results(results, random_results, num_turns, num_train,
                 num_roles, model_name, output_path="probe_results.png"):
    """Plot probe accuracy across layers with random label control."""
    layers = [r["layer"] for r in results]
    accs = [r["test_acc"] for r in results]
    rand_accs = [r["random_test_acc"] for r in random_results]
    chance = 1.0 / num_roles

    fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(layers, accs, "o-", color="#2ecc71", linewidth=2.5, markersize=9,
            label="Real labels", zorder=3)
    ax.plot(layers, rand_accs, "s--", color="#e74c3c", linewidth=2, markersize=7,
            label="Random labels (control)", zorder=3)
    ax.axhline(y=chance, color="gray", linestyle=":", linewidth=1.5,
               label=f"Chance ({chance:.2f})")

    for l, a in zip(layers, accs):
        ax.annotate(f"{a:.3f}", (l, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7, color="#2ecc71")
    for l, a in zip(layers, rand_accs):
        ax.annotate(f"{a:.3f}", (l, a), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="#e74c3c")

    model_short = model_name.split("/")[-1]
    ax.set_ylabel("Test Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Linear Probe (chunked, 3-speaker): Real vs Random — {model_short}\n"
        f"Chrono split · {num_turns} turns · {num_train} train tokens · {num_roles} speakers",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(max(0, chance - 0.15), 1.02)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, linestyle="--")

    ax2 = axes[1]
    gaps = [a - r for a, r in zip(accs, rand_accs)]
    colors_bar = ["#2ecc71" if g > 0.05 else "#f39c12" if g > 0 else "#e74c3c"
                  for g in gaps]
    ax2.bar(layers, gaps, color=colors_bar, alpha=0.7, width=0.7)
    ax2.axhline(y=0, color="gray", linewidth=1)
    ax2.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Real − Random", fontsize=11, fontweight="bold")
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

    for l, g in zip(layers, gaps):
        ax2.text(l, g + 0.005, f"{g:+.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ Saved plot: {output_path}")
    plt.close()


def plot_confusion_matrix(cm, speaker_names, layer, output_path="confusion_matrix.png"):
    """Plot a confusion matrix heatmap with per-speaker accuracy."""
    num_roles = len(speaker_names)
    # Normalize rows to get per-speaker accuracy
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)  # avoid div by zero

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    ax = axes[0]
    im = ax.imshow(cm, cmap="Greens", aspect="auto")
    ax.set_xticks(range(num_roles))
    ax.set_yticks(range(num_roles))
    ax.set_xticklabels(speaker_names, fontsize=11)
    ax.set_yticklabels(speaker_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix (counts) — Layer {layer}", fontsize=13, fontweight="bold")
    for i in range(num_roles):
        for j in range(num_roles):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Normalized (per-speaker accuracy)
    ax2 = axes[1]
    im2 = ax2.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(num_roles))
    ax2.set_yticks(range(num_roles))
    ax2.set_xticklabels(speaker_names, fontsize=11)
    ax2.set_yticklabels(speaker_names, fontsize=11)
    ax2.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax2.set_ylabel("True", fontsize=12, fontweight="bold")
    ax2.set_title(f"Confusion Matrix (recall) — Layer {layer}", fontsize=13, fontweight="bold")
    for i in range(num_roles):
        for j in range(num_roles):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if cm_norm[i, j] > 0.6 else "black")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Print per-speaker accuracy
    print(f"\n  Per-speaker accuracy (layer {layer}):")
    for i, name in enumerate(speaker_names):
        total = row_sums[i, 0]
        correct = cm[i, i]
        acc = cm_norm[i, i]
        print(f"    {name}: {correct}/{total} = {acc:.3f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  ✓ Saved confusion matrix: {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    probe_speakers = PROBE_SPEAKERS if PROBE_SPEAKERS else SPEAKERS
    num_roles = len(probe_speakers)

    print("=" * 70)
    print("LINEAR PROBE EXPERIMENT (CHUNKED)")
    print(f"Model: {MODEL_NAME}")
    print(f"Transcripts: {TRANSCRIPT_PATHS}")
    print(f"All speakers: {SPEAKERS}")
    print(f"Probing: {probe_speakers} ({num_roles}-class)")
    print("=" * 70)

    # ─── Parse transcripts (all speakers, so model sees full text) ───
    all_turns, speaker_to_id, turn_transcript_ids = parse_transcripts(
        TRANSCRIPT_PATHS, SPEAKERS
    )
    num_turns = len(all_turns)

    print(f"\nTotal turns: {num_turns}")
    for ti, path in enumerate(TRANSCRIPT_PATHS):
        count = sum(1 for t in turn_transcript_ids if t == ti)
        print(f"  {path}: {count} turns")
    for s, i in speaker_to_id.items():
        count = sum(1 for sid, _ in all_turns if sid == i)
        print(f"  {s}: {count} turns")

    # ─── Load model ───
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()

    # ─── Collect activations in chunks ───
    print("\nCollecting activations (chunked)...")
    activations, speaker_per_token, token_transcript_ids = collect_activations_chunked(
        model, tokenizer, all_turns, SPEAKERS, turn_transcript_ids
    )
    num_layers = len(activations)
    layers = list(range(num_layers))
    hidden_dim = activations[0].shape[1]
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")

    for s, i in speaker_to_id.items():
        count = sum(1 for sp in speaker_per_token if sp == i)
        print(f"  {s}: {count} tokens")

    # ─── Remap labels if probing a subset of speakers ───
    probe_ids = {speaker_to_id[s] for s in probe_speakers}
    old_to_new = {}
    for new_id, s in enumerate(probe_speakers):
        old_to_new[speaker_to_id[s]] = new_id

    speaker_per_token_remapped = []
    for label in speaker_per_token:
        if label in probe_ids:
            speaker_per_token_remapped.append(old_to_new[label])
        else:
            speaker_per_token_remapped.append(-1)

    if PROBE_SPEAKERS:
        excluded = [s for s in SPEAKERS if s not in probe_speakers]
        print(f"\n  Excluding {excluded} from probe (tokens marked -1)")
        for s in probe_speakers:
            count = sum(1 for sp in speaker_per_token_remapped if sp == old_to_new[speaker_to_id[s]])
            print(f"  {s} (id={old_to_new[speaker_to_id[s]]}): {count} tokens")

    # ─── Data splits (last 20% of each transcript = test) ───
    train_idx, test_idx, train_labels, test_labels = prepare_data_splits(
        speaker_per_token_remapped, token_transcript_ids,
        test_split=0.2, seed=42,
    )

    # ─── Train probes (real labels) ───
    print("\n" + "=" * 70)
    print("TRAINING PROBES (real labels)")
    print("=" * 70)

    results = []
    for layer in layers:
        print(f"\n  Layer {layer}:")
        acc, best_ep, cm = train_probe(
            activations[layer], train_idx, test_idx,
            train_labels, test_labels, num_roles
        )
        results.append({"layer": layer, "test_acc": acc, "best_epoch": best_ep, "cm": cm})

    # ─── Random label control ───
    print("\n" + "=" * 70)
    print("RANDOM LABEL CONTROL")
    print("=" * 70)

    random_results = []
    for layer in layers:
        rand_acc = train_probe_random_labels(
            activations[layer], train_idx, test_idx, num_roles
        )
        random_results.append({"layer": layer, "random_test_acc": rand_acc})
        print(f"  Layer {layer:2d}: {rand_acc:.3f}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    chance = 1.0 / num_roles
    print(f"\n{'Layer':>6} | {'Real':>8} | {'Random':>8} | {'Gap':>8}")
    print("-" * 40)
    for r, rr in zip(results, random_results):
        gap = r["test_acc"] - rr["random_test_acc"]
        print(f"{r['layer']:>6} | {r['test_acc']:>8.3f} | {rr['random_test_acc']:>8.3f} | {gap:>+8.3f}")

    print(f"\nChance baseline ({num_roles} speakers): {chance:.3f}")

    # ─── Plot ───
    probe_tag = "_".join(probe_speakers).lower()
    transcript_tag = "".join(TRANSCRIPT_PATHS)
    model_tag = MODEL_NAME.split("/")[-1]
    plot_results(results, random_results, num_turns, len(train_idx),
                 num_roles, MODEL_NAME,
                 output_path=f"probe_results_{transcript_tag}_{model_tag}_{probe_tag}.png")

    # ─── Confusion matrix for best layer ───
    best_result = max(results, key=lambda r: r["test_acc"])
    plot_confusion_matrix(
        best_result["cm"], probe_speakers, best_result["layer"],
        output_path=f"confusion_matrix_{transcript_tag}_{model_tag}_{probe_tag}_layer{best_result['layer']}.png"
    )


if __name__ == "__main__":
    main()
