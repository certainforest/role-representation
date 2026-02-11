import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange
import nnsight

# ─── Seed ─────────────────────────────────────────────
random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed(12)

def get_prompts(flip_question=False, conversational=False):
    """
    flip_question=False → querying with Alice (subject)
    flip_question=True  → querying with France (object)
    conversational=False → simple statements
    conversational=True  → transcript style
    """
    if not conversational:
        if flip_question:# our setup
            PROMPT_BASE = """Alice lives in France.
Bob lives in Thailand.
Question: Who lives in France? Answer:"""
            PROMPT_SOURCE = """Charlie lives in Italy.
David lives in China.
Question: Who lives in Italy? Answer:"""
            BASE_ANSWER = "Alice"
            SOURCE_ANSWER = "Charlie"
        else:  #paper's setup
            PROMPT_BASE = """Alice lives in France.
Bob lives in Thailand.
Question: Where does Alice live? Answer:"""
            PROMPT_SOURCE = """Charlie lives in Italy.
David lives in China.
Question: Where does Charlie live? Answer:"""
            BASE_ANSWER = "France"
            SOURCE_ANSWER = "Italy"
    else:  # conversational
        if flip_question:
            PROMPT_BASE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in France."
"I live in Thailand."
Question: Who lives in France? Answer:"""
            PROMPT_SOURCE = """This is the transcript of a conversation.
"I am Charlie."
"I am David."
"I live in Italy."
"I live in China."
Question: Who lives in Italy? Answer:"""
            BASE_ANSWER = "Alice"
            SOURCE_ANSWER = "Charlie"
        else:  # conversational of paper's setup
            PROMPT_BASE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I live in France."
"I live in Thailand."
Question: Where does Alice live? Answer:"""
            PROMPT_SOURCE = """This is the transcript of a conversation.
"I am Charlie."
"I am David."
"I live in Italy."
"I live in China."
Question: Where does Charlie live? Answer:"""
            BASE_ANSWER = "France"
            SOURCE_ANSWER = "Italy"

    OUTPUT_FILE = f"flip_{flip_question}_conv_{conversational}.png"
    return PROMPT_BASE, PROMPT_SOURCE, BASE_ANSWER, SOURCE_ANSWER, OUTPUT_FILE

def get_prompts_sports(flip_question=False, conversational=False):
    """
    flip_question=False → querying with Alice (subject)
    flip_question=True  → querying with France (object)
    conversational=False → simple statements
    conversational=True  → transcript style
    """
    if not conversational:
        if flip_question:# our setup
            PROMPT_BASE = """Alice likes basketball.
Bob likes soccer.
Question: Who likes basketball? Answer:"""
            PROMPT_SOURCE = """Charlie likes tennis.
David likes swimming.
Question: Who likes tennis? Answer:"""
            BASE_ANSWER = "Alice"
            SOURCE_ANSWER = "Charlie"
        else:  #paper's setup
            PROMPT_BASE = """Alice likes basketball.
Bob likes soccer.
Question: Which sport does Alice like? Answer:"""
            PROMPT_SOURCE = """Charlie likes tennis.
David likes swimming.
Question: Which sport does Charlie like? Answer:"""
            BASE_ANSWER = "Basketball"
            SOURCE_ANSWER = "Tennis"
    else:  # conversational
        if flip_question:
            PROMPT_BASE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I like soccer."
Question: Who likes basketball? Answer:"""
            PROMPT_SOURCE = """This is the transcript of a conversation.
"I am Charlie."
"I am David."
"I like tennis."
"I like swimming."
Question: Who likes tennis? Answer:"""
            BASE_ANSWER = "Alice"
            SOURCE_ANSWER = "Charlie"
        else:  # conversational of paper's setup
            PROMPT_BASE = """This is the transcript of a conversation.
"I am Alice."
"I am Bob."
"I like basketball."
"I like soccer."
Question: Which sport does Alice like? Answer:"""
            PROMPT_SOURCE = """This is the transcript of a conversation.
"I am Charlie."
"I am David."
"I like tennis."
"I like swimming."
Question: Which sport does Charlie like? Answer:"""
            BASE_ANSWER = "Basketball"
            SOURCE_ANSWER = "Tennis"

    OUTPUT_FILE = f"flip_{flip_question}_conv_{conversational}_sport.png"
    return PROMPT_BASE, PROMPT_SOURCE, BASE_ANSWER, SOURCE_ANSWER, OUTPUT_FILE

def run_one(model,MODEL_NAME, PROMPT_BASE, PROMPT_SOURCE, BASE_ANSWER, SOURCE_ANSWER, OUTPUT_FILE):
    # ─── Token IDs ──────────────────────────────────────
    source_answer_id = model.tokenizer.encode(f" {SOURCE_ANSWER}", add_special_tokens=False)[0]
    base_answer_id   = model.tokenizer.encode(f" {BASE_ANSWER}", add_special_tokens=False)[0]

    source_prompt_ids = model.tokenizer(PROMPT_SOURCE).input_ids
    base_prompt_ids   = model.tokenizer(PROMPT_BASE).input_ids

    # ─── Define context tokens ──────────────────────────
    question_pos = PROMPT_BASE.lower().find("question")
    context_tokens = model.tokenizer(PROMPT_BASE[:question_pos]).input_ids
    context_token_indices = list(range(len(context_tokens)))
    print(f"Patching {len(context_token_indices)} context tokens (everything before 'Question')")

    # ─── Get activations ───────────────────────────────
    print("\nGetting activations...")

    source_activations = []
    with torch.no_grad():
        with model.trace(PROMPT_SOURCE) as tracer:
            for layer in model.model.layers:
                source_activations.append(layer.output[0].save())
            source_logits = model.output.logits.save()

    source_pred_id = source_logits.argmax(dim=-1)[0, -1].item()
    source_pred = model.tokenizer.decode(source_pred_id)
    assert source_pred_id == source_answer_id, f"Source prompt should predict '{SOURCE_ANSWER}' but got {source_pred!r}"

    base_activations = []
    with torch.no_grad():
        with model.trace(PROMPT_BASE) as tracer:
            for layer in model.model.layers:
                base_activations.append(layer.output[0].save())
            base_logits = model.output.logits.save()

    base_pred_id = base_logits.argmax(dim=-1)[0, -1].item()
    base_pred = model.tokenizer.decode(base_pred_id)
    assert base_pred_id == base_answer_id, f"Base prompt should predict '{BASE_ANSWER}' but got {base_pred!r}"

    # ─── Patch context tokens ──────────────────────────
    is_2d = len(source_activations[0].shape) == 2
    patching_results = []

    for layer_index in trange(model.config.num_hidden_layers, desc="Patching"):
        patching_per_layer = []
        for token_index in context_token_indices:
            with torch.no_grad():
                with model.trace(PROMPT_BASE) as tracer:
                    if is_2d:
                        model.model.layers[layer_index].output[0][token_index, :] = \
                            source_activations[layer_index][token_index, :]
                    else:
                        model.model.layers[layer_index].output[0][0, token_index, :] = \
                            source_activations[layer_index][0, token_index, :]

                    patched_logits = model.output.logits.save()
                    patched_probs = patched_logits[:, -1].softmax(dim=-1)
                    src_prob = patched_probs[0, source_answer_id].item()
                    base_prob = patched_probs[0, base_answer_id].item()
                    patching_per_layer.append(src_prob - base_prob)
        patching_results.append(patching_per_layer)

    patching_results = np.array(patching_results, dtype=float)
    print(f"Patching results shape: {patching_results.shape}")

    # ─── Build token labels ─────────────────────────────
    token_strings = [model.tokenizer.decode([base_prompt_ids[i]]).replace("\n", "\\n")
                    for i in context_token_indices]

    # ─── Plot ───────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.35)

    # Heatmap
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(patching_results, aspect="auto", cmap="RdBu_r",
                origin="lower", vmin=-0.5, vmax=0.5)
    fig.colorbar(im, ax=ax, label=f"P({SOURCE_ANSWER}) - P({BASE_ANSWER})")

    ax.set_xticks(range(len(token_strings)))
    ax.set_xticklabels(token_strings, rotation=90, ha="center", fontsize=6)
    ax.set_yticks(range(0, patching_results.shape[0], 2))
    ax.set_yticklabels(range(0, patching_results.shape[0], 2))
    ax.set_xlabel("Context token positions")
    ax.set_ylabel("Layer")
    ax.set_title(f"Activation Patching: replace context tokens in BASE with SOURCE\n"
                f"Red = shifted toward {SOURCE_ANSWER} | Blue = shifted toward {BASE_ANSWER}")

    # Prompt panel
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    source_display = PROMPT_SOURCE.strip().replace("\n", "\n    ")
    base_display = PROMPT_BASE.strip().replace("\n", "\n    ")
    prompt_text = (f"SOURCE (activations FROM this prompt): prediction: {source_pred!r}\n"
                f"    {source_display}\n\n"
                f"BASE (activations patched INTO this prompt): prediction: {base_pred!r}\n"
                f"    {base_display}")
    ax_text.text(0.02, 0.95, prompt_text, transform=ax_text.transAxes,
                fontsize=7, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_FILE}")

def main():
    MODEL_NAME = "Qwen/Qwen3-8B"
    # ─── Load model ─────────────────────────────────────
    model = nnsight.LanguageModel(MODEL_NAME, device_map="auto")

    for flip_question in [False, True]:
        for conversational in [False, True]:
            PROMPT_BASE, PROMPT_SOURCE, BASE_ANSWER, SOURCE_ANSWER, OUTPUT_FILE = get_prompts(flip_question, conversational)
            OUTPUT_FILE = f"{MODEL_NAME.replace('/', '_')}_{OUTPUT_FILE}"
            run_one(model, MODEL_NAME, PROMPT_BASE, PROMPT_SOURCE, BASE_ANSWER, SOURCE_ANSWER, OUTPUT_FILE)

if __name__ == "__main__":
    main()