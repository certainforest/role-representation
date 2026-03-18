"""
Shared experiment definitions and model configs for speaker binding circuit analysis.
"""

MODEL_CONFIGS = {
    "llama": {
        "name":  "meta-llama/Llama-3.1-8B-Instruct",
        "cache": "/mnt/ssd/aryawu/.cache/huggingface/hub",
    },
    "qwen": {
        "name":  "Qwen/Qwen3-8B",
        "cache": "/mnt/ssd/aryawu/.cache/huggingface/hub",
    },
}


# Example 1: France / Thailand
# Hypothesis: binding-ID heads spike only on BINDING conditions, not CONTROL.

SWAP_EXPERIMENTS = [
    # ── Example 1: France / Thailand ──────────────────────────────────────────
    {
        "key":         "Q_name_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "Swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n"I am Alice."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": " Thailand",
    },
    {
        "key":        "Q_country_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "Swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in Thailand."\n"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Bob",
    },
    {
        "key":        "Q_name_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "Swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in Thailand."\n"I live in France."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": " Thailand",
    },
    {
        "key":        "Q_country_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "Swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n"I am Alice."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Bob",
    },
]

NULL_EXPERIMENTS = [
    # ── Example 1: France / Thailand ──────────────────────────────────────────
    {
        "key":        "Q_name_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "Null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Claire."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        # Null model also predicts France (less confidently); use Thailand as fixed
        # foil so logit-diff = logit[France]-logit[Thailand] has a non-zero gap.
        "base_answer": " Thailand",
    },
    {
        "key":        "Q_country_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "Null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in Germany."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        # Null model may predict Alice or nobody; use Bob as fixed foil.
        "base_answer": " Bob",
    },
    {
        "key":        "Q_name_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "Null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in Germany."\n"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": "Germany",
    },
    {
        "key":        "Q_country_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "Null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Claire."\n"I am Bob."\n'
            '"I live in France."\n"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Claire",
    },
]

# ── HI experiments: greeting "Hi Bob!" breaks simple attribute-order pairing ──
# SOURCE: Alice(1) Bob(2) Alice(3="Hi Bob!") Bob(4=Thailand) Alice(5=France) → France
# BASE:   Alice(1) Bob(2) Alice(3="Hi Bob!") Bob(4=France)   Alice(5=Thailand) → Thailand
# Both models SUCCEED on this. Only attr positions (24,33) differ.
# Used to study how binding circuit changes with disrupted line parity.
HI_SWAP_EXPERIMENTS = [
    {
        "key":        "Q_name_hi_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "hi_swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n'
            '"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n'
            '"I am Alice."\n'
            '"Hi Alice!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": " Thailand",
    },
    {
        "key":        "Q_name_hi_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "hi_swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in France. What about you?"\n'
            '"I live in Thailand."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": " Thailand",
    },
    {
        "key":        "Q_country_hi_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "hi_swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in France. What about you?"\n'
            '"I live in Thailand."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Bob",
    },
    {
        "key":        "Q_country_hi_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "hi_swap",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n"I am Alice."\n'
            '"Hi Alice!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Bob",
    },
]


HI_NULL_EXPERIMENTS = [
    {
        "key":        "Q_name_hi_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "hi_null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n'
            '"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Claire."\n'
            '"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": " Thailand",  # fixed foil; null model also predicts France
    },
    {
        "key":        "Q_name_hi_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "hi_null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "source_answer": " France",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in Germany."\n'
            'Question: Where does Alice live? Answer:'
        ),
        "base_answer": " Germany",
    },
    {
        "key":        "Q_country_hi_bind",
        "kind":       "BINDING",
        "example":    "ex1",
        "type": "hi_null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in Germany."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Bob",  # fixed foil; null model may predict Alice or nobody
    },
    {
        "key":        "Q_country_hi_control",
        "kind":       "CONTROL",
        "example":    "ex1",
        "type": "hi_null",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Claire."\n"I am Bob."\n'
            '"Hi Bob!"\n'
            '"I live in Thailand. What about you?"\n'
            '"I live in France."\n'
            'Question: Who lives in France? Answer:'
        ),
        "base_answer": " Claire",
    },
]



# BINDING_EXPS = [e for e in  if e["kind"] == "BINDING"]
# CONTROL_EXPS = [e for e in  if e["kind"] == "CONTROL"]