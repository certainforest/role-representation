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

# 8 experiments: 2 examples × 4 conditions (2 binding + 2 controls)
#
# Conditions:
#   1.1 Query_Entity_Swap_Entity  (BINDING: swap names → answer changes)
#   1.2 Query_Attr_Swap_Attr      (BINDING: swap attrs → answer changes)
#   1.3 Query_Entity_Swap_Attr    (CONTROL: swap attrs, query entity → answer changes but binding ID same)
#   1.4 Query_Attr_Swap_Entity    (CONTROL: swap names, query attr → answer changes but binding ID same)
#
# Example 1: France / Thailand
# Example 2: Basketball / Soccer
#
# Hypothesis: binding-ID heads spike only on BINDING conditions, not CONTROL.

EXPERIMENTS = [
    # ── Example 1: France / Thailand ──────────────────────────────────────────
    {
        "key":        "ex1_1_1",
        "label":      "Ex1-1.1\nQuery Entity\nSwap Entity",
        "kind":       "BINDING",
        "example":    "ex1",
        "query_type": "entity",
        "swap_type":  "entity",
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
        "key":        "ex1_1_2",
        "label":      "Ex1-1.2\nQuery Attr\nSwap Attr",
        "kind":       "BINDING",
        "example":    "ex1",
        "query_type": "attr",
        "swap_type":  "attr",
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
        "key":        "ex1_1_3",
        "label":      "Ex1-1.3\nQuery Entity\nSwap Attr\n(control)",
        "kind":       "CONTROL",
        "example":    "ex1",
        "query_type": "entity",
        "swap_type":  "attr",
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
        "key":        "ex1_1_4",
        "label":      "Ex1-1.4\nQuery Attr\nSwap Entity\n(control)",
        "kind":       "CONTROL",
        "example":    "ex1",
        "query_type": "attr",
        "swap_type":  "entity",
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

    # ── Example 2: Basketball / Soccer ────────────────────────────────────────
    {
        "key":        "ex2_1_1",
        "label":      "Ex2-1.1\nQuery Entity\nSwap Entity",
        "kind":       "BINDING",
        "example":    "ex2",
        "query_type": "entity",
        "swap_type":  "entity",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: What sport does Alice like? Answer: Alice likes'
        ),
        "source_answer": " basketball",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n"I am Alice."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: What sport does Alice like? Answer: Alice likes'
        ),
        "base_answer": " soccer",
    },
    {
        "key":        "ex2_1_2",
        "label":      "Ex2-1.2\nQuery Attr\nSwap Attr",
        "kind":       "BINDING",
        "example":    "ex2",
        "query_type": "attr",
        "swap_type":  "attr",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: Who likes basketball? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like soccer."\n"I like basketball."\n'
            'Question: Who likes basketball? Answer:'
        ),
        "base_answer": " Bob",
    },
    {
        "key":        "ex2_1_3",
        "label":      "Ex2-1.3\nQuery Entity\nSwap Attr\n(control)",
        "kind":       "CONTROL",
        "example":    "ex2",
        "query_type": "entity",
        "swap_type":  "attr",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: What sport does Alice like? Answer: Alice likes'
        ),
        "source_answer": " basketball",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like soccer."\n"I like basketball."\n'
            'Question: What sport does Alice like? Answer: Alice likes'
        ),
        "base_answer": " soccer",
    },
    {
        "key":        "ex2_1_4",
        "label":      "Ex2-1.4\nQuery Attr\nSwap Entity\n(control)",
        "kind":       "CONTROL",
        "example":    "ex2",
        "query_type": "attr",
        "swap_type":  "entity",
        "source_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Alice."\n"I am Bob."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: Who likes basketball? Answer:'
        ),
        "source_answer": " Alice",
        "base_prompt": (
            'This is the transcript of a conversation.\n'
            '"I am Bob."\n"I am Alice."\n'
            '"I like basketball."\n"I like soccer."\n'
            'Question: Who likes basketball? Answer:'
        ),
        "base_answer": " Bob",
    },
]

BINDING_EXPS = [e for e in EXPERIMENTS if e["kind"] == "BINDING"]
CONTROL_EXPS = [e for e in EXPERIMENTS if e["kind"] == "CONTROL"]
