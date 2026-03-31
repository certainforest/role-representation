"""
Zero-shot evaluation of Llama-3.1-8B-Instruct on FriendsQA via NDIF remote inference.

Usage:
    # Full test set
    python python/src/examples/run_llama_eval.py \
        --data dat/friendsqa_tst.json \
        --output results/

    # Quick smoke test (5 questions)
    python python/src/examples/run_llama_eval.py \
        --data dat/friendsqa_tst.json \
        --output results/ \
        --max-questions 5
"""

import argparse
import json
import os
import sys
from collections import Counter
import re
import string

from tqdm import tqdm
from nnsight import LanguageModel


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_friendsqa(path):
    """Load FriendsQA JSON and return flat list of QA items + raw dataset."""
    with open(path) as f:
        raw = json.load(f)

    items = []
    for article in raw['data']:
        para = article['paragraphs'][0]
        # Build dialogue string
        lines = []
        for utt in para['utterances:']:
            speaker = utt['speakers'][0] if utt['speakers'] else ''
            text = utt['utterance']
            lines.append(f"{speaker}: {text}" if speaker else text)
        dialogue = '\n'.join(lines)

        for qa in para['qas']:
            items.append({
                'id': qa['id'],
                'dialogue': dialogue,
                'question': qa['question'],
                'answers': [a['answer_text'] for a in qa['answers']],
            })

    return raw, items


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You answer questions about TV show conversations. "
    "Reply with a short phrase or name extracted directly from the dialogue. "
    "Do not explain or repeat the question — just give the answer."
)


def make_prompt(dialogue, question, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Dialogue:\n{dialogue}\n\nQuestion: {question}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Evaluation (mirrors evaluate_whole_context.py)
# ---------------------------------------------------------------------------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude or ch == '_')

    def lower(text):
        return text.lower()

    def remove_underline(text):
        return text.replace('_', ' ')

    return remove_underline(white_space_fix(remove_articles(remove_punc(lower(s)))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_predictions(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for para in article['paragraphs']:
            for qa in para['qas']:
                total += 1
                qid = qa['id']
                if qid not in predictions:
                    print(f"Warning: unanswered question {qid}", file=sys.stderr)
                    continue
                gold_answers = [a['answer_text'] for a in qa['answers']]
                pred = predictions[qid]
                exact_match += max(exact_match_score(pred, g) for g in gold_answers)
                f1 += max(f1_score(pred, g) for g in gold_answers)

    return {
        'exact_match': 100.0 * exact_match / total,
        'f1': 100.0 * f1 / total,
        'total': total,
        'answered': len(predictions),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(items, model, max_new_tokens, resume_from=None):
    """Run NDIF remote generation for each item. Returns {qid: predicted_answer}."""
    predictions = dict(resume_from) if resume_from else {}
    remaining = [it for it in items if it['id'] not in predictions]

    print(f"Running inference on {len(remaining)} questions "
          f"({len(predictions)} already cached)...")

    for item in tqdm(remaining):
        prompt = make_prompt(item['dialogue'], item['question'], model.tokenizer)
        input_ids = model.tokenizer(prompt, return_tensors='pt')['input_ids']
        input_len = input_ids.shape[1]

        with model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            remote=True,
            scan=False,
        ) as _:
            output = model.generator.output.save()

        generated_tokens = output.value[0][input_len:]
        raw_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Take only the first non-empty line as the answer
        answer = next(
            (line.strip() for line in raw_text.splitlines() if line.strip()),
            raw_text.strip()
        )
        predictions[item['id']] = answer

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot Llama-3.1-8B-Instruct evaluation on FriendsQA'
    )
    parser.add_argument('--data', required=True, help='Path to FriendsQA JSON (test/dev)')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct',
                        help='HuggingFace model ID')
    parser.add_argument('--max-new-tokens', type=int, default=30,
                        help='Max tokens to generate per answer')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Limit number of questions (for smoke testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing predictions file if present')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pred_path = os.path.join(args.output, 'llama_predictions.json')
    score_path = os.path.join(args.output, 'llama_scores.json')

    # Load data
    print(f"Loading data from {args.data}...")
    raw_dataset, items = load_friendsqa(args.data)
    if args.max_questions:
        items = items[:args.max_questions]
    print(f"  {len(items)} questions loaded")

    # Resume from existing predictions if requested
    existing_preds = {}
    if args.resume and os.path.exists(pred_path):
        with open(pred_path) as f:
            existing_preds = json.load(f)
        print(f"  Resuming from {len(existing_preds)} existing predictions")

    # Load model (dispatch=True for NDIF remote)
    print(f"Loading model {args.model}...")
    model = LanguageModel(args.model, dispatch=True)

    # Run inference
    predictions = run_inference(
        items, model,
        max_new_tokens=args.max_new_tokens,
        resume_from=existing_preds,
    )

    # Save predictions
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {pred_path}")

    # Evaluate
    scores = evaluate_predictions(raw_dataset['data'], predictions)
    print(f"\nResults:")
    print(f"  F1:            {scores['f1']:.2f}")
    print(f"  Exact Match:   {scores['exact_match']:.2f}")
    print(f"  Questions:     {scores['answered']} / {scores['total']}")

    with open(score_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to {score_path}")


if __name__ == '__main__':
    main()
