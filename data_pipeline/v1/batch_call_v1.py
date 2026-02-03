import json
import requests

# vllm serve Qwen/Qwen2-1.5B --port 5555
# vllm serve Qwen/Qwen2-1.5B-Instruct --port 5555
# vllm serve Qwen/Qwen2-7B --port 5555
# vllm serve Qwen/Qwen2-7B-Instruct --port 5555
# vllm serve Qwen/QWQ-32B --tensor-parallel-size 2 --port 5555
# vllm serve allenai/Olmo-3-1025-7B --port 5555
# vllm serve allenai/Olmo-3-1125-32B --tensor-parallel-size 2 --port 5555

# Read from JSON file (new format with metadata and examples)
def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["examples"]


# Construct prompt from unlabeled transcript
def construct_prompt_unlabeled(transcript, qa_prompt, level):
    transcript_string = '\n'.join(f'"{line}"' for line in transcript)
    
    if level == "2.5":
        prompt = (
            f"This is a transcript of a three-people turn-taking conversation. Each line switches speakers; no two consecutive lines belong to the same speaker.\n\n"
            f"{transcript_string}\n\n"
            f"{qa_prompt}"
        )
    else:
        prompt = (
            f"This is a transcript of a two-people round-robin conversation. \n\n"
            f"{transcript_string}\n\n"
            f"{qa_prompt}"
        )
    return prompt


# Construct prompt from labeled transcript
def construct_prompt_labeled(transcript, qa_prompt):
    transcript_string = '\n'.join(f'{line}' for line in transcript)
    prompt = (
        f"This is a transcript of a conversation.\n\n"
        f"{transcript_string}\n\n"
        f"{qa_prompt}"
    )
    return prompt


# Generate response via vLLM API (chat endpoint for instruct models)
def generate_response_instruct(VLLM_URL, model_name, prompt):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    response = requests.post(VLLM_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


# Generate response via vLLM API (completion endpoint for base models)
def generate_response(VLLM_URL, model_name, prompt):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.0,
    }
    response = requests.post(VLLM_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["text"].strip()

# Check if model answer matches groundtruth (handles multiple acceptable answers separated by ;)
def check_answer(model_answer, groundtruth):
    if "</think>" in model_answer:
        model_answer = model_answer.split("</think>")[-1]
    model_answer = model_answer.lower().strip()
    
    # Split groundtruth by semicolon to get multiple acceptable answers
    acceptable_answers = [ans.strip().lower() for ans in groundtruth.split(';')]
    
    # Check if any acceptable answer appears in the model output
    for ans in acceptable_answers:
        if ans in model_answer:
            return True
    
    return False

# Compute accuracy by level and transcript type (example-level, all QAs must be correct)
def compute_accuracy(results):
    stats = {}
    
    for r in results:
        level = r["level"]
        transcript_type = r["transcript_type"]
        key = f"{level}_{transcript_type}"
        
        if key not in stats:
            stats[key] = {"correct": 0, "total": 0}
        
        stats[key]["total"] += 1
        if r["all_correct"]:
            stats[key]["correct"] += 1
    
    print("\n" + "=" * 60)
    print("ACCURACY RESULTS (all QAs per example must be correct)")
    print("=" * 60)
    
    for key, data in sorted(stats.items()):
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"{key}: {data['correct']}/{data['total']} = {acc:.1%}")
    
    return stats


# Main execution
def main():
    # Configuration
    model_name = "Qwen/Qwen2-7B"
    model_type = "base"  # "instruct" or "base"
    use_labeled = False  # True for labeled transcripts (Level 1), False for unlabeled (Level 2, 2.5)
    
    dataset_path = "gpt-5_speaker_attribution_dataset.json"
    transcript_type = "labeled" if use_labeled else "unlabeled"
    result_path = f"model_results_v1/{model_name.split('/')[1]}_{transcript_type}_results.json"
    
    examples = load_dataset(dataset_path)
    print(f"Loaded {len(examples)} examples from {dataset_path}")
    
    results = []
    
    for idx, example in enumerate(examples):
        level = example["level"]
        
        # Select transcript type
        if use_labeled:
            transcript = example["transcript_labeled"]
        else:
            transcript = example["transcript_unlabeled"]
        
        qa_pairs = example["qa_pairs"]
        
        # Collect all QA results for this example
        example_qa_results = []
        
        for qa in qa_pairs:
            # Construct prompt based on model type and transcript type
            if model_type == "instruct":
                if use_labeled:
                    prompt = construct_prompt_labeled(transcript, qa["prompt_instruct"])
                else:
                    prompt = construct_prompt_unlabeled(transcript, qa["prompt_instruct"], level)
                
                VLLM_URL = "http://localhost:5555/v1/chat/completions"
                answer = generate_response_instruct(VLLM_URL, model_name, prompt)
                qa_prompt_used = qa["prompt_instruct"]
                
            elif model_type == "base":
                if use_labeled:
                    prompt = construct_prompt_labeled(transcript, qa["prompt"])
                else:
                    prompt = construct_prompt_unlabeled(transcript, qa["prompt"], level)
                
                VLLM_URL = "http://localhost:5555/v1/completions"
                answer = generate_response(VLLM_URL, model_name, prompt)
                qa_prompt_used = qa["prompt"]
            
            correct = check_answer(answer, qa["groundtruth"])
            
            example_qa_results.append({
                "prompt": qa_prompt_used,
                "groundtruth": qa["groundtruth"],
                "model_answer": answer,
                "correct": correct
            })
        
        # Check if ALL QAs for this example are correct
        all_correct = all(qa_result["correct"] for qa_result in example_qa_results)
        
        results.append({
            "example_idx": idx,
            "level": level,
            "transcript_type": transcript_type,
            "topic": example.get("topic", "unknown"),
            "names": example["names"],
            "transcript": transcript,
            "qa_results": example_qa_results,
            "all_correct": all_correct,
            "num_correct": sum(1 for qa in example_qa_results if qa["correct"]),
            "num_total": len(example_qa_results)
        })
        
        # Progress indicator
        print(f"Processed example {idx + 1}/{len(examples)} - {level} - all_correct: {all_correct} ({sum(1 for qa in example_qa_results if qa['correct'])}/{len(example_qa_results)})")
    
    # Compute and print accuracy
    stats = compute_accuracy(results)
    
    # Save results
    output = {
        "model": model_name,
        "model_type": model_type,
        "transcript_type": transcript_type,
        "accuracy_stats": stats,
        "results": results
    }
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()