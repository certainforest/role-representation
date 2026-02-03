import json
import requests
#vllm serve Qwen/Qwen2-1.5B --port 5555
#vllm serve Qwen/Qwen2-1.5B-Instruct --port 5555
#vllm serve Qwen/Qwen2-7B --port 5555
#vllm serve Qwen/Qwen2-7B-Instruct --port 5555
#vllm serve Qwen/QWQ-32B --tensor-parallel-size 2 --port 5555
#vllm serve allenai/Olmo-3-1025-7B --port 5555
#vllm serve allenai/Olmo-3-1125-32B --tensor-parallel-size 2 --port 5555

# Read from JSON file (array of items)
def load_prompt_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Construct prompt from transcript and qa pair
def construct_prompt(transcript, qa_prompt):
    transcript_string = '\n'.join(f'"{line}"' for line in transcript)
    prompt = (
        f"This is a transcript of a conversation.\n\n"
        f"{transcript_string}\n\n"
        f"{qa_prompt}"
    )
    return prompt

def construct_prompt_labeled(transcript, qa_prompt):
    transcript_string = '\n'.join(f'{line}' for line in transcript)
    prompt = (
        f"This is a transcript of a conversation.\n\n"
        f"{transcript_string}\n\n"
        f"{qa_prompt}"
    )
    return prompt

# Generate response via vLLM API (chat endpoint)
def generate_response_instruct(VLLM_URL,model_name,prompt):
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

def generate_response(VLLM_URL, model_name, prompt):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
    }
    
    response = requests.post(VLLM_URL, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["text"].strip()

# Main execution
def main():
    model_name = "Qwen/Qwen2-1.5B-Instruct" #"Qwen/Qwen2-7B"   # eg. "Qwen/Qwen2-7B" or "Qwen/QWQ-32B"
    model_type = "instruct" #instruct or base
    prompt_path = "labeled_simple_prompt.json"
    result_path = f"{model_name.split('/')[1]}_{prompt_path}_results.json"
    data_items = load_prompt_data(prompt_path)
    print(f"Loaded {len(data_items)} data items from {prompt_path}")
    results = []
    for data in data_items:
        transcript = data["transcript"]
        qa_pairs = data["qa_pairs"]
        for qa in qa_pairs:
            if model_type == "instruct":
                prompt = construct_prompt_labeled(transcript, qa["prompt_instruct"])
                print("prompt:",prompt)
                VLLM_URL = "http://localhost:5555/v1/chat/completions"
                answer = generate_response_instruct(VLLM_URL, model_name, prompt)

                results.append({
                    "prompt": qa["prompt_instruct"],
                    "groundtruth": qa["groundtruth"],
                    "model_answer": answer
                })

            elif model_type == "base":
                prompt = construct_prompt(transcript, qa["prompt"])
                VLLM_URL = "http://localhost:5555/v1/completions"
                answer = generate_response(VLLM_URL, model_name, prompt)
            
                results.append({
                    "prompt": qa["prompt"],
                    "groundtruth": qa["groundtruth"],
                    "model_answer": answer
                })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()