import os 
import requests
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import time


models = {
    # 'qwen3-1.7b': 'Qwen/Qwen3-1.7B', - not on openrouter
    # 'qwen3-4b': 'qwen/qwen3-4b:free', - openrouter limits free models
    # 'qwen3-8b': 'qwen/qwen3-8B',
    # 'qwen3-14b': 'qwen/qwen3-14B',
    # 'qwen3-32b': 'qwen/qwen3-32B',
    # 'gemma2b-it': 'google/gemma-2-9b-it'
    'olmo3-7b-instruct': 'allenai/olmo-3.1-7b-instruct',
    'olmo3.1-32b-instruct': 'allenai/olmo-3.1-32b-instruct'
}

# tk: was this an old function, seems like I was going to add an option for few-shot :) 
def generate_chat_template_input(
        input: str,
        answer: str, 
        is_one_shot = True,
        system_prompt: str = (
        'You will be provided with a multiple-choice question, as well as a list of '
        'possible answer choices. Respond exactly with: “The correct answer is {X}“, '
        'substituting in X with the code for the correct choice.'
     )
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    if is_one_shot: 
        messages.append({'role': 'user', 'content': 'What is 5+10?\nA. 5\nB. 6\nC. 15\nD. 20'})
        messages.append({'role': 'assistant', 'content': 'The correct answer is C'})
    messages.append({'role': 'user', 'content': input})
    messages.append({'role': 'assistant', 'content': 'The correct answer is'})

    return messages, answer



def send_slack(text): 
    '''basic slack request w/ webhook'''
    url = os.getenv('SLACK_WEBHOOK_URL')
    msg = requests.post(url, json = {'text': text})
    return msg
    
def send_openrouter_request(messages, 
                            model = 'google/gemini-2.5-pro',
                            provider_order = ['deepinfra/fp4'],
                            allow_fallbacks = True, 
                            temperature  =  0.0,
                            max_tokens = 4000): 
    '''
    a simple function that submits a single prompt to a selected model (defaults to gemini 2.5-pro)on openrouter.
    temperature is set to 0 by default for reproducibility. 
    '''
    OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        'X-Title': 'speaker-tracking', 
        'HTTP-Referer': 'https://localhost'
    }

    payload   =   {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if provider_order is not None:
        payload["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks
        }

    for attempt in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers   =   headers, json   =   payload, timeout   =   120)
            r.raise_for_status()
            final_response = r.json()['choices'][0]['message']['content']
            reasoning = r.json()['choices'][0]['message']['reasoning']
            refusal = r.json()['choices'][0]['message']['refusal']
            provider = r.json()['provider']
            return final_response, reasoning, refusal, provider
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e