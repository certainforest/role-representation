import os 
import requests
from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal, Optional
from pathlib import Path
import time
import base64
from docx import Document

def send_openrouter_request(messages, 
                            model = 'google/gemini-2.5-pro',
                            provider_order = ['deepinfra/fp4'],
                            allow_fallbacks = True, 
                            temperature  =  0.0,
                            max_tokens = 4000,
                            attachment_path: Optional[str] = None): 
    '''
    a simple function that submits a single prompt to a selected model (defaults to gemini 2.5-pro)on openrouter.
    temperature is set to 0 by default for reproducibility. 
    
    Args:
        messages: List of message dicts in OpenAI format
        attachment_path: Optional path to a PDF or .docx file
            - PDFs: Sent as base64-encoded attachment (OpenRouter native support)
            - .docx: Text extracted and prepended to the last user message
    '''
    OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        'X-Title': 'speaker-tracking', 
        'HTTP-Referer': 'https://localhost'
    }

    # Handle attachments
    processed_messages = messages.copy()
    if attachment_path:
        attachment_path = Path(attachment_path)
        if not attachment_path.exists():
            raise FileNotFoundError(f"Attachment file not found: {attachment_path}")
        
        file_ext = attachment_path.suffix.lower()
        
        if file_ext == '.pdf':
            # Read PDF and encode as base64 data URL
            with open(attachment_path, 'rb') as f:
                pdf_data = base64.b64encode(f.read()).decode('utf-8')
                pdf_data_url = f"data:application/pdf;base64,{pdf_data}"
            
            # Add PDF as attachment to the last user message
            # OpenRouter uses OpenAI-compatible format with image_url type for documents
            if processed_messages and processed_messages[-1].get('role') == 'user':
                if 'content' not in processed_messages[-1]:
                    processed_messages[-1]['content'] = []
                elif isinstance(processed_messages[-1]['content'], str):
                    processed_messages[-1]['content'] = [{"type": "text", "text": processed_messages[-1]['content']}]
                
                processed_messages[-1]['content'].append({
                    "type": "image_url",
                    "image_url": {
                        "url": pdf_data_url
                    }
                })
            else:
                # Create new user message with PDF
                processed_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": pdf_data_url
                        }
                    }]
                })
        
        elif file_ext == '.docx':
            # Extract text from .docx and prepend to last user message
            doc = Document(attachment_path)
            docx_text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            
            if processed_messages and processed_messages[-1].get('role') == 'user':
                # Prepend extracted text to existing message
                existing_content = processed_messages[-1].get('content', '')
                if isinstance(existing_content, str):
                    processed_messages[-1]['content'] = f"Document content:\n{docx_text}\n\n---\n\n{existing_content}"
                else:
                    # If content is already a list (multimodal), prepend text
                    text_item = {"type": "text", "text": f"Document content:\n{docx_text}\n\n---\n\n"}
                    if isinstance(existing_content, list):
                        processed_messages[-1]['content'] = [text_item] + existing_content
                    else:
                        processed_messages[-1]['content'] = [text_item, existing_content]
            else:
                # Create new user message with extracted text
                processed_messages.append({
                    "role": "user",
                    "content": f"Document content:\n{docx_text}"
                })
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: .pdf, .docx")

    payload   =   {
        "model": model,
        "messages": processed_messages,
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
            r   =   requests.post(OPENROUTER_URL, headers   =   headers, json   =   payload, timeout   =   120)
            r.raise_for_status()
            final_response   = r.json()['choices'][0]['message']['content']
            reasoning = r.json()['choices'][0]['message']['reasoning']
            refusal = r.json()['choices'][0]['message']['refusal']
            provider = r.json()['provider']
            return final_response, reasoning, refusal, provider
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e