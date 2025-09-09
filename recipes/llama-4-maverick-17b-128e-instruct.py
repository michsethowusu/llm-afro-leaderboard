import pandas as pd
import time
import os
import re
import requests
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List

# Load environment variables from .env file
load_dotenv()

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_language_name, get_iso2_code, get_nllb_code

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

# Initialize NLLB model for backtranslation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} for NLLB backtranslation")

# Load NLLB model and tokenizer for backtranslation
backtranslation_model_name = "facebook/nllb-200-3.3B"
backtranslation_tokenizer = AutoTokenizer.from_pretrained(backtranslation_model_name)
backtranslation_model = AutoModelForSeq2SeqLM.from_pretrained(backtranslation_model_name).to(device)

def extract_text_from_brackets(text):
    """Extract text from square brackets, return empty string if not found"""
    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1).strip()
    return ""

def translate_text_with_nvidia(text, source_lang, target_lang, max_retries=5):
    """Translate text using NVIDIA Build API via HTTP request"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    prompt = f"""Please translate the following text from {source_lang_name} to {target_lang_name}. 
Return ONLY the translation inside square brackets.

Text to translate: "{text}"\n\nTranslation:"""

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    api_key = os.getenv("NVIDIA_BUILD_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json['choices'][0]['message']['content'].strip()
            translation = extract_text_from_brackets(response_text)
            if not translation:
                translation = response_text.strip()
            return translation
        except Exception as e:
            print(f"Attempt {attempt+1} failed for text '{text}': {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""

def backtranslate_with_nllb(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """Backtranslate texts using NLLB-3B model"""
    # Convert language codes to NLLB format
    nllb_source = get_nllb_code(target_lang)  # Note: target_lang becomes source for backtranslation
    nllb_target = get_nllb_code(source_lang)  # Note: source_lang becomes target for backtranslation
    
    backtranslations = []
    
    for i, text in enumerate(texts):
        if not text:  # Skip empty texts
            backtranslations.append("")
            continue
            
        try:
            # Set source language
            backtranslation_tokenizer.src_lang = nllb_source
            
            # Encode the text
            inputs = backtranslation_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            
            # Generate translation
            generated_tokens = backtranslation_model.generate(
                **inputs,
                forced_bos_token_id=backtranslation_tokenizer.convert_tokens_to_ids(nllb_target),
                max_length=512
            )
            
            # Decode the translation
            backtranslation = backtranslation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            backtranslations.append(backtranslation)
            
            # Show progress
            print(f"Backtranslated {i+1}/{len(texts)}: {text[:50]}... → {backtranslation[:50]}...")
            
        except Exception as e:
            print(f"NLLB backtranslation failed for text '{text}': {str(e)}")
            backtranslations.append("")
            
        # Add a small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    return backtranslations

def calculate_similarity(original, backtranslated):
    """Calculate cosine similarity between original and backtranslated text"""
    try:
        if not original or not backtranslated:
            return 0.0

        clean_backtranslated = extract_text_from_brackets(backtranslated)
        if not clean_backtranslated:
            clean_backtranslated = backtranslated

        embeddings = similarity_model.encode([original, clean_backtranslated])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

def process_dataframe(df, source_lang, target_lang):
    """Main processing function"""
    print(f"Forward translation: NVIDIA Build API | Backtranslation: NLLB-3.3B")
    print(f"Rate limiting: 38 requests per minute (~1.58 seconds between requests)")

    result_df = df.copy()
    result_df['translated'] = ""
    result_df['backtranslated'] = ""
    result_df['similarity_score'] = 0.0

    # Calculate delay between requests to achieve 38 requests per minute
    delay_between_requests = 60 / 38  # Approximately 1.58 seconds

    # Forward translations with rate limiting
    translations = []
    total_texts = len(result_df)
    
    for i, text in enumerate(result_df['text']):
        print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
        translation = translate_text_with_nvidia(text, source_lang, target_lang)
        translations.append(translation)
        
        # Show translation result
        if translation:
            print(f"  → {translation[:50]}...")
        else:
            print("  → [Translation failed]")
        
        # Rate limiting: wait before next request (except after the last one)
        if i < total_texts - 1:
            print(f"Waiting {delay_between_requests:.2f} seconds before next request...")
            time.sleep(delay_between_requests)

    result_df['translated'] = translations

    # Backtranslations using NLLB-3B
    print("Starting backtranslation with NLLB-3.3B...")
    backtranslations = backtranslate_with_nllb(result_df['translated'].tolist(), source_lang, target_lang)
    result_df['backtranslated'] = backtranslations

    # Calculate similarity
    print("Calculating similarity scores...")
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']) if row['backtranslated'] else 0.0,
        axis=1
    )

    print("Translation process completed!")
    return result_df
