import pandas as pd
import time
import os
import re
import requests
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from googletrans import Translator
from typing import List

# Load environment variables from .env file
load_dotenv()

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_language_name, get_iso2_code

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

# Global translator for Google backtranslation
_translator = Translator(service_urls=['translate.google.com', 'translate.google.co.kr'])

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

def backtranslate_with_google(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """Backtranslate texts using Google Translate one by one"""
    source_iso2 = get_iso2_code(source_lang)
    target_iso2 = get_iso2_code(target_lang)

    backtranslations = []
    for text in texts:
        try:
            if text:
                back = _translator.translate(text, src=target_iso2, dest=source_iso2)
                backtranslations.append(back.text)
            else:
                backtranslations.append("")
        except Exception as e:
            print(f"Google Translate backtranslation failed for text '{text}': {str(e)}")
            backtranslations.append("")
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
    print(f"Forward translation: NVIDIA Build API | Backtranslation: Google Translate")

    result_df = df.copy()
    result_df['translated'] = ""
    result_df['backtranslated'] = ""
    result_df['similarity_score'] = 0.0

    # Forward translations (rate limited to 1 request every 2 minutes)
    translations = []
    for i, text in enumerate(result_df['text']):
        translation = translate_text_with_nvidia(text, source_lang, target_lang)
        translations.append(translation)
        if i < len(result_df) - 1:
            time.sleep(120)

    result_df['translated'] = translations

    # Backtranslations using Google Translate (one by one, no wait)
    backtranslations = backtranslate_with_google(result_df['translated'].tolist(), source_lang, target_lang)
    result_df['backtranslated'] = backtranslations

    # Calculate similarity
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']) if row['backtranslated'] else 0.0,
        axis=1
    )

    return result_df

