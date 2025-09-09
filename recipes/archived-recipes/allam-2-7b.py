import pandas as pd
import time
import os
import re
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from googletrans import Translator
from typing import List

# Load environment variables from .env file if present
load_dotenv()

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_language_name, get_iso2_code

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

# Global variables
_groq_client = None
_translator = Translator(service_urls=[
    'translate.google.com',
    'translate.google.co.kr',
])

def initialize_groq_client():
    """Initialize the Groq client with API key from .env or environment"""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY in a .env file or as an environment variable.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

def extract_text_from_brackets(text):
    """Extract text from square brackets, return empty string if not found"""
    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1).strip()
    return ""

def translate_text_with_retry(text, source_lang, target_lang, client, max_retries=5):
    """Translate text using Groq with retry logic and square bracket delimiters"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    prompt = f"""Please translate the following text from {source_lang_name} to {target_lang_name}. \nReturn ONLY the translation inside square brackets without any additional text, explanations, or formatting.\n\nText to translate: \"{text}\"\n\nTranslation:"""

    for attempt in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Only return the translated text inside square brackets."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )

            response_text = chat_response.choices[0].message.content.strip()
            translation = extract_text_from_brackets(response_text)

            if not translation:
                translation = response_text.strip()
                if translation.startswith('"') and translation.endswith('"'):
                    translation = translation[1:-1].strip()

            return translation

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for text: '{text}'. Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"All {max_retries} attempts failed for text: '{text}'")
                return ""

    return ""

def backtranslate_with_google(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """Backtranslate texts using googletrans one by one with ISO2 codes"""
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
    """Main processing function for the recipe"""
    print(f"Forward translation: Groq API | Backtranslation: Google Translate")

    client = initialize_groq_client()

    result_df = df.copy()
    result_df['translated'] = ""
    result_df['backtranslated'] = ""
    result_df['similarity_score'] = 0.0

    # Forward translations with Groq (rate limited)
    translations = []
    for i, text in enumerate(result_df['text']):
        translation = translate_text_with_retry(text, source_lang, target_lang, client)
        translations.append(translation)

        if i < len(result_df) - 1:
            time.sleep(120)  # 1 request every 2 minutes for Groq

    result_df['translated'] = translations

    # Back translations with Google Translate (no waiting, one by one)
    backtranslations = backtranslate_with_google(result_df['translated'].tolist(), source_lang, target_lang)
    result_df['backtranslated'] = backtranslations

    # Similarity scoring
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']) if row['backtranslated'] else 0.0,
        axis=1
    )

    return result_df

