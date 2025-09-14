import pandas as pd
import time
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

# Initialize NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_BUILD_API_KEY")
)

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_language_name, get_iso2_code, get_nllb_code
from model_cache import get_backtranslation_model, get_similarity_model, clear_model_cache

# Remove global model variables since we'll use the cache

def extract_text_from_brackets(text):
    """Extract text from square brackets, return empty string if not found"""
    match = re.search(r'\[(.*?)\]', text, flags=re.S)
    if match:
        return match.group(1).strip()
    return ""

def translate_text_with_nvidia(text, source_lang, target_lang, max_retries=5):
    """Translate text using NVIDIA Build API via OpenAI client"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    prompt = f"Translate the following {source_lang_name} text into {target_lang_name} and return ONLY the translation inside square brackets:\n\n{text}"

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="abacusai/dracarys-llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                top_p=0.95,
                max_tokens=2024,
                stream=False
            )
            
            # Directly get the response content like in your working example
            response_text = completion.choices[0].message.content
            
            # Simply return the response as-is without any extraction
            return response_text.strip()
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed for text '{text}': {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""

def forward_translation_only(df, source_lang, target_lang):
    """Perform only forward translation"""
    print(f"Forward translation only: NVIDIA Build API")
    print(f"Rate limiting: 38 requests per minute (~1.58 seconds between requests)")

    result_df = df.copy()
    result_df['translated'] = ""

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
    return result_df

def backtranslation_only_no_similarity(df, source_lang, target_lang):
    """Perform only backtranslation without similarity calculation"""
    print(f"Backtranslation only (no similarity): NLLB-3.3B")
    
    result_df = df.copy()
    
    # Check if backtranslated column exists, if not create it
    if 'backtranslated' not in result_df.columns:
        result_df['backtranslated'] = ""
    
    # Backtranslations using NLLB-3B
    print("Starting backtranslation with NLLB-3.3B...")
    backtranslations = backtranslate_with_nllb(result_df['translated'].tolist(), source_lang, target_lang)
    result_df['backtranslated'] = backtranslations

    print("Backtranslation process completed (without similarity)!")
    return result_df

def similarity_only(df, source_lang, target_lang):
    """Perform only similarity calculation on existing backtranslated text"""
    print(f"Similarity calculation only")
    
    result_df = df.copy()
    
    # Check if similarity_score column exists, if not create it
    if 'similarity_score' not in result_df.columns:
        result_df['similarity_score'] = 0.0

    # Calculate similarity
    print("Calculating similarity scores...")
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']) if row['backtranslated'] else 0.0,
        axis=1
    )

    print("Similarity calculation completed!")
    return result_df

def backtranslate_with_nllb(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """Backtranslate texts using NLLB-3B model"""
    # Get models from cache
    backtranslation_models = get_backtranslation_model()
    tokenizer = backtranslation_models['tokenizer']
    model = backtranslation_models['model']
    device = backtranslation_models['device']
    
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
            tokenizer.src_lang = nllb_source
            
            # Encode the text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            
            # Generate translation
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_target),
                max_length=512
            )
            
            # Decode the translation
            backtranslation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
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

        # Get similarity model from cache
        similarity_model = get_similarity_model()
        
        from sentence_transformers import util
        embeddings = similarity_model.encode([original, clean_backtranslated])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

def process_dataframe(df, source_lang, target_lang):
    """Main processing function - full process"""
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

# Add a function to clear models when needed
def clear_models():
    """Clear cached models to free memory"""
    clear_model_cache()
