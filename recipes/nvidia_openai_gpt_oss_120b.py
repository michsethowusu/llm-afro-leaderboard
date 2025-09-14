import pandas as pd
import time
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import torch
import numpy as np

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
    backtranslations = backtranslate_with_nllb_batch(result_df['translated'].tolist(), source_lang, target_lang)
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

    # Calculate similarity in batches for better performance
    print("Calculating similarity scores...")
    similarity_scores = calculate_similarity_batch(result_df['text'].tolist(), result_df['backtranslated'].tolist())
    result_df['similarity_score'] = similarity_scores

    print("Similarity calculation completed!")
    return result_df

def backtranslate_with_nllb_batch(texts: List[str], source_lang: str, target_lang: str, batch_size: int = 16) -> List[str]:
    """Backtranslate texts using NLLB-3B model with batch processing"""
    # Get models from cache
    backtranslation_models = get_backtranslation_model()
    tokenizer = backtranslation_models['tokenizer']
    model = backtranslation_models['model']
    device = backtranslation_models['device']
    
    # Convert language codes to NLLB format
    nllb_source = get_nllb_code(target_lang)  # Note: target_lang becomes source for backtranslation
    nllb_target = get_nllb_code(source_lang)  # Note: source_lang becomes target for backtranslation
    
    backtranslations = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Process texts in batches
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_indices = list(range(batch_idx, min(batch_idx + batch_size, len(texts))))
        
        # Filter out empty texts
        non_empty_indices = [i for i, text in enumerate(batch_texts) if text and text.strip()]
        non_empty_texts = [text for text in batch_texts if text and text.strip()]
        
        if not non_empty_texts:
            # All texts in this batch are empty
            backtranslations.extend([""] * len(batch_texts))
            continue
            
        try:
            # Set source language
            tokenizer.src_lang = nllb_source
            
            # Encode the batch of texts
            inputs = tokenizer(
                non_empty_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128,  # Reduced max_length for faster processing
                return_attention_mask=True
            ).to(device)
            
            # Generate translations
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_target),
                    max_length=150,  # Reduced max_length for faster processing
                    num_beams=4,     # Reduced from default (often 5-6) for speed
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode the translations
            batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Map translations back to their original positions
            batch_results = [""] * len(batch_texts)
            for idx, translation in zip(non_empty_indices, batch_translations):
                batch_results[idx] = translation
            
            backtranslations.extend(batch_results)
            
            # Show progress
            print(f"Backtranslated batch {batch_idx//batch_size + 1}/{total_batches}: "
                  f"{batch_idx + 1}-{min(batch_idx + batch_size, len(texts))}/{len(texts)}")
            
        except Exception as e:
            print(f"NLLB backtranslation failed for batch starting at index {batch_idx}: {str(e)}")
            backtranslations.extend([""] * len(batch_texts))
            
        # Clear GPU cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return backtranslations

def calculate_similarity_batch(originals: List[str], backtranslateds: List[str], batch_size: int = 64) -> List[float]:
    """Calculate cosine similarity between original and backtranslated texts in batches"""
    try:
        # Get similarity model from cache
        similarity_model = get_similarity_model()
        
        # Preprocess texts
        clean_backtranslateds = []
        for bt in backtranslateds:
            clean_bt = extract_text_from_brackets(bt)
            if not clean_bt:
                clean_bt = bt
            clean_backtranslateds.append(clean_bt)
        
        # Calculate similarities in batches
        similarities = []
        for i in range(0, len(originals), batch_size):
            batch_originals = originals[i:i+batch_size]
            batch_backtranslateds = clean_backtranslateds[i:i+batch_size]
            
            # Filter out pairs where either text is empty
            valid_indices = []
            valid_originals = []
            valid_backtranslateds = []
            
            for j, (orig, bt) in enumerate(zip(batch_originals, batch_backtranslateds)):
                if orig and bt:
                    valid_indices.append(j)
                    valid_originals.append(orig)
                    valid_backtranslateds.append(bt)
            
            if not valid_originals:
                # All pairs in this batch are invalid
                similarities.extend([0.0] * len(batch_originals))
                continue
            
            # Encode texts
            orig_embeddings = similarity_model.encode(valid_originals, convert_to_tensor=True)
            bt_embeddings = similarity_model.encode(valid_backtranslateds, convert_to_tensor=True)
            
            # Calculate similarities
            from sentence_transformers import util
            batch_similarities = util.pytorch_cos_sim(orig_embeddings, bt_embeddings).diag().cpu().numpy()
            
            # Map similarities back to their original positions
            batch_results = [0.0] * len(batch_originals)
            for idx, sim in zip(valid_indices, batch_similarities):
                batch_results[idx] = sim
            
            similarities.extend(batch_results)
            
            # Show progress
            print(f"Calculated similarity for batch {i//batch_size + 1}/{(len(originals) + batch_size - 1) // batch_size}")
        
        return similarities
        
    except Exception as e:
        print(f"Error calculating similarity in batch: {str(e)}")
        return [0.0] * len(originals)

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
    backtranslations = backtranslate_with_nllb_batch(result_df['translated'].tolist(), source_lang, target_lang)
    result_df['backtranslated'] = backtranslations

    # Calculate similarity
    print("Calculating similarity scores...")
    result_df['similarity_score'] = calculate_similarity_batch(result_df['text'].tolist(), result_df['backtranslated'].tolist())

    print("Translation process completed!")
    return result_df

# Add a function to clear models when needed
def clear_models():
    """Clear cached models to free memory"""
    clear_model_cache()
