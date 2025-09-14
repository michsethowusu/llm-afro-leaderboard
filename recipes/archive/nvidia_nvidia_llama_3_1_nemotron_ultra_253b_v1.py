import pandas as pd
import time
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import ctranslate2
from transformers import AutoTokenizer

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

# Initialize variables for models (will be loaded on demand)
similarity_model = None
backtranslation_tokenizer = None
translator = None  # Changed from backtranslation_model to translator

def load_backtranslation_models():
    """Load backtranslation models only when needed"""
    global similarity_model, backtranslation_tokenizer, translator
    
    if similarity_model is None:
        from sentence_transformers import SentenceTransformer
        similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
        similarity_model = SentenceTransformer(similarity_model_name)
        print("Loaded similarity model")
    
    if backtranslation_tokenizer is None or translator is None:
        # Load tokenizer
        backtranslation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
        print("Loaded NLLB tokenizer")
        
        # Load quantized CTranslate2 model
        model_path = "nllb-200-3.3B-float16-ct2"  # Path to your quantized model
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        translator = ctranslate2.Translator(model_path, device=device)
        print(f"Loaded quantized NLLB model on {device}")

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

def backtranslation_only(df, source_lang, target_lang):
    """Perform only backtranslation and similarity calculation using quantized NLLB model"""
    # Load models if not already loaded
    load_backtranslation_models()
    
    print(f"Backtranslation only: NLLB-3.3B (Quantized)")
    
    result_df = df.copy()
    result_df['backtranslated'] = ""
    result_df['similarity_score'] = 0.0

    # Backtranslations using quantized NLLB-3.3B
    print("Starting backtranslation with quantized NLLB-3.3B...")
    backtranslations = backtranslate_with_nllb(result_df['translated'].tolist(), source_lang, target_lang)
    result_df['backtranslated'] = backtranslations

    # Calculate similarity
    print("Calculating similarity scores...")
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']) if row['backtranslated'] else 0.0,
        axis=1
    )

    print("Backtranslation process completed!")
    return result_df

def backtranslate_with_nllb(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """Backtranslate texts using quantized NLLB model with CTranslate2"""
    # Load models if not already loaded
    load_backtranslation_models()
    
    # Convert language codes to NLLB format
    nllb_source = get_nllb_code(target_lang)  # Note: target_lang becomes source for backtranslation
    nllb_target = get_nllb_code(source_lang)  # Note: source_lang becomes target for backtranslation
    
    backtranslations = []
    
    # Precompute special tokens
    src_lang_token = backtranslation_tokenizer.convert_tokens_to_ids([nllb_source])[0]
    tgt_lang_token = backtranslation_tokenizer.convert_tokens_to_ids([nllb_target])[0]
    eos_token = backtranslation_tokenizer.convert_tokens_to_ids(["</s>"])[0]
    
    # Process in batches for efficiency
    batch_size = 4
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_tokens = []
        
        for text in batch_texts:
            if not text:  # Skip empty texts
                batch_tokens.append([])
                continue
                
            try:
                # Encode with source language code
                src_tokens = [src_lang_token] + backtranslation_tokenizer.encode(text, add_special_tokens=False) + [eos_token]
                tokens = backtranslation_tokenizer.convert_ids_to_tokens(src_tokens)
                batch_tokens.append(tokens)
            except Exception as e:
                print(f"Tokenization failed for text '{text}': {str(e)}")
                batch_tokens.append([])
        
        # Translate batch
        try:
            results = translator.translate_batch(
                batch_tokens,
                target_prefix=[[nllb_target]] * len(batch_tokens),
                batch_type="examples",
                max_batch_size=batch_size
            )
            
            # Decode results
            for j, result in enumerate(results):
                if result.hypotheses:
                    output_tokens = result.hypotheses[0]
                    # Skip the target language token at the beginning
                    translation = backtranslation_tokenizer.decode(
                        backtranslation_tokenizer.convert_tokens_to_ids(output_tokens[1:]), 
                        skip_special_tokens=True
                    )
                    backtranslations.append(translation)
                    
                    # Show progress
                    idx = i + j
                    if idx < len(texts):
                        print(f"Backtranslated {idx+1}/{len(texts)}: {texts[idx][:50]}... → {translation[:50]}...")
                else:
                    backtranslations.append("")
                    print(f"Backtranslation failed for text '{texts[i+j]}'")
                    
        except Exception as e:
            print(f"Batch translation failed: {str(e)}")
            # Fallback to individual translations
            for j in range(len(batch_texts)):
                try:
                    text = batch_texts[j]
                    if not text:
                        backtranslations.append("")
                        continue
                        
                    # Encode with source language code
                    src_tokens = [src_lang_token] + backtranslation_tokenizer.encode(text, add_special_tokens=False) + [eos_token]
                    tokens = backtranslation_tokenizer.convert_ids_to_tokens(src_tokens)
                    
                    # Translate individually
                    result = translator.translate_batch(
                        [tokens],
                        target_prefix=[[nllb_target]],
                        max_batch_size=1
                    )
                    
                    if result and result[0].hypotheses:
                        output_tokens = result[0].hypotheses[0]
                        translation = backtranslation_tokenizer.decode(
                            backtranslation_tokenizer.convert_tokens_to_ids(output_tokens[1:]), 
                            skip_special_tokens=True
                        )
                        backtranslations.append(translation)
                        print(f"Backtranslated {i+j+1}/{len(texts)}: {text[:50]}... → {translation[:50]}...")
                    else:
                        backtranslations.append("")
                        print(f"Backtranslation failed for text '{text}'")
                        
                except Exception as e2:
                    print(f"Individual backtranslation failed for text '{text}': {str(e2)}")
                    backtranslations.append("")
    
    return backtranslations

def calculate_similarity(original, backtranslated):
    """Calculate cosine similarity between original and backtranslated text"""
    # Load models if not already loaded
    load_backtranslation_models()
    
    try:
        if not original or not backtranslated:
            return 0.0

        clean_backtranslated = extract_text_from_brackets(backtranslated)
        if not clean_backtranslated:
            clean_backtranslated = backtranslated

        from sentence_transformers import util
        embeddings = similarity_model.encode([original, clean_backtranslated])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

def process_dataframe(df, source_lang, target_lang):
    """Main processing function - full process"""
    # Load models if not already loaded
    load_backtranslation_models()
    
    print(f"Forward translation: NVIDIA Build API | Backtranslation: NLLB-3.3B (Quantized)")
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

    # Backtranslations using quantized NLLB-3.3B
    print("Starting backtranslation with quantized NLLB-3.3B...")
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
