import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_nllb_code

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

translation_model_name = "facebook/nllb-200-distilled-600M"
similarity_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load models (only once)
def load_models():
    print("Loading translation model...")
    translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to(device)

    print("Loading similarity model...")
    similarity_model = SentenceTransformer(similarity_model_name)
    
    return translation_tokenizer, translation_model, similarity_model

# Load models at module level
translation_tokenizer, translation_model, similarity_model = load_models()

def translate_text(text, source_lang="eng", target_lang="twi"):
    """Translate text using NLLB-200-600M model"""
    try:
        # Convert language codes to NLLB format
        nllb_source = get_nllb_code(source_lang)
        nllb_target = get_nllb_code(target_lang)
        
        # Set source language
        translation_tokenizer.src_lang = nllb_source
        
        # Encode the text
        inputs = translation_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        # Generate translation
        generated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids(nllb_target),
            max_length=512
        )
        
        # Decode the translation
        return translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error translating text: {text}. Error: {str(e)}")
        return ""

def calculate_similarity(original, backtranslated):
    """Calculate cosine similarity between original and backtranslated text"""
    try:
        embeddings = similarity_model.encode([original, backtranslated])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

def process_dataframe(df, source_lang="eng", target_lang="twi"):
    """Main processing function for the recipe"""
    print(f"Translating from {source_lang} to {target_lang} using NLLB-200-600M")
    
    # Add new columns
    df['translated'] = df['text'].apply(
        lambda x: translate_text(x, source_lang, target_lang)
    )
    df['backtranslated'] = df['translated'].apply(
        lambda x: translate_text(x, target_lang, source_lang)
    )
    df['similarity_score'] = df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']), axis=1
    )
    
    return df