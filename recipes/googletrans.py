import pandas as pd
import time
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_iso2_code

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-MiniLM-L6-v2"
similarity_model = SentenceTransformer(similarity_model_name)

# Initialize translator
translator = Translator()

def translate_text(text, source_lang, target_lang):
    """Translate text using Google Translate API"""
    try:
        # Convert language codes to ISO 639-1 (2-letter) format for googletrans
        source_iso2 = get_iso2_code(source_lang)
        target_iso2 = get_iso2_code(target_lang)
        
        # Translate the text
        translation = translator.translate(text, src=source_iso2, dest=target_iso2)
        
        return translation.text
        
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

def process_dataframe(df, source_lang, target_lang):
    """Main processing function for the recipe"""
    print(f"Translating from {source_lang} to {target_lang} using Google Translate")
    
    # Make a copy to preserve the original DataFrame including the source column
    result_df = df.copy()
    
    # Add new columns
    translations = []
    for i, text in enumerate(result_df['text']):
        translation = translate_text(text, source_lang, target_lang)
        translations.append(translation)
        
        # Rate limiting: 1 request per second + add a bit of jitter
        time.sleep(1 + (i % 3) * 0.1)
    
    result_df['translated'] = translations
    
    backtranslations = []
    for i, translation in enumerate(result_df['translated']):
        if translation:  # Only backtranslate if we have a translation
            backtranslation = translate_text(translation, target_lang, source_lang)
            backtranslations.append(backtranslation)
            
            # Rate limiting: 1 request per second + add a bit of jitter
            time.sleep(1 + (i % 3) * 0.1)
        else:
            backtranslations.append("")
    
    result_df['backtranslated'] = backtranslations
    
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']), axis=1
    )
    
    return result_df