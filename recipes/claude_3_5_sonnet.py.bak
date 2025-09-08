import pandas as pd
import time
import anthropic
from sentence_transformers import SentenceTransformer, util
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_language_name

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-MiniLM-L6-v2"
similarity_model = SentenceTransformer(similarity_model_name)

def get_api_key():
    """Prompt user for Anthropic API key"""
    api_key = input("Please enter your Anthropic API key: ").strip()
    return api_key

def initialize_claude_client():
    """Initialize the Anthropic client with API key"""
    api_key = get_api_key()
    client = anthropic.Anthropic(api_key=api_key)
    return client

def translate_text(text, source_lang, target_lang, client):
    """Translate text using Claude 3.5 Sonnet with careful prompting"""
    # Get language names for better prompting
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)
    
    # Create a precise prompt to get only the translation
    prompt = f"""Please translate the following text from {source_lang_name} to {target_lang_name}. 
Return ONLY the translation without any additional text, explanations, or formatting.

Text to translate: "{text}"

Translation:"""
    
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            temperature=0.1,  # Low temperature for more deterministic output
            system="You are a professional translator. Your task is to provide accurate translations without any additional commentary, explanations, or formatting. Only return the translated text.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract the translation from the response
        translation = message.content[0].text.strip()
        
        # Clean up any potential extra text
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]
        
        return translation
        
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
    print(f"Translating from {source_lang} to {target_lang} using Claude 3.5 Sonnet")
    
    # Initialize Claude client
    client = initialize_claude_client()
    
    # Make a copy to preserve the original DataFrame including the source column
    result_df = df.copy()
    
    # Add new columns
    translations = []
    for text in result_df['text']:
        translation = translate_text(text, source_lang, target_lang, client)
        translations.append(translation)
        time.sleep(1)  # Rate limiting: 1 request per second
    
    result_df['translated'] = translations
    
    backtranslations = []
    for translation in result_df['translated']:
        if translation:  # Only backtranslate if we have a translation
            backtranslation = translate_text(translation, target_lang, source_lang, client)
            backtranslations.append(backtranslation)
            time.sleep(1)  # Rate limiting: 1 request per second
        else:
            backtranslations.append("")
    
    result_df['backtranslated'] = backtranslations
    
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']), axis=1
    )
    
    return result_df
