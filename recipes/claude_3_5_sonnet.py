import pandas as pd
import anthropic
from sentence_transformers import SentenceTransformer, util
import sys
import os
import getpass

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Initialize models
print("Loading similarity model...")
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Get API key
api_key = getpass.getpass("Enter your Anthropic API key: ")
client = anthropic.Anthropic(api_key=api_key)

def translate_text(text, source_lang, target_lang):
    """Translate text using Claude 3.5 Sonnet v2"""
    try:
        prompt = f"Translate the following text from {source_lang} to {target_lang}. Return only the translation, no explanations or additional text:\n\n{text}"
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Rate limiting: wait 1 second after each API call
        time.sleep(1)
        
        return message.content[0].text.strip()
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
    print(f"Translating from {source_lang} to {target_lang} using Claude 3.5 Sonnet v2")
    
    # Make a copy to preserve the original DataFrame including the source column
    result_df = df.copy()
    
    # Add new columns
    result_df['translated'] = result_df['text'].apply(
        lambda x: translate_text(x, source_lang, target_lang)
    )
    result_df['backtranslated'] = result_df['translated'].apply(
        lambda x: translate_text(x, target_lang, source_lang)
    )
    result_df['similarity_score'] = result_df.apply(
        lambda row: calculate_similarity(row['text'], row['backtranslated']), axis=1
    )
    
    return result_df
