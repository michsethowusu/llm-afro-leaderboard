import pandas as pd
import time
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import ctranslate2
from transformers import AutoTokenizer
import gc
import torch

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

# Initialize global variables
backtranslation_tokenizer = None
translator = None
similarity_model = None

# -------------------------------
# Load backtranslation model (float16, GPU)
# -------------------------------
def load_backtranslation_models():
    global backtranslation_tokenizer, translator
    if backtranslation_tokenizer is None or translator is None:
        backtranslation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
        print("Loaded NLLB tokenizer")
        
        # Full precision (float16) on GPU
        model_path = "nllb-200-3.3B-float16-ct2"
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        translator = ctranslate2.Translator(model_path, device=device, compute_type="float16")
        print(f"Loaded NLLB model on {device} (float16)")

# -------------------------------
# Extract text inside brackets
# -------------------------------
def extract_text_from_brackets(text):
    match = re.search(r'\[(.*?)\]', text, flags=re.S)
    if match:
        return match.group(1).strip()
    return ""

# -------------------------------
# NVIDIA Build API translation
# -------------------------------
def translate_text_with_nvidia(text, source_lang, target_lang, max_retries=5):
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)
    prompt = f"Translate the following {source_lang_name} text into {target_lang_name} and return ONLY the translation inside square brackets:\n\n{text}"

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="abacusai/dracarys-llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.95,
                max_tokens=2024,
                stream=False
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""

# -------------------------------
# Forward translation only
# -------------------------------
def forward_translation_only(df, source_lang, target_lang):
    print("Forward translation using NVIDIA Build API")
    result_df = df.copy()
    translations = []

    delay = 60 / 38  # 38 requests per minute

    for i, text in enumerate(result_df['text']):
        print(f"Translating {i+1}/{len(result_df)}: {text[:50]}...")
        translation = translate_text_with_nvidia(text, source_lang, target_lang)
        translations.append(translation or "")
        if i < len(result_df)-1:
            time.sleep(delay)

    result_df['translated'] = translations
    return result_df

# -------------------------------
# Backtranslation (one by one)
# -------------------------------
def backtranslate_with_nllb(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    load_backtranslation_models()
    nllb_source = get_nllb_code(target_lang)
    nllb_target = get_nllb_code(source_lang)
    src_token = backtranslation_tokenizer.convert_tokens_to_ids([nllb_source])[0]
    tgt_token = backtranslation_tokenizer.convert_tokens_to_ids([nllb_target])[0]
    eos_token = backtranslation_tokenizer.convert_tokens_to_ids(["</s>"])[0]

    backtranslations = []
    for idx, text in enumerate(texts):
        if not text:
            backtranslations.append("")
            continue
        try:
            tokens = [src_token] + backtranslation_tokenizer.encode(text, add_special_tokens=False) + [eos_token]
            token_ids = backtranslation_tokenizer.convert_ids_to_tokens(tokens)
            result = translator.translate_batch([token_ids], target_prefix=[[tgt_token]], max_batch_size=1)
            if result and result[0].hypotheses:
                hyp = result[0].hypotheses[0]
                decoded = backtranslation_tokenizer.decode(backtranslation_tokenizer.convert_tokens_to_ids(hyp[1:]), skip_special_tokens=True)
                backtranslations.append(decoded)
                print(f"Backtranslated {idx+1}/{len(texts)}: {text[:50]}... → {decoded[:50]}...")
            else:
                backtranslations.append("")
                print(f"Backtranslation failed for text '{text}'")
        except Exception as e:
            print(f"Error backtranslating text '{text}': {str(e)}")
            backtranslations.append("")
    return backtranslations

# -------------------------------
# Similarity (CPU only)
# -------------------------------
def calculate_similarity_cpu(df: pd.DataFrame) -> pd.Series:
    global similarity_model
    if similarity_model is None:
        from sentence_transformers import SentenceTransformer, util
        similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    originals = df['text'].tolist()
    backs = df['backtranslated'].tolist()
    embeddings_orig = similarity_model.encode(originals, batch_size=32, convert_to_tensor=True)
    embeddings_back = similarity_model.encode(backs, batch_size=32, convert_to_tensor=True)
    from sentence_transformers import util
    scores = util.pytorch_cos_sim(embeddings_orig, embeddings_back).diagonal().cpu().numpy()
    return pd.Series(scores, name="similarity_score")

# -------------------------------
# Full pipeline
# -------------------------------
def process_dataframe(df, source_lang, target_lang):
    # Forward translation
    df = forward_translation_only(df, source_lang, target_lang)

    # Backtranslation
    print("Starting backtranslation...")
    df['backtranslated'] = backtranslate_with_nllb(df['translated'].tolist(), source_lang, target_lang)

    # Free GPU before similarity
    print("Freeing GPU memory...")
    global translator, backtranslation_tokenizer
    del translator, backtranslation_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Similarity calculation on CPU
    print("Calculating similarity on CPU...")
    df['similarity_score'] = calculate_similarity_cpu(df)

    return df

