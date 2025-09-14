from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

# Global cache for models
_model_cache = {}

def get_backtranslation_model(model_name="facebook/nllb-200-3.3B"):
    """Get or load backtranslation model from cache"""
    if model_name not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading backtranslation model: {model_name} on {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        _model_cache[model_name] = {
            'tokenizer': tokenizer,
            'model': model,
            'device': device
        }
    
    return _model_cache[model_name]

def get_similarity_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """Get or load similarity model from cache"""
    if model_name not in _model_cache:
        print(f"Loading similarity model: {model_name}")
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
    
    return _model_cache[model_name]

def clear_model_cache():
    """Clear the model cache to free memory"""
    global _model_cache
    for model_name in list(_model_cache.keys()):
        if hasattr(_model_cache[model_name], 'cpu'):
            _model_cache[model_name].cpu()
        del _model_cache[model_name]
    _model_cache = {}
    torch.cuda.empty_cache()