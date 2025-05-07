"""
Functions for parsing model references and mapping model names to IDs.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def parse_models_csv(csv_path):
    """Parse the models reference CSV file."""
    try:
        models_df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(models_df)} models from CSV")
        return models_df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return None

def is_huggingface_model(website, model_name):
    """Check if the model is available on Hugging Face."""
    huggingface_patterns = [
        "huggingface.co",
        "hf.co",
    ]
    
    known_hf_models = [
        "BERT", "ALBERT", "DeBERTa", "ViT", "Whisper", "XLSR", "wav2vec"
    ]
    
    if website:
        for pattern in huggingface_patterns:
            if pattern in str(website).lower():
                return True
    
    if model_name:
        for model in known_hf_models:
            if model.lower() in str(model_name).lower():
                return True
    
    return False

def get_model_id(model_name, model_type):
    """Map model names to Hugging Face model IDs."""
    model_mappings = {
        "ALBERT": "albert-base-v2",
        "DeBERTa": "microsoft/deberta-base",
        "XLSR": "facebook/wav2vec2-large-xlsr-53",
        "Sentence-BERT": "sentence-transformers/all-MiniLM-L6-v2",
        "WhisperX": "openai/whisper-small",
        "SimCSE": "princeton-nlp/sup-simcse-bert-base-uncased",
    }
    
    if model_name in model_mappings:
        return model_mappings[model_name]
    
    if "bert" in model_name.lower() and "text" in model_type.lower():
        return "bert-base-uncased"
    elif "whisper" in model_name.lower():
        return "openai/whisper-base"
    
    logger.warning(f"No direct Hugging Face mapping for {model_name}, using generic ID")
    return None
