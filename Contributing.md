Contributing to Africa MT Benchmark
===================================

We welcome contributions to expand the range of machine translation models supported by this benchmark!

How to Contribute a New Recipe
------------------------------

### Step 1: Fork the Repository

Start by forking the repository to your own GitHub account.

    git clone https://github.com/michsethowusu/africa-mt-benchmark.git
    cd africa-mt-benchmark

### Step 2: Create a New Recipe File

Create a new Python file in the `recipes` directory following the naming convention `model_name.py`.

Example: `recipes/your_model.py`

### Step 3: Implement the Recipe Structure

Your recipe should follow this basic structure:

    import pandas as pd
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    from sentence_transformers import SentenceTransformer, util
    import sys
    import os
    
    # Add utils to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from language_mapping import get_nllb_code
    
    # Initialize your model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "your/model/name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def translate_text(text, source_lang, target_lang):
        """Your translation function implementation"""
        # Implement translation logic
        pass
    
    def calculate_similarity(original, backtranslated):
        """Calculate similarity between original and back-translated text"""
        embeddings = similarity_model.encode([original, backtranslated])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    def process_dataframe(df, source_lang, target_lang):
        """Main processing function"""
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

### Step 4: Test Your Recipe

Create a test folder in the input directory and add a sample CSV file:

    input/
    └── eng-twi-test/
        └── test_sentences.csv

Run your recipe to ensure it works correctly:

    python main.py

### Step 5: Submit a Pull Request

Once your recipe is working correctly, submit a pull request to the main repository.

Recipe Requirements
-------------------

*   Must implement the `process_dataframe` function
*   Must preserve all original columns in the input CSV
*   Must add `translated`, `backtranslated`, and `similarity_score` columns
*   Should include appropriate error handling
*   Should support batch processing for efficiency with large datasets

Best Practices
--------------

### Model Initialization

Initialize models once at the module level, not per translation:

    # Good practice
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "your/model/name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Avoid initializing inside functions
    def translate_text(text, source_lang, target_lang):
        # Don't initialize here
        pass

### Error Handling

Include robust error handling for translation failures:

    def translate_text(text, source_lang, target_lang):
        try:
            # Translation logic
            return translation
        except Exception as e:
            print(f"Error translating text: {text}. Error: {str(e)}")
            return ""

### Memory Management

Be mindful of memory usage, especially with large models:

    # Use FP16 precision for memory efficiency
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16
    ).to(device)

**Note:** For very large models (3B+ parameters), consider implementing additional memory optimization techniques like gradient checkpointing or quantization.

Testing Guidelines
------------------

*   Test with multiple language pairs
*   Verify similarity scores are within expected ranges (0-1)
*   Ensure the recipe works with both small and large datasets
*   Test on both CPU and GPU environments if possible

Adding New Language Support
---------------------------

To add support for new languages, update the language mapping in `utils/language_mapping.py`:

    "your_lang_code": {
        "iso2": "xx",
        "iso3": "xxx",
        "name": "Language Name",
        "nllb_code": "xxx_Latn",  # or appropriate script
        "script": "Latn"  # or "Arab", "Ethi", etc.
    }

Questions?
----------

If you have questions about contributing, please open an issue on GitHub or contact the maintainers.

Thank you for contributing to the Africa MT Benchmark project!
