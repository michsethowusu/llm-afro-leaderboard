Africa MT Benchmark
===================

A comprehensive benchmarking framework for evaluating machine translation models on African languages using back-translation and semantic similarity analysis.

Machine Translation Benchmarking NLLB Back-translation Semantic Similarity

Overview
--------

Africa MT Benchmark is a framework designed to evaluate the performance of machine translation models on African languages using a robust back-translation and similarity scoring methodology.

Evaluation Methodology
----------------------

### Back-Translation Evaluation Process

**Step 1: Forward Translation**

Source text (Language A) → Machine Translation → Translated text (Language B)

**Step 2: Back Translation**

Translated text (Language B) → Machine Translation → Back-translated text (Language A)

**Step 3: Similarity Calculation**

Compare original source text with back-translated text using semantic similarity metrics

**Step 4: Performance Scoring**

Calculate average similarity scores across the dataset to evaluate translation quality

### Semantic Similarity Measurement

The framework uses cosine similarity between sentence embeddings to measure translation quality:

**Formula:** similarity = cos(θ) = (A·B) / (||A||·||B||)

Where A and B are vector representations of the original and back-translated sentences.

**Implementation:** We use the all-MiniLM-L6-v2 model from SentenceTransformers to generate high-quality sentence embeddings.

### Why Back-Translation Evaluation?

This approach offers several advantages for evaluating MT systems for African languages:

*   **No parallel data required:** Works with monolingual text in the source language
*   **Semantic preservation:** Measures how well meaning is preserved through translation
*   **Model-agnostic:** Can evaluate any translation model regardless of architecture
*   **Quantifiable results:** Provides numerical scores for easy comparison
*   **Error analysis:** Allows examination of specific translation errors through comparison

Features
--------

*   Support for multiple MT models through a recipe system
*   Automatic language detection from folder names
*   Comprehensive language mapping for African languages
*   Semantic similarity scoring using sentence embeddings
*   Performance visualization and reporting
*   Extensible architecture for adding new models
*   Automatic skip of already processed files

Quick Start
-----------

### Installation

    git clone https://github.com/your-username/africa-mt-benchmark.git
    cd africa-mt-benchmark
    pip install -r requirements.txt

### Usage

1\. Prepare your data in the input directory with folder names following the pattern `source-target` (e.g., `eng-twi`)

2\. Place CSV files with a `text` column in these folders

3\. Run the benchmark:

    python main.py

### Example Input Structure

    input/
    ├── eng-twi/
    │   └── sentences.csv
    ├── eng-yor/
    │   └── sentences.csv
    └── fra-swc/
        └── sentences.csv

Supported Models
----------------

*   NLLB-200 Distilled 600M
*   NLLB-200 1.3B
*   NLLB-200 3.3B

Results Interpretation
----------------------

The benchmark generates comprehensive reports including:

*   **Performance comparison charts:** Visual comparison of models across language pairs
*   **Detailed CSV reports:** Similarity scores for each sentence and model
*   **Summary statistics:** Average, median, and distribution of similarity scores
*   **Model comparison visualizations:** Direct comparison of different models

**Interpreting Similarity Scores:**  
• 0.9-1.0: Excellent translation (meaning fully preserved)  
• 0.7-0.9: Good translation (minor meaning changes)  
• 0.5-0.7: Acceptable translation (some meaning loss)  
• 0.3-0.5: Poor translation (significant meaning loss)  
• 0.0-0.3: Very poor translation (complete meaning change)

Limitations and Considerations
------------------------------

While the back-translation evaluation method is powerful, it has some limitations:

*   May not capture all nuances of translation quality
*   Semantic similarity models may have their own biases
*   Does not evaluate grammatical correctness directly
*   Cultural nuances and idioms may not be fully captured

We recommend using this benchmark alongside human evaluation for comprehensive assessment.

**Note:** This project is designed to work well on Google Colab with GPU acceleration for faster model inference.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
---------------

*   Facebook Research for the NLLB models
*   Hugging Face for the Transformers library
*   SentenceTransformers for similarity scoring
*   Google Colab for providing computational resources

Citation
--------

If you use this benchmark in your research, please cite:

    @software{africa-mt-benchmark,
      title = {Africa MT Benchmark: A Framework for Evaluating Machine Translation Models on African Languages},
      author = {Your Name},
      year = {2023},
      url = {https://github.com/michsethowusu/africa-mt-benchmark}
    }
