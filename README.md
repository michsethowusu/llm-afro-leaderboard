# LLM Afro Leaderboard

LLM Afro Leaderboard is an open tool for evaluating how well Large Language Models (LLMs) handle translations to and from African languages—where data is often scarce and existing benchmarks fall short.

Our aim is simple: build a transparent, reproducible way to test zero-shot translation accuracy, so that researchers and developers can push forward inclusive, high-quality language technologies.

------

## Table of Contents

- [About the Project](#about-the-project)
- [Current Results & Limitations](#current-results--limitations)
- [Evaluated Models](#evaluated-models)
- [How to Contribute](#how-to-contribute)
- [Technical Requirements](#technical-requirements)
- [Evaluation Methodology](#evaluation-methodology)
- [License](#license)
- [Acknowledgements](#acknowledgements)

------

## About the Project

African languages remain among the least represented in today’s AI landscape. Training data is limited, and benchmarks are even rarer. LLM Afro Leaderboard sets out to change that by providing a practical, standardized way to measure how well different LLMs can translate African languages.

By building shared benchmarks, we can highlight both progress and gaps—ultimately contributing to fairer, more accessible language technology across the continent.

------

## Current Results & Limitations

The initial results in this repo were generated on a small, exploratory dataset:

- ~15 paragraphs per language (from news, literature, and academic texts)
- ~50 languages tested
- ~14 different models evaluated


**Limitations to note**:

- Results are anecdotal, not definitive benchmarks.
- Many proprietary models weren’t tested.
- Scaling across multiple languages/models is resource-intensive.

Still, this early test provides a valuable starting point for deeper work.

------

## Supported Languages

The following languages are evaluated in this project:

[Afrikaans](https://www.ethnologue.com/language/afr/), [Akan](https://www.ethnologue.com/language/aka/), [Amharic](https://www.ethnologue.com/language/amh/), [Arabic](https://www.ethnologue.com/language/ara/), [Bambara](https://www.ethnologue.com/language/bam/), [Bemba](https://www.ethnologue.com/language/bem/), [Berber](https://www.google.com/search?q=https://www.ethnologue.com/language/ber/), [Central Atlas Tamazight](https://www.ethnologue.com/language/tzm/), [Chichewa](https://www.ethnologue.com/language/nya/), [Congo Swahili](https://www.ethnologue.com/language/swc/), [Dinka](https://www.ethnologue.com/language/dik/), [Dyula](https://www.ethnologue.com/language/dyu/), [Ewe](https://www.ethnologue.com/language/ewe/), [Fon](https://www.ethnologue.com/language/fon/), [Fulah](https://www.ethnologue.com/language/ful/), [Ganda](https://www.ethnologue.com/language/lug/), [Harari](https://www.ethnologue.com/language/har/), [Hausa](https://www.ethnologue.com/language/hau/), [Herero](https://www.ethnologue.com/language/her/), [Igbo](https://www.ethnologue.com/language/ibo/), [Kabiyè](https://www.ethnologue.com/language/kbp/), [Kabuverdianu](https://www.ethnologue.com/language/kea/), [Kabyle](https://www.ethnologue.com/language/kab/), [Kamba](https://www.ethnologue.com/language/kam/), [Kikuyu](https://www.ethnologue.com/language/kik/), [Kimbundu](https://www.ethnologue.com/language/kmb/), [Kinyarwanda](https://www.ethnologue.com/language/kin/), [Kongo](https://www.ethnologue.com/language/kon/), [Kpelle](https://www.ethnologue.com/language/kpe/), [Kuanyama](https://www.ethnologue.com/language/kua/), [Lingala](https://www.ethnologue.com/language/lin/), [Luba-Katanga](https://www.ethnologue.com/language/lub/), [Lushai](https://www.ethnologue.com/language/lus/), [Mende](https://www.ethnologue.com/language/men/), [Mossi](https://www.ethnologue.com/language/mos/), [Ndonga](https://www.ethnologue.com/language/ndo/), [Northern Sotho](https://www.ethnologue.com/language/nso/), [Nuer](https://www.ethnologue.com/language/nus/), [Oromo](https://www.ethnologue.com/language/orm/), [Rundi](https://www.ethnologue.com/language/run/), [Sango](https://www.ethnologue.com/language/sag/), [Shona](https://www.ethnologue.com/language/sna/), [Sidamo](https://www.ethnologue.com/language/sid/), [Somali](https://www.ethnologue.com/language/som/), [Southern Sotho](https://www.ethnologue.com/language/sot/), [Susu](https://www.ethnologue.com/language/sus/), [Swahili](https://www.ethnologue.com/language/swa/), [Swati](https://www.ethnologue.com/language/ssw/), [Tachelhit](https://www.ethnologue.com/language/shi/), [Tamasheq](https://www.ethnologue.com/language/taq/), [Temne](https://www.ethnologue.com/language/tem/), [Tigre](https://www.ethnologue.com/language/tig/), [Tigrinya](https://www.ethnologue.com/language/tir/), [Tonga](https://www.ethnologue.com/language/toi/), [Tsonga](https://www.ethnologue.com/language/tso/), [Tswana](https://www.ethnologue.com/language/tsn/), [Tumbuka](https://www.ethnologue.com/language/tum/), [Umbundu](https://www.ethnologue.com/language/umb/), [Vai](https://www.ethnologue.com/language/vai/), [Venda](https://www.ethnologue.com/language/ven/), [West Central Oromo](https://www.ethnologue.com/language/gaz/), [Wolof](https://www.ethnologue.com/language/wol/), [Xhosa](https://www.ethnologue.com/language/xho/), [Yoruba](https://www.ethnologue.com/language/yor/), [Zulu](https://www.ethnologue.com/language/zul/)

------

## Evaluated Models

The following models were evaluated in the initial round of testing:

- [abacusai/dracarys-llama-3.1-70b-instruct](https://huggingface.co/abacusai/dracarys-llama-3.1-70b-instruct)
- [ai21labs/jamba-1.5-large-instruct](https://huggingface.co/ai21labs/jamba-1.5-large-instruct)
- [deepseek-ai/deepseek-r1-0528](https://huggingface.co/deepseek-ai/deepseek-r1-0528)
- [deepseek-ai/deepseek-v3.1](https://huggingface.co/deepseek-ai/deepseek-v3.1)
- [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it)
- [gotocompany/gemma-2-9b-cpt-sahabatai-instruct](https://huggingface.co/gotocompany/gemma-2-9b-cpt-sahabatai-instruct)
- [meta/llama-3.3-70b-instruct](https://huggingface.co/meta-llama/llama-3.3-70b-instruct)
- [moonshotai/kimi-k2-instruct](https://huggingface.co/moonshotai/kimi-k2-instruct)
- [nvidia/llama-3.1-nemotron-70b-instruct](https://huggingface.co/nvidia/llama-3.1-nemotron-70b-instruct)
- [nvidia/llama-3.1-nemotron-ultra-253b-v1](https://huggingface.co/nvidia/llama-3.1-nemotron-ultra-253b-v1)
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [qwen/qwen3-coder-480b-a35b-instruct](https://huggingface.co/qwen/qwen3-coder-480b-a35b-instruct)
- [qwen/qwq-32b](https://huggingface.co/qwen/qwq-32b)
- [writer/palmyra-creative-122b](https://huggingface.co/writer/palmyra-creative-122b)

------

## How to Contribute

We’d love your help in expanding LLM Afro Leaderboard. Here’s how you can get involved:

- **Focused testing**: Run evaluations on larger or higher-quality datasets for one or a few languages. This is far easier (and often more insightful) than testing dozens at once.
- **Broader sweeps**: If you have the compute, replicate the multi-language/multi-model setup and share your findings.
- **Missing models**: Evaluate proprietary or under-tested models by creating new “recipes.”

Contributions of all sizes are welcome—from dataset prep, to code improvements, to sharing evaluation results.

------

## Technical Requirements

LLM Afro Leaderboard relies on heavy models like Facebook’s NLLB-3B and MPNet, so GPU access is highly recommended.

You can get started in two ways:

- Use the provided Google Colab notebook for quick replication.
- Clone the repo and run locally if you have a GPU setup.

------

## Translation Accuracy Scoring

We use a backtranslation + similarity check approach:

1. Translate English → African language with the LLM.
2. Backtranslate the output into English using NLLB-3B, one of the strongest MT systems available.
3. Measure semantic similarity between the original and backtranslated English using MPNet embeddings.

This yields a numeric score that captures how faithful the LLM’s translation was.

------

## License

This project is released under the MIT License.

------

## Acknowledgements

Huge thanks to the open-source community and researchers powering this project, including:

- [Facebook AI’s No Language Left Behind (NLLB) team](https://huggingface.co/facebook/nllb-200-3.3B)
- [Sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

Without their hard work, this project wouldn’t exist.
