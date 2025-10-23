# TabDomainExtractor: Automatically infer the domain of tabular datasets using Large Language Models (LLMs)

## Overview
This repository implements an automated solution for determining the domains of tabular datasets using Large Language Models (LLM) via the OpenRouter API. The project leverages models such as gpt-4o and deepseek-chat to classify tabular data into thematic domains (e.g., medicine, finance) using zero-shot, one-shot, and few-shot learning approaches. It supports stratified splitting (StratifiedShuffleSplit) and K-fold cross-validation, with experimental scripts for different sizes of datasets. The method is evaluated on a collected benchmark of 258 OpenML datasets labeled across 14 domains, demonstrating strong performance in domain classification tasks. 

## Idea
Manually labeling the domain of numerous datasets is time-consuming and subjective. This project automates that process. The core idea is that the semantic meaning of column names and some lines in a table strongly indicates its overall domain. An LLM, with its extensive world knowledge, is perfectly suited to interpret these column names and provide a consistent, accurate domain classification.

## Features
 - **Comprehensive Benchmark:** Evaluated on a manually curated benchmark of 258 datasets from OpenML across 14 distinct domains.
 - **Automated Domain Classification**: Identifies thematic domains of tabular data using LLM.
 - **Supported Models**: gpt-4o and deepseek-chat via OpenRouter API.
 - **Classification Modes**: Zero-shot, one-shot, and few-shot learning.
 - **Flexible Partitioning**: Supports stratified splitting and K-fold cross-validation.
 - **Experimentation**: Includes scripts for processing subsets (5 and 50 rows) for testing.
   
## Repository structure

```
TabDomainExtractor/
├── baseline/                               # Baseline implementations
│   ├── d4_baseline.py                      # Baseline using D4 system approach
│   ├── embeddings_of_columns_baseline.py   # Baseline using column embeddings
│   └── metafeatures_baseline.py            # Baseline using traditional meta-features
├── benchmark/                              # Benchmarking utilities and results
│   ├── benchmark.json                      # Benchmark dataset definitions
│   └── collecting_benchmark.ipynb          # Notebook for collecting benchmark data
├── experiments/                            # Experimental runs with different configurations
│   ├── 50rows_deepseek.py                  # Experiment with 50 rows using DeepSeek
│   ├── 50rows_gpt4o.py                     # Experiment with 50 rows using GPT-4o
│   ├── 5rows_deepseek.py                   # Experiment with 5 rows using DeepSeek
│   └── 5rows_gpt4o.py                      # Experiment with 5 rows using GPT-4o
├── prompts/                                # LLM prompt templates
│   ├── few_shot_prompt.txt                 # Few-shot learning prompt template
│   └── zero_shot_prompt.txt                # Zero-shot prompt template
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```
git clone https://github.com/ITMO-NSS-team/TabDomain_Extractor.git
cd TabDomain_Extractor
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Configure the OpenRouter API key: create a .env file in the project root:

```
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

