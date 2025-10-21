import json
import os
import pandas as pd
import csv
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
import chardet
import warnings
import random
from openai import OpenAI
from collections import Counter

warnings.filterwarnings('ignore')

MODEL_NAME = "deepseek-chat"
OUTPUT_DIR = "all_datasets"
RESULTS_DIR = f"classification_results_{MODEL_NAME}"
DOMAIN_MAPPING_FILE = "clustering_domain_mapping.json"
API_KEY = os.getenv("API_KEY")
TIMEOUT = 7200
TEST_FRACTION = 0.2
MAX_ROWS = 50
MAX_TOKENS = 163840
CHARS_PER_TOKEN = 2.0
CALIBRATED_COEFFS = [2.0]
MAX_EXAMPLE_TOKENS = 50000
FEW_SHOT_SIZE = 5
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClassificationResult(BaseModel):
    primary_domain: str = Field(..., description="The most appropriate domain for the dataset")
    alternative_domains: List[str] = Field(
        default_factory=list, 
        description="List of alternative relevant domains (up to 2)",
        max_items=2
    )

class BatchClassificationResult(BaseModel):
    results: List[ClassificationResult] = Field(..., description="List of classification results for multiple datasets")

def detect_encoding(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(50000)
        result = chardet.detect(rawdata)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
    except Exception as e:
        logger.warning(f"Encoding detection error: {e}, using utf-8")
        return 'utf-8'

def read_dataset_with_memory_limit(file_path: str, dataset_id: str = None) -> pd.DataFrame:
    try:
        if dataset_id:
            full_path = os.path.join(OUTPUT_DIR, dataset_id, "full_dataset.csv")
            if not os.path.exists(full_path):
                logger.warning(f"Dataset {dataset_id}: full_dataset.csv not found at {full_path}")
                return pd.DataFrame()
        else:
            full_path = file_path

        encoding = detect_encoding(full_path)
        
        possible_delimiters = [',', ';', '\t', '|']
        df = None
        error_details = []

        with open(full_path, 'r', encoding=encoding) as f:
            sample = f.read(4096)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                separator = dialect.delimiter
                logger.info(f"Dataset {dataset_id or full_path}: Detected delimiter '{separator}'")
            except csv.Error:
                logger.warning(f"Dataset {dataset_id or full_path}: Could not detect delimiter, trying possible delimiters")
                separator = None

        for sep in ([separator] if separator else []) + possible_delimiters:
            try:
                with open(full_path, 'r', encoding=encoding) as f:
                    header = f.readline().strip()
                    num_columns = len(header.split(sep))
                    logger.info(f"Dataset {dataset_id or full_path}: Header contains {num_columns} columns: {header[:100]}...")

                df = pd.read_csv(
                    full_path,
                    encoding=encoding,
                    nrows=MAX_ROWS,
                    sep=sep,
                    quoting=csv.QUOTE_ALL,
                    on_bad_lines='skip',
                    engine='python'
                )
                logger.info(f"Dataset {dataset_id or full_path}: Successfully read {len(df)} rows and {len(df.columns)} columns with delimiter '{sep}'")
                break
            except Exception as e:
                error_details.append(f"Delimiter '{sep}': {str(e)}")
                logger.warning(f"Dataset {dataset_id or full_path}: Error with delimiter '{sep}': {e}")
                continue

        if df is None or df.empty:
            logger.error(f"Dataset {dataset_id or full_path}: Could not read CSV. Errors: {error_details}")
            with open(full_path, 'r', encoding=encoding) as f:
                lines = [f.readline().strip() for _ in range(5)]
                logger.error(f"Dataset {dataset_id or full_path}: First 5 lines: {lines}")
            return pd.DataFrame()

        if len(df) > MAX_ROWS:
            df = df.sample(n=MAX_ROWS, random_state=42)
        else:
            df = df.sample(n=len(df), random_state=42)
        
        logger.info(f"Dataset {dataset_id or full_path}: Selected {len(df)} rows and {len(df.columns)} columns")
        return df
            
    except MemoryError as e:
        logger.error(f"Memory error reading {dataset_id or full_path}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error reading CSV {dataset_id or full_path}: {e}")
        return pd.DataFrame()

def prepare_for_llm(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> str:
    df_subset = df.head(max_rows).copy()
    df_subset = df_subset.fillna('missing')
    
    columns = df_subset.columns.tolist()
    data = []
    
    for _, row in df_subset.iterrows():
        formatted_row = []
        for value in row:
            if isinstance(value, str):
                val = value.replace('"', '\\"')
                formatted_value = f'"{val}"'
            else:
                formatted_value = str(value)
            formatted_row.append(formatted_value)
        data.append(f"[{', '.join(formatted_row)}]")
    
    columns_str = "[" + ", ".join([f'"{col}"' for col in columns]) + "]"
    data_str = ",\n".join(data)
    
    return f"columns = {columns_str}\ndata = [\n{data_str}\n]"

def estimate_tokens(text: str) -> int:
    current_coeff = sum(CALIBRATED_COEFFS) / len(CALIBRATED_COEFFS)
    estimated = int(len(text) / current_coeff)
    logger.info(f"Estimated tokens: {estimated} (coeff {current_coeff:.2f}, text len {len(text)})")
    return estimated

class DeepSeekClassifier:
    def __init__(self):
        self.domains = self.load_domains()
        if not API_KEY:
            logger.error("API key not found in environment variable API_KEY")
            raise ValueError("API key not set")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "Dataset Classifier"
            }
        )
        self.parser = PydanticOutputParser(pydantic_object=BatchClassificationResult)
        logger.info(f"Initialized with {len(self.domains)} domains")

    def load_domains(self) -> List[str]:
        if not os.path.exists(DOMAIN_MAPPING_FILE):
            logger.error(f"Domain file not found: {os.path.abspath(DOMAIN_MAPPING_FILE)}")
            return []

        try:
            with open(DOMAIN_MAPPING_FILE, "r", encoding="utf-8") as f:
                domain_data = json.load(f)
                
                if isinstance(domain_data, dict):
                    domains = list(domain_data.keys())
                    logger.info(f"Loaded {len(domains)} domains")
                    return domains
                else:
                    logger.error(f"Unknown domain file format: {type(domain_data)}")
                    return []
                
        except Exception as e:
            logger.error(f"Error loading domains: {e}", exc_info=True)
            return []

    def create_examples_text(self, examples: List[Tuple[str, dict]]) -> str:
        examples_text = ""
        for i, (data_sample, classification) in enumerate(examples, 1):
            examples_text += (
                f"\nExample {i}:\nDataset sample:\n{data_sample}\n"
                f"Classification: {json.dumps(classification)}\n"
            )
        return examples_text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=180))
    def classify_batch(self, data_samples: List[str], dataset_tokens: List[int], mode: str = "zero-shot", examples: List[Tuple[str, dict]] = None) -> List[ClassificationResult]:
        if not data_samples or not self.domains:
            logger.warning("No data samples or no domains for classification")
            return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in data_samples]
        
        format_instructions = self.parser.get_format_instructions()
        if mode == "zero-shot":
            system_content = (
                "You are a dataset domain classification expert. "
                "Analyze the dataset samples provided as DataFrame-like structures "
                "and determine their primary domains with optional alternatives. "
                "Choose ONLY from the following domains: " + ", ".join(self.domains) + "\n\n" +
                format_instructions + "\n\n" +
                "Rules:\n"
                "1. For each test dataset, primary_domain: SINGLE most appropriate domain (MUST be from list)\n"
                "2. alternative_domains: 0-2 other relevant domains (only if applicable)\n"
                "3. Use EXACT domain names\n"
                "4. Output a JSON object with 'results' array containing one object per test dataset in order!\n"
                "5. No additional text/comments!"
            )
        else:
            examples_text = self.create_examples_text(examples) if examples else ""
            system_content = (
                "You are a dataset domain classification expert. "
                "Analyze dataset samples and determine their primary domains with optional alternatives. "
                "Choose ONLY from these domains: " + ", ".join(self.domains) + "\n\n" +
                format_instructions + "\n\n" +
                "Rules:\n"
                "1. For each test dataset, primary_domain: SINGLE most appropriate domain (MUST be from list)\n"
                "2. alternative_domains: 0-2 other relevant domains (only if applicable)\n"
                "3. Use EXACT domain names\n"
                "4. Output a JSON object with 'results' array containing one object per test dataset in order!\n"
                "5. No additional text/comments!\n\n" +
                "Examples:" + examples_text
            )
    
        base_prompt_tokens = estimate_tokens(system_content)
        logger.info(f"Base prompt tokens for {mode}: {base_prompt_tokens}")
        
        total_tokens = sum(dataset_tokens) + base_prompt_tokens
        if total_tokens > MAX_TOKENS:
            logger.error(f"Total tokens ({total_tokens}) exceed MAX_TOKENS ({MAX_TOKENS}) for {mode}. Cannot process all datasets.")
            return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in data_samples]
        
        results = self._process_batch(data_samples, mode, system_content)
        return results

    def _process_batch(self, batch_samples: List[str], mode: str, system_content: str) -> List[ClassificationResult]:
        try:
            test_datasets_text = ""
            for i, sample in enumerate(batch_samples, 1):
                test_datasets_text += f"\nTest Dataset {i}:\n{sample}\n"
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Classify these {'new ' if mode in ['one-shot', 'few-shot'] else ''}datasets:\n{test_datasets_text}"}
            ]
        
            prompt_text = "".join([msg["content"] for msg in messages])
            estimated_prompt_tokens = estimate_tokens(prompt_text)
            logger.info(f"Estimated prompt tokens for batch (char-based): {estimated_prompt_tokens}")
        
            response = self.client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=300 * len(batch_samples),
                response_format={"type": "json_object"},
                top_p=0.95
            )
        
            if not hasattr(response, 'choices') or not response.choices:
                logger.error(f"Empty or invalid API response: {response}")
                return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in batch_samples]
        
            actual_prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            prompt_chars = len(prompt_text)
            
            if actual_prompt_tokens > 0:
                chars_per_token = prompt_chars / actual_prompt_tokens
                CALIBRATED_COEFFS.append(chars_per_token)
                if len(CALIBRATED_COEFFS) > 10:
                    CALIBRATED_COEFFS.pop(0)
                logger.info(f"Calibrated coeff: {chars_per_token:.2f} (new avg {sum(CALIBRATED_COEFFS)/len(CALIBRATED_COEFFS):.2f})")
                logger.info(f"Token usage: prompt={actual_prompt_tokens}, completion={completion_tokens}, total={total_tokens}, chars={prompt_chars}")
    
            content = response.choices[0].message.content
            logger.info(f"Raw API response content for {mode}: {content[:1000]}... (total length: {len(content)})")
            
            try:
                parsed_content = json.loads(content)
                
                if isinstance(parsed_content, list):
                    batch_result = BatchClassificationResult(results=[ClassificationResult(**item) for item in parsed_content])
                    logger.info(f"Parsed direct list response with {len(batch_result.results)} classifications for {mode}")
                elif isinstance(parsed_content, dict) and 'results' in parsed_content:
                    batch_result = self.parser.parse(content)
                    logger.info(f"Parsed standard response with {len(batch_result.results)} classifications for {mode}")
                else:
                    logger.error(f"Unexpected response structure for {mode}: {content[:1000]}...")
                    return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in batch_samples]
                    
                logger.debug(f"Batch classification result ({mode}): {batch_result}")
                return batch_result.results
                
            except ValidationError as e:
                logger.error(f"Validation error in batch for {mode}: {e}\nResponse content: {content[:1000]}...")
                return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in batch_samples]
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {mode}: {e}\nResponse content: {content[:1000]}...")
                try:
                    partial_json = content[:e.pos]
                    json_start = partial_json.rfind('{')
                    json_end = partial_json.rfind('}') + 1
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        parsed_partial = json.loads(partial_json[json_start:json_end])
                        if isinstance(parsed_partial, list):
                            partial_result = BatchClassificationResult(results=[ClassificationResult(**item) for item in parsed_partial])
                        else:
                            partial_result = self.parser.parse(partial_json[json_start:json_end])
                        logger.info(f"Extracted partial result with {len(partial_result.results)} classifications for {mode}")
                        return partial_result.results + [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in range(len(batch_samples) - len(partial_result.results))]
                    else:
                        logger.warning(f"Could not extract valid partial JSON for {mode}")
                        return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in batch_samples]
                except Exception as pe:
                    logger.error(f"Failed to parse partial JSON for {mode}: {pe}")
                    return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in batch_samples]
                        
        except Exception as e:
            logger.error(f"Batch classification error ({mode}): {e}", exc_info=True)
            return [ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in batch_samples]

def load_all_datasets() -> Dict[str, Dict]:
    datasets = {}
    
    if not os.path.exists(OUTPUT_DIR):
        logger.error(f"Dataset directory not found: {os.path.abspath(OUTPUT_DIR)}")
        return datasets

    dataset_ids = [d for d in os.listdir(OUTPUT_DIR) 
                   if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    if not dataset_ids:
        logger.error(f"No dataset directories found in {os.path.abspath(OUTPUT_DIR)}")
        return datasets
    
    logger.info(f"Found {len(dataset_ids)} dataset directories")
    
    valid_datasets = []
    for d in dataset_ids:
        meta_path = os.path.join(OUTPUT_DIR, d, "metadata.json")
        data_path = os.path.join(OUTPUT_DIR, d, "full_dataset.csv")
        if not os.path.exists(meta_path):
            logger.warning(f"Skipping {d}: metadata.json not found at {meta_path}")
            continue
        if not os.path.exists(data_path):
            logger.warning(f"Skipping {d}: full_dataset.csv not found at {data_path}")
            continue
        valid_datasets.append(d)
    
    logger.info(f"Found {len(valid_datasets)} valid datasets with required files")
    
    token_counts = []
    for dataset_id in tqdm(valid_datasets, desc="Loading datasets"):
        try:
            meta_path = os.path.join(OUTPUT_DIR, dataset_id, "metadata.json")
            data_path = os.path.join(OUTPUT_DIR, dataset_id, "full_dataset.csv")

            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            if "domain" not in metadata:
                logger.warning(f"Skipping {dataset_id}: 'domain' field missing in metadata")
                continue

            df = read_dataset_with_memory_limit(data_path, dataset_id=dataset_id)
            if df.empty:
                logger.warning(f"Empty DataFrame for {dataset_id}, skipping")
                continue

            data_sample = prepare_for_llm(df)
            token_count = estimate_tokens(data_sample)
            logger.info(f"Dataset {dataset_id}: {len(df)} rows, {len(df.columns)} columns, {token_count} tokens, file size: {os.path.getsize(data_path) / (1024 * 1024):.2f} MB")

            datasets[dataset_id] = {
                'metadata': metadata,
                'dataframe': df,
                'domain': metadata["domain"].lower(),
                'data_sample': data_sample,
                'token_count': token_count
            }
            token_counts.append(token_count)
                
        except Exception as e:
            logger.error(f"Error loading {dataset_id}: {e}")
            continue
    
    if not datasets:
        logger.error(f"No valid datasets loaded from {os.path.abspath(OUTPUT_DIR)}")
    else:
        avg_tokens = np.mean(token_counts) if token_counts else 0
        std_tokens = np.std(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        logger.info(f"Token statistics: Average={avg_tokens:.2f} tokens, Std={std_tokens:.2f}, Min={min_tokens}, Max={max_tokens}, Total datasets={len(token_counts)}")
        
        stats_file = os.path.join(RESULTS_DIR, "token_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'average_tokens_per_dataset': avg_tokens,
                'std_tokens': std_tokens,
                'min_tokens': min_tokens,
                'max_tokens': max_tokens,
                'total_datasets': len(token_counts)
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Token statistics saved to {stats_file}")
    
    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets

def split_datasets_with_token_limit(datasets: Dict[str, Dict], test_fraction: float = TEST_FRACTION, base_prompt_tokens: int = 500, random_state: int = 42) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    dataset_ids = list(datasets.keys())
    labels = [datasets[ds_id]['domain'] for ds_id in dataset_ids]
    
    original_distribution = Counter(labels)
    total_datasets = len(labels)
    logger.info(f"Original dataset domain distribution: {original_distribution}")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    
    for train_idx, test_idx in sss.split(dataset_ids, labels):
        train_ids = [dataset_ids[i] for i in train_idx]
        test_ids = [dataset_ids[i] for i in test_idx]
    
    train_datasets = {ds_id: datasets[ds_id] for ds_id in train_ids}
    test_datasets = {ds_id: datasets[ds_id] for ds_id in test_ids}
    
    test_items = list(test_datasets.items())
    test_tokens = sum(item[1]['token_count'] for item in test_items)
    logger.debug(f"Initial test set: {len(test_items)} datasets, {test_tokens} tokens")
    
    domain_to_datasets = {}
    for ds_id, data in test_items:
        domain = data['domain']
        if domain not in domain_to_datasets:
            domain_to_datasets[domain] = []
        domain_to_datasets[domain].append((ds_id, data))
    
    min_datasets_per_domain = {domain: max(1, int(round((count / total_datasets) * test_fraction * total_datasets))) for domain, count in original_distribution.items()}
    logger.debug(f"Minimum datasets per domain: {min_datasets_per_domain}")
    
    domain_counts = Counter(data['domain'] for _, data in test_items)
    logger.debug(f"Initial test set domain distribution: {domain_counts}")
    
    while test_tokens + base_prompt_tokens > MAX_TOKENS * 0.7 and test_items:
        domain_counts = Counter(data['domain'] for _, data in test_items)
        logger.debug(f"Current test set domain distribution: {domain_counts}")
        logger.debug(f"Current test set datasets: {[item[0] for item in test_items]}")
        logger.debug(f"Current domain_to_datasets: { {domain: [ds[0] for ds in datasets] for domain, datasets in domain_to_datasets.items()} }")
        
        max_excess_domain = None
        max_excess = -1
        for domain in domain_counts:
            excess = domain_counts[domain] - min_datasets_per_domain.get(domain, 1)
            if excess > max_excess and domain_counts[domain] > 1 and domain_to_datasets.get(domain, []):
                max_excess = excess
                max_excess_domain = domain
        
        if max_excess_domain and domain_to_datasets.get(max_excess_domain, []):
            domain_datasets = sorted(domain_to_datasets[max_excess_domain], key=lambda x: x[1]['token_count'], reverse=True)
            if domain_datasets:
                ds_id, data = domain_datasets[0]
                test_items = [(id_, d) for id_, d in test_items if id_ != ds_id]
                domain_to_datasets[max_excess_domain].pop(0)
                if not domain_to_datasets[max_excess_domain]:
                    del domain_to_datasets[max_excess_domain]
                test_tokens = sum(item[1]['token_count'] for item in test_items)
                logger.warning(f"Removed dataset {ds_id} from domain {max_excess_domain} with {data['token_count']} tokens, test tokens now: {test_tokens}")
                train_datasets[ds_id] = data
            else:
                logger.error(f"Empty domain_datasets for {max_excess_domain}, skipping removal")
                break
        else:
            if test_items:
                test_items.sort(key=lambda x: x[1]['token_count'], reverse=True)
                removed_item = test_items.pop(0)
                ds_id, data = removed_item
                domain = data['domain']
                domain_to_datasets[domain] = [(id_, d) for id_, d in domain_to_datasets.get(domain, []) if id_ != ds_id]
                if not domain_to_datasets[domain]:
                    del domain_to_datasets[domain]
                test_tokens = sum(item[1]['token_count'] for item in test_items)
                logger.warning(f"Removed dataset {ds_id} from domain {domain} with {data['token_count']} tokens, test tokens now: {test_tokens}")
                train_datasets[ds_id] = data
            else:
                logger.warning("No more datasets to remove, breaking loop")
                break
    
    test_datasets = dict(test_items)
    
    train_domains = [train_datasets[ds_id]['domain'] for ds_id in train_datasets]
    test_domains = [test_datasets[ds_id]['domain'] for ds_id in test_datasets]
    train_distribution = Counter(train_domains)
    test_distribution = Counter(test_domains)
    
    logger.info(f"Train set domain distribution: {train_distribution}")
    logger.info(f"Test set domain distribution: {test_distribution}")
    logger.info(f"Test set contains the following datasets per domain:")
    for domain, count in sorted(test_distribution.items()):
        logger.info(f"  {domain}: {count} dataset{'s' if count != 1 else ''}")
    
    logger.info(f"Final split: Train datasets={len(train_datasets)}, Test datasets={len(test_datasets)}, Test tokens={test_tokens}")
    return train_datasets, test_datasets

def adjust_test_set(train_datasets: Dict[str, Dict], test_datasets: Dict[str, Dict], base_prompt_tokens: int) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    test_items = list(test_datasets.items())
    test_items.sort(key=lambda x: x[1]['token_count'], reverse=True)
    test_tokens = sum(item[1]['token_count'] for item in test_items)
    
    while test_tokens + base_prompt_tokens > MAX_TOKENS * 0.7 and test_items:
        removed_item = test_items.pop(0)
        logger.warning(f"Adjusted removal for test set: Removed dataset {removed_item[0]} with {removed_item[1]['token_count']} tokens")
        train_datasets[removed_item[0]] = removed_item[1]
        test_tokens = sum(item[1]['token_count'] for item in test_items)
    
    logger.info(f"Adjusted test set: Train datasets={len(train_datasets)}, Test datasets={len(test_items)}, Test tokens={test_tokens}")
    return train_datasets, dict(test_items)

def run_classification(datasets: Dict[str, Dict], mode: str, train_datasets: Dict[str, Dict], test_datasets: Dict[str, Dict]) -> Tuple[float, float, Dict]:
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'=' * 80}")
    logger.info(f"=== {mode.upper()} CLASSIFICATION STARTED ===")
    logger.info(f"{'-' * 80}\n")
    
    mode_handler = logging.FileHandler(f'{RESULTS_DIR}/{MODEL_NAME}_{mode}_logs.log', mode='a')
    mode_handler.setLevel(logging.INFO)
    mode_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(mode_handler)
    
    print(f"{mode.upper()} - Number of test datasets: {len(test_datasets)}")
    logger.info(f"{mode.upper()} - Number of test datasets: {len(test_datasets)}")
    
    try:
        classifier = DeepSeekClassifier()
    except ValueError as e:
        logger.error(f"Error initializing classifier for {mode}: {e}")
        logger.removeHandler(mode_handler)
        mode_handler.close()
        return 0.0, 0.0, {}

    if not classifier.domains or not test_datasets:
        logger.error(f"No classification domains or test datasets loaded for {mode}")
        logger.removeHandler(mode_handler)
        mode_handler.close()
        return 0.0, 0.0, {}

    examples = []
    if mode == "one-shot":
        candidate_datasets = [item for item in list(train_datasets.items()) if item[1]['token_count'] < MAX_EXAMPLE_TOKENS]
        if len(candidate_datasets) < 1:
            logger.warning(f"No suitable datasets for one-shot example")
        else:
            selected_dataset = random.sample(candidate_datasets, 1)[0]
            dataset_id, data_info = selected_dataset
            classification = {
                "primary_domain": data_info['domain'],
                "alternative_domains": []
            }
            examples = [(data_info['data_sample'], classification)]
    
    elif mode == "few-shot":
        candidate_datasets = [item for item in list(train_datasets.items()) if item[1]['token_count'] < MAX_EXAMPLE_TOKENS]
        if len(candidate_datasets) < FEW_SHOT_SIZE:
            logger.warning(f"Found only {len(candidate_datasets)} suitable datasets for few-shot examples")
        selected_datasets = random.sample(candidate_datasets, min(FEW_SHOT_SIZE, len(candidate_datasets)))
        for dataset_id, data_info in selected_datasets:
            classification = {
                "primary_domain": data_info['domain'],
                "alternative_domains": []
            }
            examples.append((data_info['data_sample'], classification))

    format_instructions = classifier.parser.get_format_instructions()
    if mode == "zero-shot":
        system_content = (
            "You are a dataset domain classification expert. "
            "Analyze the dataset samples provided as DataFrame-like structures "
            "and determine their primary domains with optional alternatives. "
            "Choose ONLY from the following domains: " + ", ".join(classifier.domains) + "\n\n" +
            format_instructions + "\n\n" +
            "Rules:\n"
            "1. For each test dataset, primary_domain: SINGLE most appropriate domain (MUST be from list)\n"
            "2. alternative_domains: 0-2 other relevant domains (only if applicable)\n"
            "3. Use EXACT domain names\n"
            "4. Output a JSON object with 'results' array containing one object per test dataset in order!\n"
            "5. No additional text/comments!"
        )
    else:
        examples_text = classifier.create_examples_text(examples)
        system_content = (
            "You are a dataset domain classification expert. "
            "Analyze dataset samples and determine their primary domains with optional alternatives. "
            "Choose ONLY from these domains: " + ", ".join(classifier.domains) + "\n\n" +
            format_instructions + "\n\n" +
            "Rules:\n"
            "1. For each test dataset, primary_domain: SINGLE most appropriate domain (MUST be from list)\n"
            "2. alternative_domains: 0-2 other relevant domains (only if applicable)\n"
            "3. Use EXACT domain names\n"
            "4. Output a JSON object with 'results' array containing one object per test dataset in order!\n"
            "5. No additional text/comments!\n\n" +
            "Examples:" + examples_text
        )
    
    logger.debug(f"System prompt for {mode}: {system_content[:1000]}...")
    base_prompt_tokens = estimate_tokens(system_content)
    logger.info(f"Actual base prompt tokens for {mode}: {base_prompt_tokens}")
    
    train_datasets, test_datasets = adjust_test_set(train_datasets, test_datasets, base_prompt_tokens)
    
    matches = 0
    top3_matches = 0
    total = 0
    actual_labels = []
    predicted_labels = []
    
    test_ids = list(test_datasets.keys())
    data_samples = [test_datasets[ds_id]['data_sample'] for ds_id in test_ids]
    dataset_tokens = [test_datasets[ds_id]['token_count'] for ds_id in test_ids]
    
    total_tokens = sum(dataset_tokens) + base_prompt_tokens
    logger.info(f"Total tokens for {mode} batch: {total_tokens} (base={base_prompt_tokens}, datasets={sum(dataset_tokens)})")
    logger.info(f"{mode} mode: Test size={len(test_ids)}, Data samples={len(data_samples)}, Total tokens={total_tokens}")
    logger.info(f"Data samples sizes: {[len(sample) for sample in data_samples]}")
    
    if total_tokens > MAX_TOKENS * 0.7:
        logger.warning(f"Total tokens ({total_tokens}) exceed limit ({MAX_TOKENS * 0.7}), consider reducing test set size")
    
    batch_size = 10
    results = []
    for i in range(0, len(data_samples), batch_size):
        batch_samples = data_samples[i:i + batch_size]
        batch_tokens = dataset_tokens[i:i + batch_size]
        batch_total_tokens = sum(batch_tokens) + base_prompt_tokens
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_samples)} samples, {batch_total_tokens} tokens")
        
        if batch_total_tokens > MAX_TOKENS * 0.7:
            logger.warning(f"Batch {i//batch_size + 1} tokens ({batch_total_tokens}) exceed limit, reducing batch")
            batch_samples = batch_samples[:len(batch_samples)//2]
            batch_tokens = batch_tokens[:len(batch_samples)]
        
        try:
            batch_results = classifier.classify_batch(batch_samples, batch_tokens, mode=mode, examples=examples)
            logger.info(f"Received {len(batch_results)} classification results for batch {i//batch_size + 1}")
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Error in batch {i//batch_size + 1} for {mode}: {e}", exc_info=True)
            results.extend([ClassificationResult(primary_domain="unknown", alternative_domains=[]) for _ in batch_samples])

    logger.info(f"Received {len(results)} classification results for {len(test_ids)} test datasets in {mode} mode")
    
    for i in range(min(len(results), len(test_ids))):
        dataset_id = test_ids[i]
        data_info = test_datasets[dataset_id]
        actual = data_info['domain']
        predicted_primary = results[i].primary_domain.lower()
        
        all_predictions = [predicted_primary] + [alt.lower() for alt in results[i].alternative_domains]
        top3 = all_predictions[:3]

        if predicted_primary == actual:
            matches += 1
        if actual in top3:
            top3_matches += 1
            
        total += 1
        actual_labels.append(actual)
        predicted_labels.append(predicted_primary)
    
    if len(results) < len(test_ids):
        logger.warning(f"Received fewer results ({len(results)}) than test datasets ({len(test_ids)}) in {mode} mode")
        for i in range(len(results), len(test_ids)):
            dataset_id = test_ids[i]
            data_info = test_datasets[dataset_id]
            actual = data_info['domain']
            predicted_primary = "unknown"
            
            actual_labels.append(actual)
            predicted_labels.append(predicted_primary)
            total += 1

    if total > 0:
        accuracy = matches / total
        top3_recall = top3_matches / total
        unique_labels = list(set(actual_labels + predicted_labels))
        logger.info(f"Actual label distribution: {Counter(actual_labels)}")
        logger.info(f"Predicted label distribution: {Counter(predicted_labels)}")
        report = classification_report(actual_labels, predicted_labels, labels=unique_labels, output_dict=True, zero_division=0)
        
        report_file = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_{mode}_classification_report.json")
        labels_file = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_{mode}_labels.json")
        
        existing_reports = []
        existing_labels = []
        
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    existing_reports = json.load(f)
                    if not isinstance(existing_reports, list):
                        existing_reports = [existing_reports]
            except json.JSONDecodeError:
                logger.error(f"Error reading {report_file}, resetting existing reports")
                existing_reports = []
        
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    existing_labels = json.load(f)
                    if not isinstance(existing_labels, list):
                        existing_labels = [existing_labels]
            except json.JSONDecodeError:
                logger.error(f"Error reading {labels_file}, resetting existing labels")
                existing_labels = []
        
        existing_reports.append({
            "classification_report": report,
            "accuracy": accuracy,
            "top3_recall": top3_recall
        })
        existing_labels.append({"actual": actual_labels, "predicted": predicted_labels})
        
        logger.debug(f"Writing to {report_file}: {json.dumps({'classification_report': report, 'accuracy': accuracy, 'top3_recall': top3_recall}, indent=2)}")
        logger.debug(f"Writing to {labels_file}: {json.dumps({'actual': actual_labels, 'predicted': predicted_labels}, indent=2)}")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(existing_reports, f, indent=2, ensure_ascii=False)
        logger.info(f"{mode} classification report saved to {report_file}")
        
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(existing_labels, f, indent=2, ensure_ascii=False)
        logger.info(f"{mode} labels saved to {labels_file}")
        
        logger.info(f"{mode} - Accuracy: {accuracy:.2%}, Top-3 Recall: {top3_recall:.2%}")
        logger.info(f"{'-' * 80}")
        logger.info(f"=== {mode.upper()} CLASSIFICATION COMPLETED: Accuracy={accuracy:.4f}, Top-3 Recall={top3_recall:.4f} ===")
        logger.info(f"{'=' * 80}\n")
        
        logger.removeHandler(mode_handler)
        mode_handler.close()
        
        return accuracy, top3_recall, report
    else:
        logger.warning(f"No datasets processed for {mode}")
        logger.info(f"{'-' * 80}")
        logger.info(f"=== {mode.upper()} CLASSIFICATION COMPLETED: No datasets processed ===")
        logger.info(f"{'=' * 80}\n")
        logger.removeHandler(mode_handler)
        mode_handler.close()
        return 0.0, 0.0, {}

def main():
    logger.info(f"\n{'=' * 80}")
    logger.info("=== CLASSIFICATION EXPERIMENT STARTED ===")
    logger.info(f"{'=' * 80}\n")
    
    logger.info("Loading all datasets...")
    datasets = load_all_datasets()
    
    if not datasets:
        logger.error("No datasets loaded, exiting. Please check the following:")
        logger.error(f"1. Ensure the directory {os.path.abspath(OUTPUT_DIR)} exists and contains dataset subdirectories.")
        logger.error("2. Each dataset subdirectory must contain 'metadata.json' with a 'domain' field and 'full_dataset.csv'.")
        logger.error("3. Verify file permissions and CSV file formats.")
        return
    
    logger.info(f"Loaded {len(datasets)} datasets")
    
    NUM_SPLITS = 5
    
    results = {
        'zero-shot': {'accuracies': [], 'top3_recalls': [], 'macro_precisions': [], 'macro_recalls': [], 'macro_f1s': [], 'weighted_f1s': [], 'distributions': []},
        'one-shot': {'accuracies': [], 'top3_recalls': [], 'macro_precisions': [], 'macro_recalls': [], 'macro_f1s': [], 'weighted_f1s': [], 'distributions': []},
        'few-shot': {'accuracies': [], 'top3_recalls': [], 'macro_precisions': [], 'macro_recalls': [], 'macro_f1s': [], 'weighted_f1s': [], 'distributions': []}
    }
    
    print(f"\n{'=' * 80}")
    print("STARTING CLASSIFICATION EXPERIMENT")
    print(f"Modes: Zero-shot, One-shot, Few-shot")
    print(f"Number of stratified splits per mode: {NUM_SPLITS}")
    print(f"{'=' * 80}\n")

    labels = [datasets[ds_id]['domain'] for ds_id in datasets]
    original_distribution = Counter(labels)
    print(f"Original dataset domain distribution: {original_distribution}")

    for mode in ['zero-shot', 'one-shot', 'few-shot']:
        if mode == 'zero-shot':
            estimated_base_prompt_tokens = 1000
        elif mode == 'one-shot':
            estimated_base_prompt_tokens = 5000
        else:
            estimated_base_prompt_tokens = 15000
        
        print(f"\n{'=' * 60}")
        print(f"{mode.upper()} CLASSIFICATION (Stratified, {NUM_SPLITS} splits)".center(60))
        print(f"{'=' * 60}")
        
        for split_num in range(NUM_SPLITS):
            print(f"\nSplit {split_num + 1}/{NUM_SPLITS} for {mode}")
            logger.info(f"Starting split {split_num + 1}/{NUM_SPLITS} for {mode}")
            
            train_datasets, test_datasets = split_datasets_with_token_limit(
                datasets, 
                TEST_FRACTION, 
                estimated_base_prompt_tokens, 
                random_state=42 + split_num
            )
            print(f"Number of test datasets for {mode} split {split_num + 1} after token limit adjustment: {len(test_datasets)}")
            logger.info(f"Number of test datasets for {mode} split {split_num + 1} after token limit adjustment: {len(test_datasets)}")
            print(f"Stratified split for {mode} split {split_num + 1}: Train datasets={len(train_datasets)}, Test datasets={len(test_datasets)}")
            
            test_domains = [test_datasets[ds_id]['domain'] for ds_id in test_datasets]
            train_domains = [train_datasets[ds_id]['domain'] for ds_id in train_datasets]
            test_distribution = Counter(test_domains)
            train_distribution = Counter(train_domains)
            print(f"Test set contains the following datasets per domain:")
            for domain, count in sorted(test_distribution.items()):
                print(f"  {domain}: {count} dataset{'s' if count != 1 else ''}")
            
            distribution_info = {
                'split': split_num + 1,
                'train_distribution': dict(train_distribution),
                'test_distribution': dict(test_distribution),
            }
            results[mode]['distributions'].append(distribution_info)
            
            accuracy, top3_recall, report = run_classification(datasets, mode, train_datasets, test_datasets)
            
            if report:
                results[mode]['accuracies'].append(accuracy)
                results[mode]['top3_recalls'].append(top3_recall)
                results[mode]['macro_precisions'].append(report['macro avg']['precision'])
                results[mode]['macro_recalls'].append(report['macro avg']['recall'])
                results[mode]['macro_f1s'].append(report['macro avg']['f1-score'])
                results[mode]['weighted_f1s'].append(report['weighted avg']['f1-score'])
                
                print(f"Split {split_num + 1} - Accuracy={accuracy:.4f}, Top-3 Recall={top3_recall:.4f}")
                print(f"Macro Precision={report['macro avg']['precision']:.4f}, "
                      f"Macro Recall={report['macro avg']['recall']:.4f}, "
                      f"Macro F1={report['macro avg']['f1-score']:.4f}, "
                      f"Weighted F1={report['weighted avg']['f1-score']:.4f}")
        
        mean_acc = np.mean(results[mode]['accuracies']) if results[mode]['accuracies'] else 0
        std_acc = np.std(results[mode]['accuracies']) if results[mode]['accuracies'] else 0
        mean_top3 = np.mean(results[mode]['top3_recalls']) if results[mode]['top3_recalls'] else 0
        std_top3 = np.std(results[mode]['top3_recalls']) if results[mode]['top3_recalls'] else 0
        mean_macro_p = np.mean(results[mode]['macro_precisions']) if results[mode]['macro_precisions'] else 0
        std_macro_p = np.std(results[mode]['macro_precisions']) if results[mode]['macro_precisions'] else 0
        mean_macro_r = np.mean(results[mode]['macro_recalls']) if results[mode]['macro_recalls'] else 0
        std_macro_r = np.std(results[mode]['macro_recalls']) if results[mode]['macro_recalls'] else 0
        mean_macro_f1 = np.mean(results[mode]['macro_f1s']) if results[mode]['macro_f1s'] else 0
        std_macro_f1 = np.std(results[mode]['macro_f1s']) if results[mode]['macro_f1s'] else 0
        mean_weighted_f1 = np.mean(results[mode]['weighted_f1s']) if results[mode]['weighted_f1s'] else 0
        std_weighted_f1 = np.std(results[mode]['weighted_f1s']) if results[mode]['weighted_f1s'] else 0
        
        print(f"\n{mode.upper()} Aggregated Results (over {NUM_SPLITS} stratified splits):")
        print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Top-3 Recall: {mean_top3:.4f} ± {std_top3:.4f}")
        print(f"Macro Precision: {mean_macro_p:.4f} ± {std_macro_p:.4f}")
        print(f"Macro Recall: {mean_macro_r:.4f} ± {std_macro_r:.4f}")
        print(f"Macro F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
        print(f"Weighted F1: {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
        
        agg_results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_top3_recall': mean_top3,
            'std_top3_recall': std_top3,
            'mean_macro_precision': mean_macro_p,
            'std_macro_precision': std_macro_p,
            'mean_macro_recall': mean_macro_r,
            'std_macro_recall': std_macro_r,
            'mean_macro_f1': mean_macro_f1,
            'std_macro_f1': std_macro_f1,
            'mean_weighted_f1': mean_weighted_f1,
            'std_weighted_f1': std_weighted_f1,
            'distributions': results[mode]['distributions']
        }
        with open(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_{mode}_aggregated_results.json"), "w", encoding="utf-8") as f:
            json.dump(agg_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'=' * 80}")
    logger.info("=== EXPERIMENT COMPLETED ===")
    logger.info(f"{'=' * 80}\n")
    
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS".center(80))
    print(f"{'=' * 80}")
    
    for mode in ['zero-shot', 'one-shot', 'few-shot']:
        mean_acc = np.mean(results[mode]['accuracies']) if results[mode]['accuracies'] else 0
        std_acc = np.std(results[mode]['accuracies']) if results[mode]['accuracies'] else 0
        mean_top3 = np.mean(results[mode]['top3_recalls']) if results[mode]['top3_recalls'] else 0
        std_top3 = np.std(results[mode]['top3_recalls']) if results[mode]['top3_recalls'] else 0
        mean_macro_p = np.mean(results[mode]['macro_precisions']) if results[mode]['macro_precisions'] else 0
        std_macro_p = np.std(results[mode]['macro_precisions']) if results[mode]['macro_precisions'] else 0
        mean_macro_r = np.mean(results[mode]['macro_recalls']) if results[mode]['macro_recalls'] else 0
        std_macro_r = np.std(results[mode]['macro_recalls']) if results[mode]['macro_recalls'] else 0
        mean_macro_f1 = np.mean(results[mode]['macro_f1s']) if results[mode]['macro_f1s'] else 0
        std_macro_f1 = np.std(results[mode]['macro_f1s']) if results[mode]['macro_f1s'] else 0
        mean_weighted_f1 = np.mean(results[mode]['weighted_f1s']) if results[mode]['weighted_f1s'] else 0
        std_weighted_f1 = np.std(results[mode]['weighted_f1s']) if results[mode]['weighted_f1s'] else 0
        
        print(f"\n{mode.upper().replace('-', ' ')} RESULTS (mean ± std over {NUM_SPLITS} stratified splits):")
        print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Top-3 Recall: {mean_top3:.4f} ± {std_top3:.4f}")
        print(f"Macro Precision: {mean_macro_p:.4f} ± {std_macro_p:.4f}")
        print(f"Macro Recall: {mean_macro_r:.4f} ± {std_macro_r:.4f}")
        print(f"Macro F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
        print(f"Weighted F1: {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
    
    with open(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print("Results saved to deepseekfree_results.json and per-mode aggregated files")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()