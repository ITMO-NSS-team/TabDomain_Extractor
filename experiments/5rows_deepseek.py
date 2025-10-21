import json
import os
import requests
import pandas as pd
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, stop_after_attempt, wait_fixed
import gc
import chardet
import warnings
import random

warnings.filterwarnings('ignore')

OUTPUT_DIR = "all_datasets"
RESULTS_DIR = "classification_results"
DOMAIN_MAPPING_FILE = "clustering_domain_mapping.json"
API_KEY = os.getenv("API_KEY")
TIMEOUT = 50
N_RUNS = 1
K_FOLDS = 5
FEW_SHOT_SIZE = 5
os.makedirs(RESULTS_DIR, exist_ok=True)

if not API_KEY:
    raise ValueError("API_KEY environment variable not set!")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def detect_encoding(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(50000)
        result = chardet.detect(rawdata)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
    except Exception as e:
        logger.warning(f"Encoding detection error: {e}, using utf-8")
        return 'utf-8'

def read_csv_with_memory_limit(file_path: str, dataset_id: str = None) -> pd.DataFrame:
    try:
        if dataset_id:
            random_path = os.path.join(OUTPUT_DIR, dataset_id, "random_rows.csv")
            if os.path.exists(random_path):
                encoding = detect_encoding(random_path)
                df = pd.read_csv(random_path, encoding=encoding)
                logger.debug(f"Read random rows for {dataset_id} ({len(df)} rows, {len(df.columns)} columns)")
                return df

        encoding = detect_encoding(file_path)
        logger.debug(f"Detected encoding: {encoding} for {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding=encoding) as f:
            preview = pd.read_csv(f, nrows=1000)
        
        dtype_dict = preview.dtypes.to_dict()
        
        chunks = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(
            file_path, 
            encoding=encoding,
            chunksize=chunk_size,
            dtype=dtype_dict
        ):
            chunks.append(chunk)
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.debug(f"Read dataset with {len(df)} rows and {len(df.columns)} columns")
            return df
        else:
            logger.warning("No data read, returning empty DataFrame")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

def prepare_for_llm(df: pd.DataFrame, max_rows: int = 5) -> str:
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

class DeepSeekChatModel(BaseChatModel):
    api_key: str = Field(default=API_KEY)
    model_name: str = "deepseek-chat"
    endpoint_url: str = "https://api.deepseek.com/v1/chat/completions"
    timeout: int = TIMEOUT

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        api_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                api_messages.append({"role": "system", "content": m.content})
            else:
                api_messages.append({"role": "user", "content": m.content})
        
        payload = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": 0.3,
            "max_tokens": 2048,
            "stop": stop if stop else [],
            "stream": False
        }
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            content = response_data["choices"][0]["message"]["content"]
            
            if "```json" in content and "```" in content:
                try:
                    content = content.split("```json")[1].split("```")[0].strip()
                except IndexError:
                    logger.warning("Failed to parse JSON block from content")
            elif "{" in content and "}" in content:
                try:
                    content = content[content.index("{"):content.rindex("}") + 1]
                except ValueError:
                    logger.warning("Failed to extract JSON object from content")
            
            return ChatResult(generations=[ChatGeneration(message=HumanMessage(content=content))])
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            return ChatResult(generations=[ChatGeneration(message=HumanMessage(content="{}"))])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "endpoint_url": self.endpoint_url}

class ClassificationResult(BaseModel):
    primary_domain: str = Field(..., description="The single most appropriate domain")
    alternative_domains: List[str] = Field([], description="List of alternative relevant domains", max_items=2)

class LLMClassifier:
    def __init__(self):
        self.domains = self.load_domains()
        self.llm = DeepSeekChatModel()
        self.parser = PydanticOutputParser(pydantic_object=ClassificationResult)
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

    def create_examples_text(self, examples: List[Tuple[str, str]]) -> str:
        examples_text = ""
        for i, (data_sample, domain) in enumerate(examples, 1):
            examples_text += f"\nExample {i}:\nDataset sample:\n{data_sample}\nCorrect domain: {domain}\n"
        return examples_text

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(20))
    def classify(self, df: pd.DataFrame, mode: str = "zero-shot", examples: List[Tuple[str, str]] = None) -> List[str]:
        if df.empty or not self.domains:
            logger.warning("Empty DataFrame or no domains for classification")
            return ["unknown"] * 3

        try:
            data_sample = prepare_for_llm(df)
            
            system_message = (
                "You are a dataset domain classification expert. "
                "Analyze the dataset sample provided as a DataFrame-like structure "
                "(columns and data rows) and determine its primary domain. "
                "Choose ONLY from the following domains: {allowed_labels}\n\n"
                "Respond in strict JSON format: {{\"primary_domain\": \"domain\", \"alternative_domains\": [\"domain1\", \"domain2\"]}}\n"
                "Where:\n"
                "- primary_domain: The single most appropriate domain (MUST be from the list)\n"
                "- alternative_domains: Up to 2 other relevant domains (only if applicable)\n"
                "Use EXACT domain names. No additional comments!"
            )
            
            human_message = (
                "Here's a sample of a dataset represented as DataFrame:"
                "{dataframe}"
                "Classify this dataset. Use format: {format_instructions}"
            )
            
            if mode in ["one-shot", "few-shot"] and examples:
                human_message = (
                    "Here are some examples of correct classifications:"
                    "{examples}"
                    "\n\nNow classify this new dataset sample:"
                    "{dataframe}"
                    "Use format: {format_instructions}"
                )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", human_message)
            ])
            
            chain = prompt | self.llm | self.parser
            
            invoke_params = {
                "dataframe": data_sample,
                "allowed_labels": ", ".join(self.domains),
                "format_instructions": self.parser.get_format_instructions()
            }

            if mode in ["one-shot", "few-shot"] and examples:
                invoke_params["examples"] = self.create_examples_text(examples)
            
            result = chain.invoke(invoke_params)
            
            domains = [result.primary_domain.lower()]
            for alt in result.alternative_domains:
                if alt.lower() not in domains and len(domains) < 3:
                    domains.append(alt.lower())
            
            while len(domains) < 3:
                domains.append("unknown")
            
            logger.debug(f"Classification result ({mode}): {domains}")
            return domains
        except Exception as e:
            logger.error(f"Classification error ({mode}): {e}", exc_info=True)
            return ["unknown"] * 3

def load_all_datasets() -> Dict[str, Dict]:
    datasets = {}
    
    if not os.path.exists(OUTPUT_DIR):
        logger.error(f"Dataset directory not found: {os.path.abspath(OUTPUT_DIR)}")
        return datasets

    dataset_ids = [d for d in os.listdir(OUTPUT_DIR) 
                   if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    valid_datasets = [
        d for d in dataset_ids
        if os.path.exists(os.path.join(OUTPUT_DIR, d, "metadata.json"))
        and os.path.exists(os.path.join(OUTPUT_DIR, d, "random_rows.csv"))
    ]
    
    logger.info(f"Loading {len(valid_datasets)} valid datasets...")
    
    for dataset_id in tqdm(valid_datasets, desc="Loading datasets"):
        try:
            meta_path = os.path.join(OUTPUT_DIR, dataset_id, "metadata.json")
            data_path = os.path.join(OUTPUT_DIR, dataset_id, "random_rows.csv")

            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            if "domain" not in metadata:
                logger.warning(f"Skipping {dataset_id}: 'domain' field missing in metadata")
                continue

            df = read_csv_with_memory_limit(data_path, dataset_id=dataset_id)
            if df.empty:
                logger.warning(f"Empty DataFrame for {dataset_id}, skipping")
                continue

            datasets[dataset_id] = {
                'metadata': metadata,
                'dataframe': df,
                'domain': metadata["domain"].lower(),
                'data_sample': prepare_for_llm(df)
            }
                
        except Exception as e:
            logger.error(f"Error loading {dataset_id}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets

def run_zero_shot_classification(datasets: Dict[str, Dict], run_id: int) -> Tuple[Optional[float], List[str], List[str], Optional[Dict], Optional[float]]:
    logger.info(f"=== Starting zero-shot classification run {run_id} ===")
    classifier = LLMClassifier()

    if not classifier.domains or not datasets:
        logger.error("Classification domains or datasets not loaded")
        return None, [], [], {}, None

    test_size = len(datasets) // K_FOLDS
    dataset_items = list(datasets.items())
    
    random.shuffle(dataset_items)
    test_datasets = dict(dataset_items[:test_size])
    
    matches = 0
    top3_matches = 0
    total = 0
    actual_labels = []
    predicted_labels = []
    
    pbar = tqdm(test_datasets.items(), desc=f"Zero-shot classification (Run {run_id})", unit="dataset")

    for dataset_id, data_info in pbar:
        try:
            predicted_domains = classifier.classify(data_info['dataframe'], mode="zero-shot")
            actual = data_info['domain']

            if predicted_domains[0] == actual:
                matches += 1
            if actual in predicted_domains:
                top3_matches += 1
            total += 1

            actual_labels.append(actual)
            predicted_labels.append(predicted_domains[0])
                
            pbar.set_postfix({"Accuracy": f"{matches/total:.2%}" if total > 0 else "N/A",
                            "Top-3 Recall": f"{top3_matches/total:.2%}" if total > 0 else "N/A"})
                
        except Exception as e:
            logger.error(f"Error processing {dataset_id}: {e}")
            continue

    if total > 0:
        accuracy = matches / total
        top3_recall = top3_matches / total
        logger.info(f"Zero-shot accuracy: {accuracy:.2%}")
        logger.info(f"Zero-shot top-3 recall: {top3_recall:.2%}")
        
        report = classification_report(actual_labels, predicted_labels, output_dict=True, zero_division=0)
        
        return accuracy, actual_labels, predicted_labels, report, top3_recall
    else:
        logger.warning("No datasets processed for accuracy calculation")
        return None, [], [], {}, None

def run_kfold_classification(datasets: Dict[str, Dict], mode: str, run_id: int) -> Tuple[List[float], List[float], List[Dict]]:
    logger.info(f"=== Starting {mode} classification with K-fold validation (Run {run_id}) ===")
    classifier = LLMClassifier()

    if not classifier.domains or not datasets:
        logger.error("Classification domains or datasets not loaded")
        return [], [], []

    dataset_items = list(datasets.items())
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42 + run_id)
    
    fold_accuracies = []
    fold_top3_recalls = []
    fold_reports = []

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(dataset_items)):
        logger.info(f"Processing fold {fold_idx + 1}/{K_FOLDS}")
        
        train_datasets = [dataset_items[i] for i in train_indices]
        test_datasets = [dataset_items[i] for i in test_indices]
        
        examples = []
        if mode == "one-shot":
            domain_examples = {}
            for dataset_id, data_info in train_datasets:
                domain = data_info['domain']
                if domain not in domain_examples:
                    domain_examples[domain] = (data_info['data_sample'], domain)
            examples = list(domain_examples.values())
        
        elif mode == "few-shot":
            domain_examples = {}
            for dataset_id, data_info in train_datasets:
                domain = data_info['domain']
                if domain not in domain_examples:
                    domain_examples[domain] = []
                if len(domain_examples[domain]) < FEW_SHOT_SIZE:
                    domain_examples[domain].append((data_info['data_sample'], domain))
            
            for domain_list in domain_examples.values():
                examples.extend(domain_list)
        
        matches = 0
        top3_matches = 0
        total = 0
        actual_labels = []
        predicted_labels = []
        
        pbar = tqdm(test_datasets, desc=f"{mode.capitalize()} fold {fold_idx + 1}", unit="dataset")
        
        for dataset_id, data_info in pbar:
            try:
                predicted_domains = classifier.classify(
                    data_info['dataframe'], 
                    mode=mode, 
                    examples=examples
                )
                actual = data_info['domain']

                if predicted_domains[0] == actual:
                    matches += 1
                if actual in predicted_domains:
                    top3_matches += 1
                total += 1

                actual_labels.append(actual)
                predicted_labels.append(predicted_domains[0])
                    
                pbar.set_postfix({"Accuracy": f"{matches/total:.2%}" if total > 0 else "N/A",
                                "Top-3 Recall": f"{top3_matches/total:.2%}" if total > 0 else "N/A"})
                    
            except Exception as e:
                logger.error(f"Error processing {dataset_id}: {e}")
                continue

        if total > 0:
            fold_accuracy = matches / total
            fold_top3_recall = top3_matches / total
            fold_report = classification_report(actual_labels, predicted_labels, output_dict=True, zero_division=0)
            
            fold_accuracies.append(fold_accuracy)
            fold_top3_recalls.append(fold_top3_recall)
            fold_reports.append(fold_report)
            
            logger.info(f"Fold {fold_idx + 1} - Accuracy: {fold_accuracy:.2%}, Top-3 Recall: {fold_top3_recall:.2%}")

    return fold_accuracies, fold_top3_recalls, fold_reports

def calculate_aggregated_metrics(fold_reports: List[Dict]) -> Dict[str, float]:
    if not fold_reports:
        return {}
    
    macro_precisions = [report['macro avg']['precision'] for report in fold_reports]
    macro_recalls = [report['macro avg']['recall'] for report in fold_reports]
    macro_f1s = [report['macro avg']['f1-score'] for report in fold_reports]
    weighted_f1s = [report['weighted avg']['f1-score'] for report in fold_reports]
    
    return {
        'macro_precision_mean': np.mean(macro_precisions),
        'macro_precision_std': np.std(macro_precisions),
        'macro_recall_mean': np.mean(macro_recalls),
        'macro_recall_std': np.std(macro_recalls),
        'macro_f1_mean': np.mean(macro_f1s),
        'macro_f1_std': np.std(macro_f1s),
        'weighted_f1_mean': np.mean(weighted_f1s),
        'weighted_f1_std': np.std(weighted_f1s)
    }

def main():
    logger.info("Loading all datasets...")
    datasets = load_all_datasets()
    
    if not datasets:
        logger.error("No datasets loaded, exiting")
        return
    
    logger.info(f"Loaded {len(datasets)} datasets")
    
    results = {
        'zero-shot': {
            'accuracies': [],
            'top3_recalls': [],
            'macro_precisions': [],
            'macro_recalls': [],
            'macro_f1s': [],
            'weighted_f1s': []
        },
        'one-shot': {
            'accuracies': [],
            'top3_recalls': [],
            'macro_precisions': [],
            'macro_recalls': [],
            'macro_f1s': [],
            'weighted_f1s': []
        },
        'few-shot': {
            'accuracies': [],
            'top3_recalls': [],
            'macro_precisions': [],
            'macro_recalls': [],
            'macro_f1s': [],
            'weighted_f1s': []
        }
    }
    
    print(f"\n{'=' * 80}")
    print(f"STARTING CLASSIFICATION EXPERIMENT")
    print(f"Modes: Zero-shot, One-shot, Few-shot")
    print(f"Runs per mode: {N_RUNS}")
    print(f"K-Folds for shot learning: {K_FOLDS}")
    print(f"{'=' * 80}\n")
    
    print(f"\n{'=' * 60}")
    print("ZERO-SHOT CLASSIFICATION".center(60))
    print(f"{'=' * 60}")
    
    for run_id in range(1, N_RUNS + 1):
        accuracy, actual_labels, predicted_labels, report, top3_recall = run_zero_shot_classification(datasets, run_id)
        
        if accuracy is not None:
            results['zero-shot']['accuracies'].append(accuracy)
            results['zero-shot']['top3_recalls'].append(top3_recall)
            results['zero-shot']['macro_precisions'].append(report['macro avg']['precision'])
            results['zero-shot']['macro_recalls'].append(report['macro avg']['recall'])
            results['zero-shot']['macro_f1s'].append(report['macro avg']['f1-score'])
            results['zero-shot']['weighted_f1s'].append(report['weighted avg']['f1-score'])
            
            print(f"Run {run_id}: Accuracy={accuracy:.4f}, Top-3 Recall={top3_recall:.4f}")
    
    print(f"\n{'=' * 60}")
    print("ONE-SHOT CLASSIFICATION".center(60))
    print(f"{'=' * 60}")
    
    for run_id in range(1, N_RUNS + 1):
        fold_accuracies, fold_top3_recalls, fold_reports = run_kfold_classification(datasets, "one-shot", run_id)
        
        if fold_accuracies:
            run_accuracy = np.mean(fold_accuracies)
            run_top3_recall = np.mean(fold_top3_recalls)
            run_metrics = calculate_aggregated_metrics(fold_reports)
            
            results['one-shot']['accuracies'].append(run_accuracy)
            results['one-shot']['top3_recalls'].append(run_top3_recall)
            results['one-shot']['macro_precisions'].append(run_metrics['macro_precision_mean'])
            results['one-shot']['macro_recalls'].append(run_metrics['macro_recall_mean'])
            results['one-shot']['macro_f1s'].append(run_metrics['macro_f1_mean'])
            results['one-shot']['weighted_f1s'].append(run_metrics['weighted_f1_mean'])
            
            print(f"Run {run_id}: Accuracy={run_accuracy:.4f}, Top-3 Recall={run_top3_recall:.4f}")
    
    print(f"\n{'=' * 60}")
    print("FEW-SHOT CLASSIFICATION".center(60))
    print(f"{'=' * 60}")
    
    for run_id in range(1, N_RUNS + 1):
        fold_accuracies, fold_top3_recalls, fold_reports = run_kfold_classification(datasets, "few-shot", run_id)
        
        if fold_accuracies:
            run_accuracy = np.mean(fold_accuracies)
            run_top3_recall = np.mean(fold_top3_recalls)
            run_metrics = calculate_aggregated_metrics(fold_reports)
            
            results['few-shot']['accuracies'].append(run_accuracy)
            results['few-shot']['top3_recalls'].append(run_top3_recall)
            results['few-shot']['macro_precisions'].append(run_metrics['macro_precision_mean'])
            results['few-shot']['macro_recalls'].append(run_metrics['macro_recall_mean'])
            results['few-shot']['macro_f1s'].append(run_metrics['macro_f1_mean'])
            results['few-shot']['weighted_f1s'].append(run_metrics['weighted_f1_mean'])
            
            print(f"Run {run_id}: Accuracy={run_accuracy:.4f}, Top-3 Recall={run_top3_recall:.4f}")
    
    print(f"\n{'=' * 80}")
    print("FINAL AGGREGATED RESULTS".center(80))
    print(f"{'=' * 80}")
    
    final_results = {}
    
    for mode in ['zero-shot', 'one-shot', 'few-shot']:
        mode_results = results[mode]
        
        if mode_results['accuracies']:
            final_results[mode] = {
                'accuracy': {
                    'mean': np.mean(mode_results['accuracies']),
                    'std': np.std(mode_results['accuracies'])
                },
                'top3_recall': {
                    'mean': np.mean(mode_results['top3_recalls']),
                    'std': np.std(mode_results['top3_recalls'])
                },
                'macro_precision': {
                    'mean': np.mean(mode_results['macro_precisions']),
                    'std': np.std(mode_results['macro_precisions'])
                },
                'macro_recall': {
                    'mean': np.mean(mode_results['macro_recalls']),
                    'std': np.std(mode_results['macro_recalls'])
                },
                'macro_f1': {
                    'mean': np.mean(mode_results['macro_f1s']),
                    'std': np.std(mode_results['macro_f1s'])
                },
                'weighted_f1': {
                    'mean': np.mean(mode_results['weighted_f1s']),
                    'std': np.std(mode_results['weighted_f1s'])
                }
            }
            
            print(f"\n{mode.upper().replace('-', ' ')} RESULTS:")
            print(f"Accuracy: {final_results[mode]['accuracy']['mean']:.4f} ± {final_results[mode]['accuracy']['std']:.4f}")
            print(f"Top-3 Recall: {final_results[mode]['top3_recall']['mean']:.4f} ± {final_results[mode]['top3_recall']['std']:.4f}")
            print(f"Macro Precision: {final_results[mode]['macro_precision']['mean']:.4f} ± {final_results[mode]['macro_precision']['std']:.4f}")
            print(f"Macro Recall: {final_results[mode]['macro_recall']['mean']:.4f} ± {final_results[mode]['macro_recall']['std']:.4f}")
            print(f"Macro F1: {final_results[mode]['macro_f1']['mean']:.4f} ± {final_results[mode]['macro_f1']['std']:.4f}")
            print(f"Weighted F1: {final_results[mode]['weighted_f1']['mean']:.4f} ± {final_results[mode]['weighted_f1']['std']:.4f}")
    
    with open(os.path.join(RESULTS_DIR, "final_all_modes_results.json"), "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print("Results saved to final_all_modes_results.json")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()