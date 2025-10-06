import os
import json
import logging
import random
from collections import defaultdict
import numpy as np
import networkx as nx
from itertools import combinations
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_eqs(term_to_cols):
    eq_dict = defaultdict(list)
    for term, cols in term_to_cols.items():
        key = tuple(sorted(cols))
        eq_dict[key].append(term)
    eqs = list(eq_dict.values())
    term_to_eq = {term: i for i, eq in enumerate(eqs) for term in eq}
    return eqs, term_to_eq

def columns_to_eq(columns, eqs, term_to_eq):
    eq_columns = []
    for col in columns:
        eq_col = set(term_to_eq.get(term, -1) for term in col if term in term_to_eq)
        if -1 not in eq_col:
            eq_columns.append(eq_col)
    return eq_columns

def eq_to_terms(eq_set, eqs):
    terms = set()
    for e in eq_set:
        terms.update(eqs[e])
    return terms

def jaccard_eq(e1, e2, eq_to_cols):
    c1 = eq_to_cols.get(e1, set())
    c2 = eq_to_cols.get(e2, set())
    inter = len(c1 & c2)
    union = len(c1 | c2)
    return inter / union if union > 0 else 0.0

def context_signature_eq(e, all_eqs, eq_to_cols):
    sig = [(e2, jaccard_eq(e, e2, eq_to_cols)) for e2 in all_eqs if e2 != e and jaccard_eq(e, e2, eq_to_cols) > 0]
    sig.sort(key=lambda x: -x[1])
    return sig

def drop_index(sim_vec, start):
    if len(sim_vec) <= start + 1:
        return len(sim_vec)
    max_diff = -1
    drop_idx = start
    for i in range(start, len(sim_vec) - 1):
        diff = sim_vec[i] - sim_vec[i + 1]
        if diff > max_diff:
            max_diff = diff
            drop_idx = i + 1
    return drop_idx

def signature_blocks_eq(e, all_eqs, eq_to_cols):
    sig = context_signature_eq(e, all_eqs, eq_to_cols)
    if not sig:
        return []
    terms, sims = zip(*sig) if sig else ([], [])
    blocks = []
    start = 0
    while start < len(sims):
        drop = drop_index(sims, start)
        block = set(terms[start:drop])
        blocks.append(block)
        start = drop
    return blocks

def liberal_blocks(blocks):
    if not blocks:
        return []
    sizes = [len(b) for b in blocks]
    max_idx = np.argmax(sizes)
    return blocks[:max_idx]

def prune_centrist_eq(e, C_eq, blocks):
    x = [(set(), 0.0)]
    lib_blocks = liberal_blocks(blocks)
    for B in lib_blocks:
        inter = len(B & C_eq)
        rel = inter / len(B) if len(B) > 0 else 0.0
        x.append((B, rel))
    x.sort(key=lambda y: -y[1])
    rel_vec = [rel for _, rel in x]
    drop = drop_index(rel_vec, 1)
    robust = set()
    for i in range(1, drop):
        robust.update(x[i][0])
    return robust

def prune_conservative_eq(blocks):
    return blocks[0] if blocks else set()

def column_expand(C_eq, tau_sup, delta_dec, all_eqs, eq_to_cols, eqs, pruning_strategy='conservative'):
    C_plus = C_eq.copy()
    tau_col = tau_sup
    while tau_col > 0:
        C_prime = set()
        if pruning_strategy == 'conservative':
            robsigs = {t: prune_conservative_eq(signature_blocks_eq(t, all_eqs, eq_to_cols)) for t in C_plus}
        else:
            robsigs = {t: prune_centrist_eq(t, C_plus, signature_blocks_eq(t, all_eqs, eq_to_cols)) for t in C_plus}
        candidates = set()
        for robsig in robsigs.values():
            candidates.update(robsig)
        candidates -= C_plus
        for t_prime in candidates:
            S = set(t for t in C_plus if t_prime in robsigs[t])
            if len(S) / len(C_plus) > tau_sup and len(S & C_eq) / len(C_eq) > tau_col:
                C_prime.add(t_prime)
        if C_prime:
            C_plus.update(C_prime)
            tau_col -= delta_dec
        else:
            break
    return C_plus

def local_domains(C_plus, all_eqs, eq_to_cols, eqs, pruning_strategy='conservative'):
    G = nx.Graph()
    G.add_nodes_from(C_plus)
    for t1, t2 in combinations(C_plus, 2):
        if pruning_strategy == 'conservative':
            robsig1 = prune_conservative_eq(signature_blocks_eq(t1, all_eqs, eq_to_cols))
            robsig2 = prune_conservative_eq(signature_blocks_eq(t2, all_eqs, eq_to_cols))
        else:
            robsig1 = prune_centrist_eq(t1, C_plus, signature_blocks_eq(t1, all_eqs, eq_to_cols))
            robsig2 = prune_centrist_eq(t2, C_plus, signature_blocks_eq(t2, all_eqs, eq_to_cols))
        if t2 in robsig1 or t1 in robsig2:
            G.add_edge(t1, t2)
    components = list(nx.connected_components(G))
    local = [set(comp) for comp in components if len(comp) > 1]
    return local

def jaccard_sets(s1, s2):
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0

def domain_support(L, tau, all_local_domains):
    support = 0
    for other_locals in all_local_domains:
        for other_L in other_locals:
            if jaccard_sets(L, other_L) > tau:
                support += 1
                break
    return support

def strong_domains(all_local_domains, tau_dsim, tau_est, tau_strong):
    strong = []
    for locals in all_local_domains:
        for L in locals:
            sup_dsim = domain_support(L, tau_dsim, all_local_domains)
            sup_est = domain_support(L, tau_est, all_local_domains)
            if sup_est > 0 and sup_dsim / sup_est > tau_strong:
                strong.append(L)
    return strong

def generate_domain_terms(domain_representative_datasets, pruning_strategy='conservative', use_column_expansion=True, tau_dsim=0.3, tau_est=0.05, tau_strong=0.1):
    all_columns = []
    for dataset in tqdm(domain_representative_datasets, desc="Processing datasets for domain terms"):
        all_columns.extend(dataset)
    
    if not all_columns:
        return set()
    
    term_to_cols = defaultdict(set)
    for i, col in enumerate(all_columns):
        for term in col:
            term_to_cols[term].add(i)
    
    eqs, term_to_eq = compute_eqs(term_to_cols)
    if not eqs:
        all_terms = set()
        for col in all_columns:
            all_terms.update(col)
        return all_terms
    
    eq_columns = columns_to_eq(all_columns, eqs, term_to_eq)
    eq_to_cols = defaultdict(set)
    for term, cols in term_to_cols.items():
        eq_idx = term_to_eq[term]
        eq_to_cols[eq_idx].update(cols)
    
    all_eqs = list(range(len(eqs)))
    all_local_domains = []
    for C_eq in tqdm(eq_columns, desc="Computing local domains"):
        if use_column_expansion:
            C_plus = column_expand(C_eq, tau_sup=0.25, delta_dec=0.05, all_eqs=all_eqs, eq_to_cols=eq_to_cols, eqs=eqs, pruning_strategy=pruning_strategy)
        else:
            C_plus = C_eq
        local = local_domains(C_plus, all_eqs, eq_to_cols, eqs, pruning_strategy=pruning_strategy)
        all_local_domains.append(local)
    
    strong = strong_domains(all_local_domains, tau_dsim, tau_est, tau_strong)
    
    if not strong:
        all_terms = set()
        for col in all_columns:
            all_terms.update(col)
        return all_terms
    
    strong_terms = [eq_to_terms(L, eqs) for L in strong]
    
    domain_terms = set()
    for dom in strong_terms:
        domain_terms.update(dom)
    
    return domain_terms

def filter_common_terms(domain_terms_dict, min_domain_freq=0.8):
    term_counts = defaultdict(int)
    total_domains = len(domain_terms_dict)
    
    for domain, terms in domain_terms_dict.items():
        for term in terms:
            term_counts[term] += 1
    
    common_terms = {term for term, count in term_counts.items() if count / total_domains >= min_domain_freq}
    
    filtered_dict = {}
    for domain, terms in domain_terms_dict.items():
        filtered_dict[domain] = terms - common_terms
    
    return filtered_dict

def generate_multi_domain_terms(domains_representative):
    domain_terms_dict = {}
    for domain, reps in tqdm(domains_representative.items(), desc="Generating domain terms"):
        domain_terms = generate_domain_terms(reps)
        domain_terms_dict[domain] = domain_terms
    return filter_common_terms(domain_terms_dict)

def compute_similarity(dataset_columns, domain_terms):
    extracted_terms = set()
    for col in dataset_columns:
        extracted_terms.update(col)
    
    matched_terms = extracted_terms.intersection(domain_terms)
    sim_score = len(matched_terms) / min(len(extracted_terms), len(domain_terms)) if min(len(extracted_terms), len(domain_terms)) > 0 else 0.0
    return sim_score

def classify_dataset(dataset_columns, domain_terms_dict):
    scores = {}
    max_score = -1
    predicted_domain = None
    for domain, terms in domain_terms_dict.items():
        score = compute_similarity(dataset_columns, terms)
        scores[domain] = score
        if score > max_score:
            max_score = score
            predicted_domain = domain
    return predicted_domain, scores

def bootstrap_metrics(true_labels, pred_labels, n_iterations=1000):
    n_samples = len(true_labels)
    if n_samples == 0:
        return {key: 0.0 for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']}
    
    accuracies = []
    precisions_macro = []
    recalls_macro = []
    f1s_macro = []
    precisions_weighted = []
    recalls_weighted = []
    f1s_weighted = []
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_true = [true_labels[i] for i in indices]
        boot_pred = [pred_labels[i] for i in indices]
        
        accuracies.append(accuracy_score(boot_true, boot_pred))
        precisions_macro.append(precision_score(boot_true, boot_pred, average='macro', zero_division=0))
        recalls_macro.append(recall_score(boot_true, boot_pred, average='macro', zero_division=0))
        f1s_macro.append(f1_score(boot_true, boot_pred, average='macro', zero_division=0))
        precisions_weighted.append(precision_score(boot_true, boot_pred, average='weighted', zero_division=0))
        recalls_weighted.append(recall_score(boot_true, boot_pred, average='weighted', zero_division=0))
        f1s_weighted.append(f1_score(boot_true, boot_pred, average='weighted', zero_division=0))
    
    return {
        'accuracy': np.std(accuracies),
        'precision_macro': np.std(precisions_macro),
        'recall_macro': np.std(recalls_macro),
        'f1_macro': np.std(f1s_macro),
        'precision_weighted': np.std(precisions_weighted),
        'recall_weighted': np.std(recalls_weighted),
        'f1_weighted': np.std(f1s_weighted)
    }

def compute_classification_metrics(true_labels, pred_labels, unique_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    precision_weighted = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

def save_results(metrics_data, predictions, output_path="classification_results.csv"):
    results_df = pd.DataFrame(metrics_data)
    predictions_df = pd.DataFrame(predictions, columns=["Dataset", "True Domain", "Predicted Domain", "Scores"])
    with open(output_path, 'w') as f:
        f.write("=== Quality Metrics ===\n")
        results_df.to_csv(f, index=False)
        f.write("\n=== Predictions ===\n")
        predictions_df.to_csv(f, index=False)
    logger.info(f"Results saved to {output_path}")

def discretize_numeric_column(column, bins=10):
    try:
        valid_data = pd.to_numeric(column.dropna(), errors='coerce')
        if valid_data.empty:
            return set()
        
        min_val, max_val = valid_data.min(), valid_data.max()
        if min_val == max_val:
            return {str(min_val)}
        
        bin_edges = pd.qcut(valid_data, q=bins, duplicates='drop').cat.categories
        if len(bin_edges) < 2:
            return {str(min_val)}
        
        labels = [f"{bin_edges[i].left:.2f}-{bin_edges[i].right:.2f}" for i in range(len(bin_edges))]
        discretized = pd.cut(valid_data, bins=bin_edges, labels=labels, include_lowest=True, ordered=False)
        return set(discretized.dropna().astype(str))
    except Exception as e:
        logger.warning(f"Error discretizing column: {str(e)}")
        return set()

OUTPUT_DIR = "all_datasets"
TAG_FILE = "unique_tags.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_datasets_as_columns():
    dataset_columns = []
    unique_domains = set()
    
    subdirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    for subdir in tqdm(subdirs, desc="Loading datasets"):
        try:
            dataset_id = int(subdir)
            metadata_path = os.path.join(OUTPUT_DIR, subdir, "metadata.json")
            csv_path = os.path.join(OUTPUT_DIR, subdir, "random_rows.csv")
            
            if not os.path.exists(metadata_path) or not os.path.exists(csv_path):
                logger.warning(f"Skipped {subdir}: missing metadata.json or random_rows.csv")
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            domain = metadata.get('domain', None)
            if not domain:
                logger.warning(f"Domain missing in {subdir}")
                continue
            
            data = pd.read_csv(csv_path)
            columns_sets = []
            for col in data.columns:
                if data[col].dtype == 'object':
                    unique_vals = set(data[col].dropna().astype(str))
                    if unique_vals:
                        columns_sets.append(unique_vals)
                else:
                    unique_vals = discretize_numeric_column(data[col])
                    if unique_vals:
                        columns_sets.append(unique_vals)
            
            if columns_sets:
                dataset_columns.append((dataset_id, columns_sets, domain))
                unique_domains.add(domain)
        
        except Exception as e:
            logger.error(f"Error loading {subdir}: {str(e)}")
            continue
    
    return dataset_columns, unique_domains

def stratified_train_test_split(dataset_columns, test_size=0.2):
    domain_groups = defaultdict(list)
    for ds_id, cols, dom in dataset_columns:
        domain_groups[dom].append((ds_id, cols, dom))
    
    train_data = []
    test_data = []
    
    for dom, group in domain_groups.items():
        random.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * test_size)) if n > 1 else 0
        n_train = n - n_test
        
        if n_train == 0 and n > 0:
            n_train = 1
            n_test = 0
        
        train_data.extend(group[:n_train])
        test_data.extend(group[n_train:])
    
    return train_data, test_data

def evaluate_classification(test_datasets, domain_terms_dict):
    true_labels = []
    pred_labels = []
    predictions = []
    for name, columns, true_domain in tqdm(test_datasets, desc="Classifying datasets"):
        pred_domain, scores = classify_dataset(columns, domain_terms_dict)
        true_labels.append(true_domain)
        pred_labels.append(pred_domain)
        predictions.append((name, true_domain, pred_domain, scores))
    
    unique_labels = list(set(true_labels + pred_labels))
    results = compute_classification_metrics(true_labels, pred_labels, unique_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    std_metrics = bootstrap_metrics(true_labels, pred_labels)
    
    return results, cm, unique_labels, predictions, std_metrics

def main():
    dataset_columns, unique_domains = load_datasets_as_columns()
    
    if not dataset_columns:
        logger.error("No suitable datasets for processing!")
        return
    
    logger.info(f"Loaded datasets: {len(dataset_columns)}")
    logger.info(f"Unique domains: {len(unique_domains)}")
    
    train_data, test_data = stratified_train_test_split(dataset_columns)
    
    logger.info(f"Train datasets: {len(train_data)}")
    logger.info(f"Test datasets: {len(test_data)}")
    
    domains_representative_train = defaultdict(list)
    for _, cols, dom in train_data:
        domains_representative_train[dom].append(cols)
    
    domain_terms_dict = generate_multi_domain_terms(domains_representative_train)
    
    print("Domain Terms Dict:")
    for domain, terms in domain_terms_dict.items():
        print(f"{domain}: {sorted(terms)[:10]}...")  
    
    test_datasets = [(f"Dataset_{ds_id}", cols, dom) for ds_id, cols, dom in test_data]
    
    results, cm, unique_labels, predictions, std_metrics = evaluate_classification(test_datasets, domain_terms_dict)
    
    print("\n=== Quality Metrics ===")
    metrics_data = [
        {'Metric': 'Accuracy', 'Value': f"{results['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}"},
        {'Metric': 'Precision (Macro)', 'Value': f"{results['precision_macro']:.4f} ± {std_metrics['precision_macro']:.4f}"},
        {'Metric': 'Recall (Macro)', 'Value': f"{results['recall_macro']:.4f} ± {std_metrics['recall_macro']:.4f}"},
        {'Metric': 'F1-Score (Macro)', 'Value': f"{results['f1_macro']:.4f} ± {std_metrics['f1_macro']:.4f}"},
        {'Metric': 'Precision (Weighted)', 'Value': f"{results['precision_weighted']:.4f} ± {std_metrics['precision_weighted']:.4f}"},
        {'Metric': 'Recall (Weighted)', 'Value': f"{results['recall_weighted']:.4f} ± {std_metrics['recall_weighted']:.4f}"},
        {'Metric': 'F1-Score (Weighted)', 'Value': f"{results['f1_weighted']:.4f} ± {std_metrics['f1_weighted']:.4f}"}
    ]
    for metric in metrics_data:
        print(f"{metric['Metric']}: {metric['Value']}")
    
    print("\nConfusion Matrix:\n", cm)
    print("Unique Labels:", unique_labels)
    print("\nPredictions:")
    for name, true, pred, scores in predictions:
        print(f"{name}: True={true}, Pred={pred}, Scores={scores}")

    save_results(metrics_data, predictions)

if __name__ == "__main__":
    main()