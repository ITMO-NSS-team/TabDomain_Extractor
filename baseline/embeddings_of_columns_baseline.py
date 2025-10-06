import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import joblib
from collections import Counter
import pickle
import faiss

INPUT_DIR = "all_datasets"
MODEL_SAVE_PATH = "column_embedding_classifier.pkl"
OUTPUT_DIR = "results"
EMBEDDING_METHOD = "avg"
N_SPLITS = 5

def load_dataset_info():
    dataset_info = {}
    skipped_datasets = []
    domains = []
    
    for dataset_id in os.listdir(INPUT_DIR):
        csv_path = os.path.join(INPUT_DIR, dataset_id, "random_rows.csv")
        meta_path = os.path.join(INPUT_DIR, dataset_id, "metadata.json")
        
        if not os.path.exists(csv_path):
            skipped_datasets.append((dataset_id, "No file random_rows.csv"))
            continue
        if not os.path.exists(meta_path):
            skipped_datasets.append((dataset_id, "No file metadata.json"))
            continue
            
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                domain = metadata.get('domain', 'Other')
                domains.append(domain)
        except Exception as e:
            skipped_datasets.append((dataset_id, f"Error reading metadata.json: {e}"))
            continue
        
        try:
            df = pd.read_csv(csv_path, nrows=0)
            columns = list(df.columns)
        except Exception as e:
            skipped_datasets.append((dataset_id, f"Error reading sample.csv: {e}"))
            continue
        
        dataset_info[dataset_id] = {
            'columns': columns,
            'domain': domain
        }
    
    print(f"Total folders in {INPUT_DIR}: {len(os.listdir(INPUT_DIR))}")
    print(f"Successfully loaded datasets: {len(dataset_info)}")
    if skipped_datasets:
        print("Skipped datasets:")
        for ds_id, reason in skipped_datasets:
            print(f"  {ds_id}: {reason}")
    print("Unique domains:")
    domain_counts = pd.Series(domains).value_counts()
    print(domain_counts)
    print(f"Number of unique domains: {len(domain_counts)}")
    
    return dataset_info, domains

def voting_baseline_predict(columns, embedding_model, domain_embeddings, unique_domains, max_columns=50):
    columns = columns[:max_columns]
    
    col_embeddings = embedding_model.encode(columns, batch_size=128)
    
    col_embeddings = col_embeddings / np.linalg.norm(col_embeddings, axis=1, keepdims=True)
    domain_embeddings_norm = domain_embeddings / np.linalg.norm(domain_embeddings, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(domain_embeddings_norm.shape[1])
    index.add(domain_embeddings_norm)
    
    _, indices = index.search(col_embeddings, 1)
    nearest_domains = [unique_domains[idx[0]] for idx in indices]
    
    domain_counts = Counter(nearest_domains)
    feature_vector = np.array([domain_counts.get(d, 0) / len(columns) for d in unique_domains])
    most_common_domain = domain_counts.most_common(1)[0][0]
    
    return feature_vector, most_common_domain

def generate_embeddings(dataset_info, embedding_model, domains, method="avg", cache_file="embeddings_cache.pkl", max_columns=50):
    if os.path.exists(cache_file):
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f), list(set(domains))
    
    unique_domains = list(set(domains))
    embeddings = {}
    
    if method == "voting":
        domain_cache_file = "domain_embeddings_cache.pkl"
        if os.path.exists(domain_cache_file):
            with open(domain_cache_file, 'rb') as f:
                domain_embeddings = pickle.load(f)
        else:
            domain_embeddings = embedding_model.encode(unique_domains, batch_size=128)
            with open(domain_cache_file, 'wb') as f:
                pickle.dump(domain_embeddings, f)
    
    for ds_id, info in dataset_info.items():
        try:
            columns = info['columns']
            if method == "avg":
                col_embeddings = embedding_model.encode(columns[:max_columns], batch_size=128)
                feature_vector = np.mean(col_embeddings, axis=0)
                embeddings[ds_id] = {
                    'embedding': feature_vector,
                    'predicted_domain': None,
                    'domain': info['domain']
                }
            elif method == "voting":
                feature_vector, most_common_domain = voting_baseline_predict(
                    columns=columns,
                    embedding_model=embedding_model,
                    domain_embeddings=domain_embeddings,
                    unique_domains=unique_domains,
                    max_columns=max_columns
                )
                embeddings[ds_id] = {
                    'embedding': feature_vector,
                    'predicted_domain': most_common_domain,
                    'domain': info['domain']
                }
        except Exception as e:
            print(f"Error creating embedding for {ds_id}: {e}")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings, unique_domains

def train_models(X, y):
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    results = {}
    for name, model in models.items():
        print(f"Training and cross-validating model: {name}")
        
        acc_scores = []
        prec_macro_scores = []
        rec_macro_scores = []
        f1_macro_scores = []
        prec_weighted_scores = []
        rec_weighted_scores = []
        f1_weighted_scores = []
        
        y_pred_all = np.zeros_like(y)
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            y_pred_all[test_idx] = y_pred
            
            acc_scores.append(accuracy_score(y_test, y_pred))
            prec_macro_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            rec_macro_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_macro_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            prec_weighted_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec_weighted_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_weighted_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        results[name] = {
            'model': model,
            'y_pred': y_pred_all,
            'metrics': {
                'Accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
                'Precision (Macro)': {'mean': np.mean(prec_macro_scores), 'std': np.std(prec_macro_scores)},
                'Recall (Macro)': {'mean': np.mean(rec_macro_scores), 'std': np.std(rec_macro_scores)},
                'F1-Score (Macro)': {'mean': np.mean(f1_macro_scores), 'std': np.std(f1_macro_scores)},
                'Precision (Weighted)': {'mean': np.mean(prec_weighted_scores), 'std': np.std(prec_weighted_scores)},
                'Recall (Weighted)': {'mean': np.mean(rec_weighted_scores), 'std': np.std(rec_weighted_scores)},
                'F1-Score (Weighted)': {'mean': np.mean(f1_weighted_scores), 'std': np.std(f1_weighted_scores)}
            }
        }
        
        print(f"Accuracy: {results[name]['metrics']['Accuracy']['mean']:.2f} ± {results[name]['metrics']['Accuracy']['std']:.2f}")
        print(f"F1-Score (Macro): {results[name]['metrics']['F1-Score (Macro)']['mean']:.2f} ± {results[name]['metrics']['F1-Score (Macro)']['std']:.2f}")
        print(f"F1-Score (Weighted): {results[name]['metrics']['F1-Score (Weighted)']['mean']:.2f} ± {results[name]['metrics']['F1-Score (Weighted)']['std']:.2f}")
    
    return models, results

def clear_cache():
    cache_files = [
        "baseline_embeddings_cache.pkl",
        "classification_embeddings_cache.pkl",
        "domain_embeddings_cache.pkl"
    ]
    for file in cache_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted cache file: {file}")

def main():
    print("Loading dataset information...")
    dataset_info, domains = load_dataset_info()
    print(f"Loaded {len(dataset_info)} datasets")
    
    if not dataset_info:
        print("No datasets found for processing")
        return
    
    print("Initializing embedding model...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  
    
    print("Creating embeddings for voting method...")
    baseline_embeddings, unique_domains = generate_embeddings(
        dataset_info, 
        embedding_model, 
        domains, 
        method="voting", 
        cache_file="baseline_embeddings_cache.pkl"
    )
    
    if not baseline_embeddings:
        print("Failed to create embeddings for voting method")
        return
    
    print("\nPreparing data for voting method evaluation")
    dataset_ids = list(baseline_embeddings.keys())
    y_true = [data['domain'] for data in baseline_embeddings.values()]
    predicted_domains = [data['predicted_domain'] for data in baseline_embeddings.values()]
    
    print("\nVoting method evaluation:")
    mismatches = [(ds_id, true, pred) for ds_id, true, pred in zip(dataset_ids, y_true, predicted_domains) if true != pred]
    print(f"Domain mismatches (true vs. baseline): {len(mismatches)} from {len(dataset_ids)}")
    if mismatches:
        print("Mismatch examples:")
        for ds_id, true, pred in mismatches[:min(5, len(mismatches))]:
            print(f"Dataset {ds_id}: True={true}, Predicted (baseline)={pred}, Columns={', '.join(dataset_info[ds_id]['columns'])}")
    
    baseline_acc = accuracy_score(y_true, predicted_domains)
    baseline_prec_macro = precision_score(y_true, predicted_domains, average='macro', zero_division=0)
    baseline_rec_macro = recall_score(y_true, predicted_domains, average='macro', zero_division=0)
    baseline_f1_macro = f1_score(y_true, predicted_domains, average='macro', zero_division=0)
    baseline_prec_weighted = precision_score(y_true, predicted_domains, average='weighted', zero_division=0)
    baseline_rec_weighted = recall_score(y_true, predicted_domains, average='weighted', zero_division=0)
    baseline_f1_weighted = f1_score(y_true, predicted_domains, average='weighted', zero_division=0)
    
    print("\nBaseline metrics:")
    print(f"Accuracy: {baseline_acc:.2f} ± 0.00")
    print(f"F1-Score (Macro): {baseline_f1_macro:.2f} ± 0.00")
    print(f"F1-Score (Weighted): {baseline_f1_weighted:.2f} ± 0.00")
    
    print("\nCreating embeddings for classification...")
    classification_embeddings, _ = generate_embeddings(
        dataset_info, 
        embedding_model, 
        domains, 
        method=EMBEDDING_METHOD, 
        cache_file="classification_embeddings_cache.pkl"
    )
    
    if not classification_embeddings:
        print("Failed to create embeddings for classification")
        return
    
    print("Preparing data for classification")
    X = np.array([data['embedding'] for data in classification_embeddings.values()])
    y_true = [data['domain'] for data in classification_embeddings.values()]
    dataset_ids = list(classification_embeddings.keys())
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_true)
    
    print("\nDomain distribution:")
    domain_counts = pd.Series(y_true).value_counts()
    print(domain_counts)
    print(f"Number of unique domains: {len(domain_counts)}")
    
    print("\nTraining models and cross-validation")
    models, cv_results = train_models(X, y_encoded)
    
    print("\nBaseline and models comparison:")
    comparison = []
    
    comparison.append({
        'Model': 'Voting',
        'Accuracy': f"{baseline_acc:.2f} ± 0.00",
        'Precision (Macro)': f"{baseline_prec_macro:.2f} ± 0.00",
        'Recall (Macro)': f"{baseline_rec_macro:.2f} ± 0.00",
        'F1-Score (Macro)': f"{baseline_f1_macro:.2f} ± 0.00",
        'Precision (Weighted)': f"{baseline_prec_weighted:.2f} ± 0.00",
        'Recall (Weighted)': f"{baseline_rec_weighted:.2f} ± 0.00",
        'F1-Score (Weighted)': f"{baseline_f1_weighted:.2f} ± 0.00"
    })
    
    for model_name, res in cv_results.items():
        metrics = res['metrics']
        comparison.append({
            'Model': model_name,
            'Accuracy': f"{metrics['Accuracy']['mean']:.2f} ± {metrics['Accuracy']['std']:.2f}",
            'Precision (Macro)': f"{metrics['Precision (Macro)']['mean']:.2f} ± {metrics['Precision (Macro)']['std']:.2f}",
            'Recall (Macro)': f"{metrics['Recall (Macro)']['mean']:.2f} ± {metrics['Recall (Macro)']['std']:.2f}",
            'F1-Score (Macro)': f"{metrics['F1-Score (Macro)']['mean']:.2f} ± {metrics['F1-Score (Macro)']['std']:.2f}",
            'Precision (Weighted)': f"{metrics['Precision (Weighted)']['mean']:.2f} ± {metrics['Precision (Weighted)']['std']:.2f}",
            'Recall (Weighted)': f"{metrics['Recall (Weighted)']['mean']:.2f} ± {metrics['Recall (Weighted)']['std']:.2f}",
            'F1-Score (Weighted)': f"{metrics['F1-Score (Weighted)']['mean']:.2f} ± {metrics['F1-Score (Weighted)']['std']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    best_model_name = None
    best_accuracy = 0.0
    
    for model_name, res in cv_results.items():
        if res['metrics']['Accuracy']['mean'] > best_accuracy:
            best_accuracy = res['metrics']['Accuracy']['mean']
            best_model_name = model_name
    
    if best_accuracy <= baseline_acc:
        best_model_name = 'Voting'
    
    print(f"\nBest model: {best_model_name}")
    
    if best_model_name != 'Voting':
        best_model = cv_results[best_model_name]['model']
        best_model.fit(X, y_encoded)
        joblib.dump({
            'model': best_model,
            'label_encoder': label_encoder,
            'embedding_model': embedding_model,
            'class_names': label_encoder.classes_
        }, MODEL_SAVE_PATH)
        print(f"Best model saved as {MODEL_SAVE_PATH}")
    
    if best_model_name != 'Voting':
        y_pred = cv_results[best_model_name]['y_pred']
        y_pred_domain = label_encoder.inverse_transform(y_pred)
        errors = []
        for i, ds_id in enumerate(dataset_ids):
            if y_true[i] != y_pred_domain[i]:
                errors.append({
                    'dataset_id': ds_id,
                    'columns': dataset_info[ds_id]['columns'],
                    'actual_domain': y_true[i],
                    'predicted_domain': y_pred_domain[i],
                    'predicted_domain_baseline': predicted_domains[i]
                })
        
        print(f"\nClassification errors found ({best_model_name}): {len(errors)} from {len(dataset_ids)}")
        if errors:
            print("\nClassification error examples:")
            for error in errors[:min(5, len(errors))]:
                print(f"\nDataset {error['dataset_id']}:")
                print(f"  True domain: {error['actual_domain']}")
                print(f"  Predicted domain (model): {error['predicted_domain']}")
                print(f"  Predicted domain (baseline): {error['predicted_domain_baseline']}")
                print(f"  Columns: {', '.join(error['columns'])}")
    
    print("\nSaving results...")
    all_results = []
    for i, ds_id in enumerate(dataset_ids):
        result_row = {
            'dataset_id': ds_id,
            'actual_domain': y_true[i],
            'predicted_domain_baseline': predicted_domains[i],
            'correct_baseline': int(y_true[i] == predicted_domains[i]),
            'columns': ', '.join(dataset_info[ds_id]['columns'])
        }
        for model_name in models.keys():
            y_pred_domain = label_encoder.inverse_transform(cv_results[model_name]['y_pred'])
            result_row[f'predicted_{model_name}'] = y_pred_domain[i]
            result_row[f'correct_{model_name}'] = int(y_true[i] == y_pred_domain[i])
        all_results.append(result_row)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('classification_results.csv', index=False)
    comparison_df.to_csv('comparison_metrics.csv', index=False)

if __name__ == "__main__":
    clear_cache()
    main()