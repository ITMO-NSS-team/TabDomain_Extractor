import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymfe.mfe import MFE
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import logging
import gc
import joblib
import torch
from torch import nn, optim
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUTPUT_DIR = "all_datasets"
RESULTS_DIR = "meta_features_results"
MODEL_SAVE_PATH = "domain_classifier.pkl"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('domain_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_features(X):
    threshold = 0.5
    X = X.loc[:, X.isna().mean() < threshold]
    logger.info(f"After removing columns with missing values (>50%): {X.shape[1]} features")

    X = X.loc[:, X.nunique() > 1]
    logger.info(f"After removing constants: {X.shape[1]} features")

    X = X.loc[:, ~X.T.duplicated()]
    logger.info(f"After removing duplicates: {X.shape[1]} features")

    try:
        X_imputed = SimpleImputer(strategy='median').fit_transform(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X_imputed, i) for i in range(X_imputed.shape[1])]
        selected_features = vif_data[vif_data["VIF"] < 10]["feature"]
        X = X[selected_features]
        logger.info(f"After VIF filtering (VIF < 10): {X.shape[1]} features")
    except Exception as e:
        logger.warning(f"VIF filtering error: {str(e)}. Skipping VIF.")
    
    return X

def lasso_feature_selection(X, y, lambda_=0.01, threshold=1e-3, epochs=500, lr=0.01):
    if not isinstance(y, np.ndarray):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None

    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X)
    logger.info("Applied Yeo-Johnson scaling before Lasso")

    C = len(np.unique(y))
    X_np = np.array(X_scaled) if not isinstance(X_scaled, np.ndarray) else X_scaled
    N, D = X_np.shape

    device = torch.device('cpu')
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    model = nn.Linear(D, C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Lasso training"):
        pred = model(X_t)
        loss_ce = criterion(pred, y_t)
        l1_reg = lambda_ * torch.sum(torch.abs(model.weight))
        loss = loss_ce + l1_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    weights = model.weight.detach().cpu().numpy()
    abs_max_weights = np.max(np.abs(weights), axis=0)

    selected_indices = np.where(abs_max_weights > threshold)[0]
    selected_columns = X.columns[selected_indices].tolist() if hasattr(X, 'columns') else selected_indices

    logger.info(f"Selected {len(selected_columns)} meta-features from {D} (lambda={lambda_}, threshold={threshold})")
    return selected_columns, abs_max_weights, scaler

def extract_meta_features():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset_ids = [d for d in os.listdir(OUTPUT_DIR) 
                   if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    all_features = []
    failed_datasets = []
    domains = []

    for dataset_id in tqdm(dataset_ids, desc="Extracting meta-features"):
        try:
            meta_path = os.path.join(OUTPUT_DIR, dataset_id, "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            domain = metadata.get("domain")
            if not domain:
                logger.warning(f"Domain missing for dataset {dataset_id}")
                continue
                
            data_path = os.path.join(OUTPUT_DIR, dataset_id, "random_rows.csv")
            if not os.path.exists(data_path):
                logger.warning(f"File random_rows.csv missing for {dataset_id}")
                continue
                
            df = pd.read_csv(data_path)
            
            if df.empty:
                logger.warning(f"Empty dataset: {dataset_id}")
                continue
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            groups = ["general", "statistical", "info-theory"]
            if numeric_cols:
                groups.append("model-based")
            
            mfe = MFE(
                groups=groups,
                summary=["mean", "sd"]
            )
            
            cat_features = {}
            if categorical_cols:
                try:
                    cat_mfe = MFE(groups=["general", "info-theory"])
                    cat_mfe.fit(df[categorical_cols].values)
                    cat_features_list = cat_mfe.extract()
                    cat_features = dict(zip(cat_features_list[0], cat_features_list[1]))
                except Exception as e:
                    logger.warning(f"Error processing categorical features for {dataset_id}: {str(e)}")
            
            num_features = {}
            if numeric_cols:
                try:
                    mfe.fit(df[numeric_cols].values)
                    num_features_list = mfe.extract()
                    num_features = dict(zip(num_features_list[0], num_features_list[1]))
                except Exception as e:
                    logger.warning(f"Error processing numerical features for {dataset_id}: {str(e)}")
            
            features_dict = {**num_features, **cat_features}
            features_dict["dataset_id"] = dataset_id
            features_dict["domain"] = domain
            
            features_dict["num_numeric_cols"] = len(numeric_cols)
            features_dict["num_categorical_cols"] = len(categorical_cols)
            
            all_features.append(features_dict)
            domains.append(domain)
            
            del df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error for {dataset_id}: {str(e)}")
            failed_datasets.append(dataset_id)
    
    if all_features:
        features_df = pd.DataFrame(all_features)
        features_df.to_csv(os.path.join(RESULTS_DIR, "meta_features_with_domains.csv"), index=False)
        logger.info(f"Meta-features saved to meta_features_with_domains.csv")
        
    logger.info(f"Successfully processed: {len(all_features)} datasets")
    logger.info(f"Processing errors: {len(failed_datasets)} datasets")
    
    domain_counts = pd.Series(domains).value_counts()
    logger.info("\nDomain distribution:")
    logger.info(domain_counts.to_string())
    
    return features_df

def train_domain_classifier(features_df):
    if features_df is None or features_df.empty:
        raise ValueError("No training data")
    
    X = features_df.drop(columns=["dataset_id", "domain"], errors="ignore")
    y = features_df["domain"]
    
    X = preprocess_features(X)
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    selected_columns, weights, scaler = lasso_feature_selection(X_imputed, y)
    if not selected_columns:
        logger.warning("No selected meta-features! Using all.")
    else:
        X = X[selected_columns]
        logger.info(f"Using selected meta-features: {selected_columns}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    models = {
        'RandomForest': make_pipeline(
            SimpleImputer(strategy='median'),
            RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            )
        ),
        'LogisticRegression': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            )
        ),
        'SVM': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        ),
        'KNN': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        )
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    logger.info("\nStarting cross-validation (StratifiedKFold)")
    for name, model in models.items():
        try:
            logger.info(f"Evaluating model: {name}")
            
            cv_results = cross_validate(
                model, X, y_encoded, cv=skf, n_jobs=1,
                scoring=['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro'],
                return_train_score=False
            )
            
            y_pred = cross_val_predict(model, X, y_encoded, cv=skf, n_jobs=1)
            
            report = classification_report(
                y_encoded, y_pred, 
                target_names=class_names, 
                output_dict=True,
                zero_division=0
            )
            
            results[name] = {
                'model': model,
                'accuracy': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
                'precision_macro': cv_results['test_precision_macro'].mean(),
                'precision_macro_std': cv_results['test_precision_macro'].std(),
                'recall_macro': cv_results['test_recall_macro'].mean(),
                'recall_macro_std': cv_results['test_recall_macro'].std(),
                'f1_macro': cv_results['test_f1_macro'].mean(),
                'f1_macro_std': cv_results['test_f1_macro'].std(),
                'f1_weighted': cv_results['test_f1_weighted'].mean(),
                'f1_weighted_std': cv_results['test_f1_weighted'].std(),
                'report': report,
                'y_true': y_encoded,
                'y_pred': y_pred,
                'class_names': class_names,
                'cv_results': cv_results
            }

            logger.info(f"  Accuracy: {results[name]['accuracy']:.4f} ± {results[name]['accuracy_std']:.4f}")
            logger.info(f"  Precision: {results[name]['precision_macro']:.4f} ± {results[name]['precision_macro_std']:.4f}")
            logger.info(f"  Recall: {results[name]['recall_macro']:.4f} ± {results[name]['recall_macro_std']:.4f}")
            logger.info(f"  F1 (macro): {results[name]['f1_macro']:.4f} ± {results[name]['f1_macro_std']:.4f}")
            logger.info(f"  F1 (weighted): {results[name]['f1_weighted']:.4f} ± {results[name]['f1_weighted_std']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating model {name}: {str(e)}")
            results[name] = {
                'model': None,
                'accuracy': 0,
                'accuracy_std': 0,
                'f1_macro': 0,
                'f1_macro_std': 0,
                'f1_weighted': 0,
                'f1_weighted_std': 0,
                'report': {},
                'y_true': [],
                'y_pred': [],
                'class_names': []
            }
    
    successful_models = {k: v for k, v in results.items() if v['model'] is not None}
    
    if not successful_models:
        logger.error("No models were successfully trained!")
        return results, None
    
    best_model_name = max(successful_models, key=lambda k: successful_models[k]['accuracy'])
    best_model = successful_models[best_model_name]
    
    logger.info(f"\nBest model: {best_model_name} with accuracy {best_model['accuracy']:.4f} ± {best_model['accuracy_std']:.4f}")
    
    try:
        best_model_instance = best_model['model']
        best_model_instance.fit(X, y_encoded)
        
        joblib.dump({
            'model': best_model_instance,
            'label_encoder': le,
            'feature_names': X.columns.tolist(),
            'scaler': scaler
        }, os.path.join(RESULTS_DIR, MODEL_SAVE_PATH))
        
        logger.info(f"Best model saved to {MODEL_SAVE_PATH}")
        
        best_model['trained_model'] = best_model_instance
    except Exception as e:
        logger.error(f"Error training the best model on all data: {str(e)}")
        best_model['trained_model'] = None
    
    save_results(results, features_df, RESULTS_DIR)
    
    return results, best_model

def save_results(results, features_df, output_dir):
    metrics = []
    for name, res in results.items():
        if res['model'] is None:
            continue
            
        metrics.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Accuracy Std': res['accuracy_std'],
            'Precision (Macro)': res['precision_macro'],
            'Precision (Macro) Std': res['precision_macro_std'],
            'Recall (Macro)': res['recall_macro'],
            'Recall (Macro) Std': res['recall_macro_std'],
            'F1 (Macro)': res['f1_macro'],
            'F1 (Macro) Std': res['f1_macro_std'],
            'F1 (Weighted)': res['f1_weighted'],
            'F1 (Weighted) Std': res['f1_weighted_std'],
        })
    
    if not metrics:
        logger.warning("No metrics to save")
        return None
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)
    
    formatted_metrics = []
    for _, row in metrics_df.iterrows():
        formatted_metrics.append({
            'Model': row['Model'],
            'Accuracy': f"{row['Accuracy']:.2f} ± {row['Accuracy Std']:.2f}",
            'Precision': f"{row['Precision (Macro)']:.2f} ± {row['Precision (Macro) Std']:.2f}",
            'Recall': f"{row['Recall (Macro)']:.2f} ± {row['Recall (Macro) Std']:.2f}",
            'F1 (Macro)': f"{row['F1 (Macro)']:.2f} ± {row['F1 (Macro) Std']:.2f}",
            'F1 (Weighted)': f"{row['F1 (Weighted)']:.2f} ± {row['F1 (Weighted) Std']:.2f}"
        })
    
    formatted_df = pd.DataFrame(formatted_metrics)
    
    print("\n" + "="*80)
    print("Domain Classification Models Comparison")
    print("="*80)
    print(formatted_df.to_string(index=False))
    print("="*80)
    
    best_model_name = None
    best_accuracy = 0
    for name, res in results.items():
        if res['model'] is not None and res['accuracy'] > best_accuracy:
            best_model_name = name
            best_accuracy = res['accuracy']
    
    if best_model_name is None:
        logger.warning("No successful models for reporting")
        return metrics_df
    
    best_result = results[best_model_name]
    y_true = best_result['y_true']
    y_pred = best_result['y_pred']
    class_names = best_result['class_names']
    
    print("\n" + "="*80)
    print(f"Classification Report for Best Model ({best_model_name}):")
    print("="*80)
    print(classification_report(
        y_true, y_pred, 
        target_names=class_names,
        zero_division=0
    ))
    
    predictions = []
    for i, row in features_df.iterrows():
        predictions.append({
            'dataset_id': row['dataset_id'],
            'domain': row['domain'],
            'predicted_domain': class_names[results[best_model_name]['y_pred'][i]]
        })
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(output_dir, "domain_predictions.csv"), index=False)
    
    errors = predictions_df[predictions_df['domain'] != predictions_df['predicted_domain']]
    if not errors.empty:
        errors.to_csv(os.path.join(output_dir, "classification_errors.csv"), index=False)
        logger.info(f"Found {len(errors)} classification errors. Saved to classification_errors.csv")
    
        print("\n" + "="*80)
        print("Top-5 Classification Errors:")
        print("="*80)
        print(errors.head(5).to_string(index=False))
    
    return metrics_df

def main():
    features_df = extract_meta_features()
    
    if features_df is not None and not features_df.empty:
        results, best_model = train_domain_classifier(features_df)
        
        if best_model is None or best_model.get('trained_model') is None:
            logger.error("Failed to train any model!")
            return
        
        best_model_name = max(results, key=lambda k: results[k].get('accuracy', 0))
        if results[best_model_name].get('y_true') is not None and results[best_model_name].get('y_pred') is not None:
            print("\n" + "="*80)
            print("Final Classification Report:")
            print("="*80)
            print(classification_report(
                results[best_model_name]['y_true'],
                results[best_model_name]['y_pred'],
                target_names=results[best_model_name].get('class_names', [])
            ))

if __name__ == "__main__":
    main()