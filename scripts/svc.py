import os
import h5py
import time
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================

# --- Path to your datasets ---
DATASET_PATHS = {
    "fixed": Path("/PathtoYourDataSet/ASCAD.h5"),
    "variable": Path("/PathtoYourDataset/ASCADVariable.h5"),
}

# --- Define SVC Experiments ---
N_FEATURES_FOR_SVC = 100
EXPERIMENTS_TO_RUN = [
    # Example Runs 
    {'dataset': 'variable', 'n_features': N_FEATURES_FOR_SVC, 'params': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}},
    
    
]

# (Global constants and SBOX remain the same)
TARGET_KEY_BYTE_INDEX = 2
PLOT_MAX_TRACES = 1000
PLOT_STEP = 10
RANDOM_STATE = 42
SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def setup_output_directory(base_dir, config):
    n_feat = config['n_features']
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    param_str = "_".join([f"{k}{v}" for k, v in config['params'].items()]).replace("_", "").replace(".", "")
    dir_name = f"SVC_{config['dataset']}_feat{n_feat}_{param_str}_{timestamp}"
    output_path = Path(base_dir) / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / 'summary.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()], force=True)
    return output_path

def load_and_preprocess_data(dataset_type):
    path = DATASET_PATHS[dataset_type]
    if not path.exists(): return None
    with h5py.File(path, "r") as f:
        X_prof_raw = f["Profiling_traces/traces"][:]
        y_prof = f["Profiling_traces/labels"][:]
        X_atk_raw = f["Attack_traces/traces"][:]
        pt_atk = f["Attack_traces/metadata"]["plaintext"][:].astype(np.uint8)
        key_atk = int(f["Attack_traces/metadata"]["key"][0][TARGET_KEY_BYTE_INDEX])
    
    # Normalize data either using StandardScaler or manual normalization
    mean, std = X_prof_raw.mean(), X_prof_raw.std()
    X_prof_norm = (X_prof_raw - mean) / std
    X_atk_norm = (X_atk_raw - mean) / std
    
    return X_prof_norm, y_prof, X_atk_norm, pt_atk, key_atk

def compute_key_rank(model_probs, plaintexts, true_key, n_traces):
    if n_traces == 0 or n_traces > model_probs.shape[0]: return 255, np.zeros(256)
    probs = np.log(np.maximum(model_probs[:n_traces], 1e-40))
    key_scores = np.zeros(256)
    for k_guess in range(256):
        sbox_out = SBOX[plaintexts[:n_traces, TARGET_KEY_BYTE_INDEX] ^ k_guess]
        key_scores[k_guess] = np.sum(probs[np.arange(n_traces), sbox_out])
    ranked_keys = np.argsort(key_scores)[::-1]
    return np.where(ranked_keys == true_key)[0][0], key_scores

def select_features_with_rf(X_profiling, y_profiling, X_attack, n_features):
    selector_model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1)
    selector_model.fit(X_profiling, y_profiling)
    indices = np.argsort(selector_model.feature_importances_)[::-1]
    top_k_indices = indices[:n_features]
    return X_profiling[:, top_k_indices], X_attack[:, top_k_indices]

# ==============================================================================
# 3. EXPERIMENT EXECUTION
# ==============================================================================
def run_svc_experiment(config, output_dir):
    data = load_and_preprocess_data(config['dataset'])
    if data is None: return None
    X_prof, y_prof, X_atk, pt_atk, key_atk = data
    X_prof_final, X_atk_final = select_features_with_rf(X_prof, y_prof, X_atk, config['n_features'])
    
    logging.info("Training scikit-learn SVC model (This will be slow)...")
    model = SVC(probability=True, random_state=RANDOM_STATE, **config['params'])
    start_time = time.time()
    model.fit(X_prof_final, y_prof)
    training_time = time.time() - start_time
    logging.info(f"Model training finished in {training_time:.2f} seconds.")

    logging.info("Evaluating model using predict_proba (slow but original method)...")
    eval_start_time = time.time()
    attack_probs = model.predict_proba(X_atk_final)
    logging.info(f"Probability prediction finished in {time.time() - eval_start_time:.2f} seconds.")

    # (The rest of the function remains the same)
    ranks, traces_to_rank_0 = [], -1
    max_eval_traces = min(PLOT_MAX_TRACES, X_atk_final.shape[0])
    trace_counts = range(0, max_eval_traces + 1, PLOT_STEP)
    for n in trace_counts:
        rank, _ = compute_key_rank(attack_probs, pt_atk, key_atk, n)
        ranks.append(rank)
        if rank == 0 and traces_to_rank_0 == -1: traces_to_rank_0 = n
    
    plt.figure(figsize=(10, 6)); plt.plot(trace_counts, ranks, marker='o'); plt.title(f"Key Rank vs. Traces\n{config}"); plt.xlabel("Num Traces"); plt.ylabel("Key Rank"); plt.grid(True); plt.ylim(-5, 260); plt.savefig(output_dir / "key_rank_vs_traces.png"); plt.close()
    final_rank, final_scores = compute_key_rank(attack_probs, pt_atk, key_atk, max_eval_traces)
    predicted_key = np.argmax(final_scores)
    
    results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'dataset': config['dataset'],
        'n_features': config['n_features'], **{f"param_{k}": v for k, v in config['params'].items()},
        'training_time_s': round(training_time, 2), 'traces_to_rank_0': traces_to_rank_0,
        'final_rank': final_rank, 'predicted_key': predicted_key, 'true_key': key_atk,
        'attack_successful': bool(predicted_key == key_atk and traces_to_rank_0 != -1)
    }
    return results

# ==============================================================================
# 4. MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("⚠️ WARNING: This may be very slow, especially for the variable-key dataset.")
    base_output_directory = "svc_sklearn_experiments_output_"
    csv_path = Path(base_output_directory) / "master_svc_sklearn_results.csv"
    csv_headers = [ 'timestamp', 'dataset', 'n_features', 'training_time_s', 'traces_to_rank_0', 'final_rank', 'predicted_key', 'true_key', 'attack_successful', 'param_C', 'param_gamma', 'param_kernel' ]
    Path(base_output_directory).mkdir(exist_ok=True)
    file_exists = csv_path.is_file()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
        if not file_exists: writer.writeheader()
        for i, experiment_config in enumerate(EXPERIMENTS_TO_RUN):
            output_path = setup_output_directory(base_output_directory, experiment_config)
            try:
                result_data = run_svc_experiment(experiment_config, output_path)
                if result_data: writer.writerow(result_data); f.flush()
            except Exception as e:
                logging.error(f"FATAL ERROR in experiment: {experiment_config}", exc_info=True)