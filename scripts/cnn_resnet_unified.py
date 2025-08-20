import os
import h5py
import time
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================

# --- Path to your datasets ---
DATASET_PATHS = {
    "fixed": Path("/PathtoYourDataSet/ASCAD.h5"),
    "variable": Path("/PathtoYourDataset/ASCADVariable.h5"),
}

# --- Define Final CNN Experiments ---
BEST_FIXED_CONFIG = {
    "epochs": 150,
    "batch_size": 128,
    "learning_rate": 1e-5,
    "model_params": {
        "num_blocks": 4,
        "kernel_size": 11,
        "filters": [64, 128, 256, 512],
    },
}

EXPERIMENTS_TO_RUN = [
    # ==================================================
    # == Part 1: FIXED KEY DATASET - Find the best model
    # ==================================================
    {
        "dataset": "fixed",
        **BEST_FIXED_CONFIG,
        "architecture": "resnet",
        "optimizer": "rmsprop",
    },
    # {'dataset': 'fixed', **BEST_FIXED_CONFIG, 'architecture': 'resnet', 'optimizer': 'rmsprop'},
    # ==================================================
    # == Part 2: VARIABLE KEY DATASET - Try to find a working model
    # ==================================================
    # --- 2.1: Test ResNet (often more robust) ---
    {
        "dataset": "variable",
        "epochs": 150,
        "batch_size": 128,
        "learning_rate": 1e-5,
        "optimizer": "rmsprop",
        "architecture": "resnet",
        "model_params": {
            "num_blocks": 4,
            "kernel_size": 11,
            "filters": [64, 128, 256, 512],
        },
    },
    ]


# (Global constants and SBOX remain the same)
TARGET_KEY_BYTE_INDEX = 2
# Reduce the number of traces in plots for clarity
PLOT_MAX_TRACES = 1000
PLOT_STEP = 10
SBOX = np.array(
    [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    ],
    dtype=np.uint8,
)

# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================


# --- Architecture 1: Standard CNN (VGG-style) ---
class Standard_SCA_CNN(nn.Module):
    def __init__(self, input_length=700, num_blocks=4, kernel_size=11, filters=None):
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
            assert len(filters) >= num_blocks
        layers = []
        in_channels = 1
        for i in range(num_blocks):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels, filters[i], kernel_size, padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(filters[i]),
                    nn.ReLU(),
                    nn.AvgPool1d(2),
                ]
            )
            in_channels = filters[i]
        self.features = nn.Sequential(*layers)
        dummy_out = self.features(torch.randn(1, 1, input_length))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dummy_out.numel(), 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 256),
        )

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


# --- Architecture 2: ResNet-style CNN ---
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, ks, padding=ks // 2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class ResNet_SCA_CNN(nn.Module):
    def __init__(self, input_length=700, num_blocks=4, kernel_size=11, filters=None):
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        layers = []
        in_channels = 1
        for i in range(num_blocks):
            layers.extend(
                [ResidualBlock(in_channels, filters[i], kernel_size), nn.AvgPool1d(2)]
            )
            in_channels = filters[i]
        self.features = nn.Sequential(*layers)
        dummy_out = self.features(torch.randn(1, 1, input_length))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dummy_out.numel(), 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 256),
        )

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def setup_output_directory(base_dir, config):
    arch, bs, lr = (
        config["architecture"],
        config["batch_size"],
        config.get("learning_rate", "default"),
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir_name = f"CNN_{config['dataset']}_{arch}_bs{bs}_lr{lr}_{timestamp}"
    output_path = Path(base_dir) / dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "summary.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return output_path


def load_and_preprocess_data(dataset_type, batch_size):
    path = DATASET_PATHS[dataset_type]
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        X_prof_raw, y_prof = (
            f["Profiling_traces/traces"][:],
            f["Profiling_traces/labels"][:],
        )
        X_atk_raw, y_atk = f["Attack_traces/traces"][:], f["Attack_traces/labels"][:]
        pt_atk, key_atk = f["Attack_traces/metadata"]["plaintext"][:].astype(
            np.uint8
        ), int(f["Attack_traces/metadata"]["key"][0][TARGET_KEY_BYTE_INDEX])
    mean, std = X_prof_raw.mean(), X_prof_raw.std()
    X_prof_norm, X_atk_norm = (X_prof_raw - mean) / std, (X_atk_raw - mean) / std
    train_ds, val_ds = TensorDataset(
        torch.from_numpy(X_prof_norm).float(), torch.from_numpy(y_prof).long()
    ), TensorDataset(
        torch.from_numpy(X_atk_norm).float(), torch.from_numpy(y_atk).long()
    )
    train_loader, val_loader = DataLoader(
        train_ds, batch_size, shuffle=True, pin_memory=True, num_workers=2
    ), DataLoader(val_ds, batch_size, shuffle=False, pin_memory=True, num_workers=2)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X_atk": X_atk_norm,
        "pt_atk": pt_atk,
        "key_atk": key_atk,
        "trace_length": X_prof_raw.shape[1],
    }


def compute_key_rank(model_probs, plaintexts, true_key, n_traces):
    if n_traces == 0 or n_traces > model_probs.shape[0]:
        return 255, np.zeros(256)
    probs = np.log(np.maximum(model_probs[:n_traces], 1e-40))
    key_scores = np.zeros(256)
    for k_guess in range(256):
        sbox_out = SBOX[plaintexts[:n_traces, TARGET_KEY_BYTE_INDEX] ^ k_guess]
        key_scores[k_guess] = np.sum(probs[np.arange(n_traces), sbox_out])
    ranked_keys = np.argsort(key_scores)[::-1]
    return np.where(ranked_keys == true_key)[0][0], key_scores


# ==============================================================================
# 4. EXPERIMENT EXECUTION
# ==============================================================================
def run_cnn_experiment(config, output_dir, device):
    logging.info(f"--- Starting Experiment: {config} ---")
    data = load_and_preprocess_data(config["dataset"], config["batch_size"])
    if data is None:
        return None

    arch_type = config.get("architecture", "standard")
    if arch_type == "resnet":
        model = ResNet_SCA_CNN(
            input_length=data["trace_length"], **config["model_params"]
        ).to(device)
    else:
        model = Standard_SCA_CNN(
            input_length=data["trace_length"], **config["model_params"]
        ).to(device)
    logging.info(f"Using architecture: {arch_type}")

    optimizer_type = config.get("optimizer", "rmsprop")
    if optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
        )
    else:
        optimizer = optim.RMSprop(
            model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
        )
    logging.info(f"Using optimizer: {optimizer_type}")
    criterion = nn.CrossEntropyLoss()

    train_start_time = time.time()
    train_losses, val_losses = [], []
    for epoch in range(config["epochs"]):
        model.train()
        epoch_train_loss = 0.0
        for Xb, yb in data["train_loader"]:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(data["train_loader"]))
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for Xv, yv in data["val_loader"]:
                Xv, yv = Xv.to(device), yv.to(device)
                epoch_val_loss += criterion(model(Xv), yv).item()
        val_losses.append(epoch_val_loss / len(data["val_loader"]))
        if (epoch + 1) == 1 or (epoch + 1) == config["epochs"]:
            logging.info(
                f"--> Epoch {epoch+1}/{config['epochs']} -> Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}"
            )
    training_time = time.time() - train_start_time
    logging.info(f"Model training finished in {training_time:.2f} seconds.")

    # --- Save Loss Curve Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Training & Validation Loss\n{config['dataset']} arch={arch_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curves.png")
    plt.close()

    logging.info("Evaluating model on attack set...")
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(data["X_atk"]), config["batch_size"]):
            batch = (
                torch.from_numpy(data["X_atk"][i : i + config["batch_size"]])
                .float()
                .to(device)
            )
            all_probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    attack_probs = np.concatenate(all_probs, axis=0)

    ranks, traces_to_rank_0 = [], -1
    max_eval_traces = min(PLOT_MAX_TRACES, len(data["X_atk"]))
    trace_counts = range(0, max_eval_traces + 1, PLOT_STEP)
    for n in trace_counts:
        rank, _ = compute_key_rank(attack_probs, data["pt_atk"], data["key_atk"], n)
        if rank == 0 and traces_to_rank_0 == -1:
            traces_to_rank_0 = n
        ranks.append(rank)
    if traces_to_rank_0 != -1:
        logging.info(f"Success! Key Rank 0 achieved at {traces_to_rank_0} traces.")

    plt.figure(figsize=(10, 6))
    plt.plot(trace_counts, ranks, marker="o")
    plt.title(f"Key Rank vs. Traces\n{config['dataset']} arch={arch_type}")
    plt.xlabel("Num Traces")
    plt.ylabel("Key Rank")
    plt.grid(True)
    plt.ylim(-5, 260)
    plt.savefig(output_dir / "key_rank_vs_traces.png")
    plt.close()

    final_rank, final_scores = compute_key_rank(
        attack_probs, data["pt_atk"], data["key_atk"], max_eval_traces
    )
    predicted_key = np.argmax(final_scores)

    # --- Save Key Score Distribution Plot ---
    plt.figure(figsize=(12, 5))
    plt.bar(range(256), final_scores, color="purple")
    plt.axvline(predicted_key, color="g", ls="--", label=f"Predicted: {predicted_key}")
    plt.axvline(data["key_atk"], color="r", ls=":", label=f"True: {data['key_atk']}")
    plt.title(f"Key Scores after {max_eval_traces} Traces")
    plt.legend()
    plt.savefig(output_dir / "key_score_distribution.png")
    plt.close()

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": config["dataset"],
        "architecture": arch_type,
        "optimizer": optimizer_type,
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        **{
            f"param_{k}": str(v) if isinstance(v, list) else v
            for k, v in config["model_params"].items()
        },
        "training_time_s": round(training_time, 2),
        "traces_to_rank_0": traces_to_rank_0,
        "final_rank": final_rank,
        "predicted_key": predicted_key,
        "true_key": data["key_atk"],
        "attack_successful": bool(
            predicted_key == data["key_atk"] and traces_to_rank_0 != -1
        ),
    }


# ==============================================================================
# 5. MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_output_directory = "cnn_sca_experiments_output"
    csv_path = Path(base_output_directory) / "master_cnn_results.csv"
    all_keys = set(
        [
            "timestamp",
            "dataset",
            "architecture",
            "optimizer",
            "epochs",
            "batch_size",
            "learning_rate",
            "training_time_s",
            "traces_to_rank_0",
            "final_rank",
            "predicted_key",
            "true_key",
            "attack_successful",
        ]
    )
    for exp in EXPERIMENTS_TO_RUN:
        for p_key in exp["model_params"].keys():
            all_keys.add(f"param_{p_key}")
    csv_headers = sorted(list(all_keys))
    Path(base_output_directory).mkdir(exist_ok=True)
    file_exists = csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for i, experiment_config in enumerate(EXPERIMENTS_TO_RUN):
            print(
                f"\n{'='*20} RUNNING EXPERIMENT {i+1}/{len(EXPERIMENTS_TO_RUN)} {'='*20}"
            )
            output_path = setup_output_directory(
                base_output_directory, experiment_config
            )
            try:
                result_data = run_cnn_experiment(experiment_config, output_path, device)
                if result_data:
                    writer.writerow(result_data)
                    f.flush()
            except Exception as e:
                logging.error(
                    f"FATAL ERROR in experiment: {experiment_config}", exc_info=True
                )
