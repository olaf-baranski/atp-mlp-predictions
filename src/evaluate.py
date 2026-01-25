import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ===== Model (musi być identyczny jak w train.py) =====
class MLP(nn.Module):
    def __init__(self, input_dim: int, h1: int = 64, h2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # [batch]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(npz_path))
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)  # BCE expects float targets
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


@torch.no_grad()
def eval_split(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    all_probs = []
    all_y = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += float(loss.item()) * X.size(0)
        total += X.size(0)
        correct += int((preds == y).sum().item())

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    probs_np = np.concatenate(all_probs, axis=0)
    y_np = np.concatenate(all_y, axis=0)

    # Log-loss (binary cross-entropy) liczony w numpy (dla raportowania)
    eps = 1e-12
    probs_clip = np.clip(probs_np, eps, 1 - eps)
    logloss = float(-np.mean(y_np * np.log(probs_clip) + (1 - y_np) * np.log(1 - probs_clip)))

    # Brier score
    brier = float(np.mean((probs_np - y_np) ** 2))

    return {"bce_loss": float(avg_loss), "accuracy": float(acc), "logloss": logloss, "brier": brier}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ewaluacja zapisanej sieci MLP na val/test")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="mlp_model.pt")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--save_json", action="store_true", help="Zapisz metryki do models/eval_metrics.json")
    parser.add_argument("--surface", type=str, default="", help="Jeśli ustawisz np. 'Hard', ewaluacja tylko na tej nawierzchni (wg *_meta.csv)")
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    model_path = models_dir / args.model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Brak modelu: {model_path}. Najpierw uruchom train.py")

    # Wczytaj checkpoint
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    input_dim = int(ckpt["input_dim"])
    h1 = int(ckpt["h1"])
    h2 = int(ckpt["h2"])
    dropout = float(ckpt["dropout"])

    model = MLP(input_dim=input_dim, h1=h1, h2=h2, dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # Wczytaj dane
    val_npz = data_dir / "val.npz"
    test_npz = data_dir / "test.npz"
    if not val_npz.exists() or not test_npz.exists():
        raise FileNotFoundError("Brak val.npz/test.npz. Najpierw uruchom prepare_data.py")

    # Wczytaj meta, żeby ewentualnie filtrować po nawierzchni
    val_meta = data_dir / "val_meta.csv"
    test_meta = data_dir / "test_meta.csv"
    if not val_meta.exists() or not test_meta.exists():
        raise FileNotFoundError("Brak val_meta.csv/test_meta.csv. One powstają w prepare_data.py")
    
    meta_val = pd.read_csv(val_meta)
    meta_test = pd.read_csv(test_meta)

    surface_val = meta_val["surface"].astype(str).to_numpy()
    surface_test = meta_test["surface"].astype(str).to_numpy()


    X_val, y_val = load_npz(val_npz)
    X_test, y_test = load_npz(test_npz)

    if args.surface:
        mask_val = (surface_val == args.surface)
        mask_test = (surface_test == args.surface)

        X_val = X_val[mask_val]
        y_val = y_val[mask_val]
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]

        print("Filtr surface='{}' -> val N={}, test N={}".format(args.surface, X_val.shape[0], X_test.shape[0]))

    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size)
    test_loader = make_loader(X_test, y_test, batch_size=args.batch_size)

    val_metrics = eval_split(model, val_loader, device)
    test_metrics = eval_split(model, test_loader, device)

    metrics = {"val": val_metrics, "test": test_metrics}

    print("\nVAL  :", val_metrics)
    print("TEST :", test_metrics)

    if args.save_json:
        out_path = models_dir / "eval_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("\nOK. Zapisano:", out_path)


if __name__ == "__main__":
    main()
