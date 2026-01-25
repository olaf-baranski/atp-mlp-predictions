import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Dict, Any



# =========================
# Ustawienia reproducibility
# =========================
def set_seed(seed: int) -> None:
    """Ustawia ziarno losowości dla numpy/torch (reproducible run)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministyczność (koszt: czasem wolniej)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Wybiera device: CUDA jeśli dostępne, inaczej CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Model MLP
# =========================
class MLP(nn.Module):
    """
    Prosta sieć MLP do klasyfikacji binarnej.

    UWAGA:
    - Model zwraca LOGITY (bez sigmoid na końcu).
    - Do loss używamy BCEWithLogitsLoss (stabilniejsze numerycznie).
    """

    def __init__(self, input_dim: int, h1: int = 64, h2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),  # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # shape: [batch]


# =========================
# Metryki
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Liczy loss (BCE) i accuracy na podanym loaderze."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * X.size(0)
        total += X.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += int((preds == y).sum().item())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    return {"loss": avg_loss, "accuracy": acc}


# =========================
# Ładowanie danych
# =========================
def load_npz(npz_path: Path) -> tuple:
    """
    Wczytuje X,y z pliku .npz (z prepare_data.py).
    X: float32, shape [N, D]
    y: int64 lub podobne, shape [N]
    """
    data = np.load(str(npz_path))
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)  # BCE expects float targets 0/1
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Tworzy DataLoader na bazie TensorDataset."""
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# =========================
# Trening
# =========================
def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> tuple:
    """
    Trenuje model i stosuje early stopping po val_loss.
    Zwraca: (best_state_dict, history)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    history = {"train_loss": [], "train_acc": []}
    if val_loader is not None:
        history.update({"val_loss": [], "val_acc": []})

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total = 0
        correct = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * X.size(0)
            total += X.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += int((preds == y).sum().item())

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                "Epoch {}/{} | train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f}".format(
                    epoch, epochs, train_loss, train_acc, val_loss, val_acc
                )
            )

            # Early stopping (tylko gdy jest walidacja)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping: brak poprawy val_loss przez {} epok.".format(patience))
                    break
        else:
            # Final fit (bez walidacji)
            print(
                "Epoch {}/{} | train loss {:.4f} acc {:.4f}".format(
                    epoch, epochs, train_loss, train_acc
                )
            )


    return best_state, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening MLP (PyTorch) na danych z data/processed/*.npz")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--models_dir", type=str, default="models")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--h1", type=int, default=64)
    parser.add_argument("--h2", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_npz = data_dir / "train.npz"
    val_npz = data_dir / "val.npz"

    if not train_npz.exists():
        raise FileNotFoundError("Brak train.npz. Najpierw uruchom prepare_data.py")

    has_val = val_npz.exists()
    if not has_val:
        print("INFO: Brak val.npz w", data_dir, "-> trening bez walidacji (final fit).")

    X_train, y_train = load_npz(train_npz)
    if has_val:
        X_val, y_val = load_npz(val_npz)
    else:
        X_val, y_val = None, None

    train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False) if has_val else None

    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, h1=args.h1, h2=args.h2, dropout=args.dropout).to(device)

    best_state, history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    if best_state is None:
        # Fallback (gdyby coś poszło bardzo źle)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Zapis najlepszego modelu + konfiguracji
    out_model = models_dir / "mlp_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": int(input_dim),
            "h1": int(args.h1),
            "h2": int(args.h2),
            "dropout": float(args.dropout),
        },
        out_model,
    )

    out_history = models_dir / "train_history.json"
    with open(out_history, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("OK. Zapisano:")
    print("- {}".format(out_model))
    print("- {}".format(out_history))


if __name__ == "__main__":
    main()
