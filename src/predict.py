import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from joblib import load as joblib_load


# =========================
# Model (taki sam jak w train.py)
# =========================
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
            nn.Linear(h2, 1),  # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # [batch]


def normalize_name(s: str) -> str:
    """Ujednolica zapis nazwiska do matchowania (małe litery, pojedyncze spacje)."""
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    return s


def surface_col(surface: str) -> str:
    """Mapuje surface -> kolumna Elo w snapshot."""
    s = (surface or "").strip().lower()
    if s == "hard":
        return "elo_hard"
    if s == "clay":
        return "elo_clay"
    if s == "grass":
        return "elo_grass"
    if s == "carpet":
        return "elo_carpet"
    # fallback
    return "elo_hard"


def build_snapshot_maps(snapshot_df: pd.DataFrame) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Buduje mapy:
    - po znormalizowanej nazwie
    - po player_id (string)
    """
    by_name: Dict[str, dict] = {}
    by_id: Dict[str, dict] = {}

    for _, r in snapshot_df.iterrows():
        pid = str(r.get("player_id", "")).strip()
        pname = str(r.get("player_name", "")).strip()
        key = normalize_name(pname)

        row_dict = r.to_dict()
        if key:
            by_name[key] = row_dict
        if pid:
            by_id[pid] = row_dict

    return by_name, by_id


def get_player_elo(
    by_name: Dict[str, dict],
    by_id: Dict[str, dict],
    name: str,
    pid: Optional[str],
    surface: str,
    base: float = 1500.0,
) -> Tuple[float, float]:
    """
    Zwraca (elo_global, elo_surface) dla zawodnika.
    Priorytet: pid jeśli podany -> inaczej match po nazwie.
    Jeśli brak w snapshot: fallback do base.
    """
    row = None
    if pid:
        row = by_id.get(str(pid).strip())
    if row is None:
        row = by_name.get(normalize_name(name))

    if row is None:
        return float(base), float(base)

    eg = row.get("elo_global", base)
    es = row.get(surface_col(surface), base)
    try:
        eg = float(eg)
    except Exception:
        eg = float(base)
    try:
        es = float(es)
    except Exception:
        es = float(base)

    return eg, es


def to_float(x) -> float:
    """Bezpieczna konwersja do float, NaN jeśli puste."""
    if x is None:
        return np.nan
    if isinstance(x, str) and x.strip() == "":
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def build_feature_df(df_in: pd.DataFrame, elo_snapshot_path: Path, base_elo: float = 1500.0) -> pd.DataFrame:
    """
    Z wejściowych kolumn A_*/B_* buduje DataFrame cech zgodny z prepare_data.py:
    - rank_diff, rank_points_diff, age_diff, height_diff, seed_diff
    - elo_diff, surface_elo_diff
    - surface, tourney_level, best_of
    """
    snap = pd.read_csv(elo_snapshot_path)
    by_name, by_id = build_snapshot_maps(snap)

    # Domyślne wartości meczu (AO)
    if "surface" not in df_in.columns:
        df_in["surface"] = "Hard"
    if "tourney_level" not in df_in.columns:
        df_in["tourney_level"] = "G"
    if "best_of" not in df_in.columns:
        df_in["best_of"] = 5

    rows = []
    for _, r in df_in.iterrows():
        surface = str(r.get("surface", "Hard"))
        A_name = str(r.get("A_name", "")).strip()
        B_name = str(r.get("B_name", "")).strip()

        A_id = str(r.get("A_id", "")).strip() if "A_id" in df_in.columns else ""
        B_id = str(r.get("B_id", "")).strip() if "B_id" in df_in.columns else ""
        A_id = A_id if A_id != "" else None
        B_id = B_id if B_id != "" else None

        A_elo_g, A_elo_s = get_player_elo(by_name, by_id, A_name, A_id, surface, base=base_elo)
        B_elo_g, B_elo_s = get_player_elo(by_name, by_id, B_name, B_id, surface, base=base_elo)

        feat = {
            "rank_diff": to_float(r.get("A_rank")) - to_float(r.get("B_rank")),
            "rank_points_diff": to_float(r.get("A_rank_points")) - to_float(r.get("B_rank_points")),
            "age_diff": to_float(r.get("A_age")) - to_float(r.get("B_age")),
            "height_diff": to_float(r.get("A_height")) - to_float(r.get("B_height")),
            "seed_diff": to_float(r.get("A_seed")) - to_float(r.get("B_seed")),
            "elo_diff": float(A_elo_g - B_elo_g),
            "surface_elo_diff": float(A_elo_s - B_elo_s),
            "surface": str(r.get("surface", "Hard")),
            "tourney_level": str(r.get("tourney_level", "G")),
            "best_of": int(to_float(r.get("best_of")) if not np.isnan(to_float(r.get("best_of"))) else 5),
            # pomocniczo (do outputu / debug)
            "A_elo_global": float(A_elo_g),
            "B_elo_global": float(B_elo_g),
            "A_elo_surface": float(A_elo_s),
            "B_elo_surface": float(B_elo_s),
        }
        rows.append(feat)

    return pd.DataFrame(rows)


def load_checkpoint(model_path: Path) -> dict:
    """Ładuje checkpoint torch (z trybem weights_only jeśli dostępne)."""
    try:
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except TypeError:
        # starsze torch bez weights_only
        ckpt = torch.load(str(model_path), map_location="cpu")
    return ckpt


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """Zwraca P(A_wins) dla każdej próbki."""
    model.eval()
    probs_all: List[float] = []

    X_t = torch.from_numpy(X.astype(np.float32))
    n = X_t.size(0)
    for i in range(0, n, batch_size):
        xb = X_t[i : i + batch_size].to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.extend(probs.tolist())

    return np.array(probs_all, dtype=np.float32)


def simulate_bracket_rounds(df_r1: pd.DataFrame, probs: np.ndarray) -> Tuple[pd.DataFrame, str]:
    """
    Prosta symulacja drabinki:
    - zakłada, że mecze R1 są w kolejności drabinki (1 vs 2, 3 vs 4 itd.)
    - zwycięzca meczu = A jeśli P(A_wins) >= 0.5 else B
    - kolejne rundy: winner(mecz1) vs winner(mecz2), itd.
    Zwraca: (tabela wszystkich meczów ze wszystkich rund, champion_name)
    """
    # Zbuduj słownik atrybutów zawodników (żeby przenosić do kolejnych rund)
    def player_from_row(row: pd.Series, side: str) -> dict:
        # side: "A" lub "B"
        return {
            "name": row[f"{side}_name"],
            "rank": row.get(f"{side}_rank", np.nan),
            "rank_points": row.get(f"{side}_rank_points", np.nan),
            "age": row.get(f"{side}_age", np.nan),
            "height": row.get(f"{side}_height", np.nan),
            "seed": row.get(f"{side}_seed", np.nan),
            "id": row.get(f"{side}_id", "") if f"{side}_id" in row.index else "",
        }

    all_rows = []
    current = df_r1.copy()
    current_round = 1

    # przypnij probsy do aktualnej rundy
    current["_pA"] = probs

    while len(current) >= 1:
        # wyniki rundy
        winners = []
        for idx, row in current.iterrows():
            pA = float(row["_pA"])
            A = str(row["A_name"])
            B = str(row["B_name"])
            win_side = "A" if pA >= 0.5 else "B"
            winner = A if win_side == "A" else B
            winner_prob = pA if win_side == "A" else (1.0 - pA)

            out = row.to_dict()
            out["round"] = f"R{current_round}"
            out["pA_wins"] = pA
            out["pred_winner"] = winner
            out["pred_winner_prob"] = float(winner_prob)
            all_rows.append(out)

            # zapamiętaj winnera z atrybutami
            if win_side == "A":
                winners.append(player_from_row(row, "A"))
            else:
                winners.append(player_from_row(row, "B"))

        if len(current) == 1:
            # finał zakończony
            champion = str(all_rows[-1]["pred_winner"])
            return pd.DataFrame(all_rows), champion

        # zbuduj następną rundę
        if len(winners) % 2 != 0:
            raise RuntimeError("Liczba zwycięzców nieparzysta — sprawdź kolejność meczów wejściowych.")

        next_rows = []
        for i in range(0, len(winners), 2):
            p1 = winners[i]
            p2 = winners[i + 1]
            next_rows.append(
                {
                    "A_name": p1["name"],
                    "B_name": p2["name"],
                    "A_rank": p1["rank"],
                    "B_rank": p2["rank"],
                    "A_rank_points": p1["rank_points"],
                    "B_rank_points": p2["rank_points"],
                    "A_age": p1["age"],
                    "B_age": p2["age"],
                    "A_height": p1["height"],
                    "B_height": p2["height"],
                    "A_seed": p1["seed"],
                    "B_seed": p2["seed"],
                    "A_id": p1["id"],
                    "B_id": p2["id"],
                    "surface": current.iloc[0].get("surface", "Hard"),
                    "tourney_level": current.iloc[0].get("tourney_level", "G"),
                    "best_of": current.iloc[0].get("best_of", 5),
                }
            )

        current_round += 1
        current = pd.DataFrame(next_rows)

        # UWAGA: do kolejnej rundy prawdopodobieństwa policzymy później w main() (bo trzeba cechy + model)
        # tutaj tylko kontynuujemy pętlę przez return w main()
        # to jest placeholder, main() nadpisze current i dopnie _pA
        return pd.DataFrame(all_rows), "__NEED_NEXT_ROUND__"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predykcje meczów (AO2026) na podstawie modelu MLP + snapshot Elo.")
    parser.add_argument("--input_csv", type=str, default="examples/input_matches.csv")
    parser.add_argument("--output_csv", type=str, default="examples/predictions.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--model_path", type=str, default="models/mlp_model_final.pt")
    parser.add_argument("--preprocessor_path", type=str, default="models/preprocessor_final_fit.joblib")
    parser.add_argument("--elo_snapshot", type=str, default="models/elo_snapshot_2024.csv")
    parser.add_argument("--simulate", action="store_true", help="Symuluj cały turniej od R1 do finału (kolejność drabinki).")
    parser.add_argument("--device", type=str, default="cpu", help="cpu lub cuda")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    model_path = Path(args.model_path)
    preproc_path = Path(args.preprocessor_path)
    elo_path = Path(args.elo_snapshot)

    if not input_csv.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {input_csv}")

    if not model_path.exists():
        raise FileNotFoundError(f"Brak modelu: {model_path}")

    if not preproc_path.exists():
        raise FileNotFoundError(f"Brak preprocessora: {preproc_path}")

    if not elo_path.exists():
        raise FileNotFoundError(f"Brak snapshot Elo: {elo_path}")

    df_in = pd.read_csv(input_csv)

    # Minimalne wymagane kolumny wejścia (bez Elo)
    required = [
        "A_name", "B_name",
        "A_rank", "B_rank",
        "A_rank_points", "B_rank_points",
        "A_age", "B_age",
        "A_height", "B_height",
        "A_seed", "B_seed",
        "surface", "tourney_level", "best_of",
    ]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise ValueError("Brak wymaganych kolumn w input_csv: " + ", ".join(missing))

    # Device
    dev = torch.device(args.device if args.device in ["cpu", "cuda"] else "cpu")

    # Preprocessor
    preprocessor = joblib_load(str(preproc_path))

    # Model
    ckpt = load_checkpoint(model_path)
    input_dim = int(ckpt.get("input_dim"))
    h1 = int(ckpt.get("h1", 64))
    h2 = int(ckpt.get("h2", 32))
    dropout = float(ckpt.get("dropout", 0.2))

    model = MLP(input_dim=input_dim, h1=h1, h2=h2, dropout=dropout).to(dev)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state", None))
    if state_dict is None:
        raise ValueError("Checkpoint nie zawiera state_dict/model_state.")
    model.load_state_dict(state_dict, strict=True)

    # ====== Predykcje R1 lub “jeden plik” ======
    feat_df = build_feature_df(df_in, elo_snapshot_path=elo_path)
    # kolumny, które rzeczywiście trafiają do preprocessora
    X_raw = feat_df[["rank_diff", "rank_points_diff", "age_diff", "height_diff", "seed_diff",
                    "elo_diff", "surface_elo_diff", "surface", "tourney_level", "best_of"]]

    X = preprocessor.transform(X_raw)
    # sparse -> dense jeśli trzeba
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    probs = predict_probs(model, X, device=dev)

    out = df_in.copy()
    out["pA_wins"] = probs
    out["pred_winner"] = np.where(out["pA_wins"] >= 0.5, out["A_name"], out["B_name"])
    out["pred_winner_prob"] = np.where(out["pA_wins"] >= 0.5, out["pA_wins"], 1.0 - out["pA_wins"])

    # dopnij Elo do outputu (przydatne do raportu/debug)
    out["A_elo_global"] = feat_df["A_elo_global"]
    out["B_elo_global"] = feat_df["B_elo_global"]
    out["A_elo_surface"] = feat_df["A_elo_surface"]
    out["B_elo_surface"] = feat_df["B_elo_surface"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print("OK. Zapisano:", output_csv)

    if not args.simulate:
        return

    # ====== Symulacja rund: pętla do wyłonienia championa ======
    # UWAGA: Zakładamy, że input_csv to R1 w kolejności drabinki (64 mecze dla 128 draw).
    current_df = df_in.copy()
    all_rounds = []
    round_idx = 1

    while True:
        feat_df = build_feature_df(current_df, elo_snapshot_path=elo_path)
        X_raw = feat_df[["rank_diff", "rank_points_diff", "age_diff", "height_diff", "seed_diff",
                        "elo_diff", "surface_elo_diff", "surface", "tourney_level", "best_of"]]
        X = preprocessor.transform(X_raw)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        probs = predict_probs(model, X, device=dev)

        tmp = current_df.copy()
        tmp["pA_wins"] = probs
        tmp["round"] = f"R{round_idx}"
        tmp["pred_winner"] = np.where(tmp["pA_wins"] >= 0.5, tmp["A_name"], tmp["B_name"])
        tmp["pred_winner_prob"] = np.where(tmp["pA_wins"] >= 0.5, tmp["pA_wins"], 1.0 - tmp["pA_wins"])
        all_rounds.append(tmp)

        if len(current_df) == 1:
            champion = str(tmp.iloc[0]["pred_winner"])
            break

        # zbuduj następną rundę z winnerów (kolejność: 1 vs 2, 3 vs 4 itd.)
        winners_info = []
        for i in range(len(tmp)):
            row = tmp.iloc[i]
            if float(row["pA_wins"]) >= 0.5:
                winners_info.append({
                    "name": row["A_name"],
                    "rank": row["A_rank"], "rank_points": row["A_rank_points"],
                    "age": row["A_age"], "height": row["A_height"], "seed": row["A_seed"],
                    "id": row["A_id"] if "A_id" in tmp.columns else "",
                })
            else:
                winners_info.append({
                    "name": row["B_name"],
                    "rank": row["B_rank"], "rank_points": row["B_rank_points"],
                    "age": row["B_age"], "height": row["B_height"], "seed": row["B_seed"],
                    "id": row["B_id"] if "B_id" in tmp.columns else "",
                })

        next_rows = []
        for i in range(0, len(winners_info), 2):
            p1 = winners_info[i]
            p2 = winners_info[i + 1]
            next_rows.append({
                "A_name": p1["name"],
                "B_name": p2["name"],
                "A_rank": p1["rank"], "B_rank": p2["rank"],
                "A_rank_points": p1["rank_points"], "B_rank_points": p2["rank_points"],
                "A_age": p1["age"], "B_age": p2["age"],
                "A_height": p1["height"], "B_height": p2["height"],
                "A_seed": p1["seed"], "B_seed": p2["seed"],
                "A_id": p1["id"], "B_id": p2["id"],
                "surface": current_df.iloc[0].get("surface", "Hard"),
                "tourney_level": current_df.iloc[0].get("tourney_level", "G"),
                "best_of": current_df.iloc[0].get("best_of", 5),
            })

        current_df = pd.DataFrame(next_rows)
        round_idx += 1

    all_df = pd.concat(all_rounds, ignore_index=True)
    out_bracket = output_csv.parent / "predictions_bracket_sim.csv"
    all_df.to_csv(out_bracket, index=False)
    print("OK. Zapisano symulację drabinki:", out_bracket)
    print("CHAMPION (symulacja):", champion)


if __name__ == "__main__":
    main()
