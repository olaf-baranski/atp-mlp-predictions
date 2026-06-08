import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Kolumny potrzebne do cech "przedmeczowych" + do Elo (id + match_num do sortowania)
NEEDED_COLS = [
    "tourney_date",
    "match_num",
    "surface",
    "tourney_level",
    "best_of",
    "winner_id",
    "loser_id",
    "winner_name",
    "loser_name",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "winner_age",
    "loser_age",
    "winner_ht",
    "loser_ht",
    "winner_seed",
    "loser_seed",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in NEEDED_COLS if c in cols]
    df = pd.read_csv(path, usecols=usecols)
    return df


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _elo_expected(r_a: float, r_b: float) -> float:
    """Prawdopodobieństwo wygranej A w klasycznym Elo."""
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / 400.0))


def add_elo_columns(matches: pd.DataFrame, k: float = 32.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dodaje do matches kolumny Elo sprzed meczu:
    - winner_elo, loser_elo (global)
    - winner_surface_elo, loser_surface_elo (osobne Elo per nawierzchnia)

    WAŻNE: iterujemy po meczach w porządku czasowym, bez przecieku.
    """
    df = matches.copy()

    # Upewniamy się, że pola do sortowania istnieją
    if "match_num" not in df.columns:
        df["match_num"] = np.nan

    # Konwersje numeryczne do sortowania
    df["tourney_date"] = df["tourney_date"].astype(str)
    df["year"] = _to_numeric(df["tourney_date"].str.slice(0, 4))
    df["match_num"] = _to_numeric(df["match_num"])

    # Nawierzchnia jako string
    df["surface"] = df["surface"].fillna("Unknown").astype(str)

    # Sort (przybliżony porządek): data + match_num
    df = df.sort_values(["tourney_date", "match_num"], kind="mergesort").reset_index(drop=True)

    # Ratingi
    base = 1500.0
    elo_global: Dict[str, float] = {}
    elo_surface: Dict[Tuple[str, str], float] = {}

    w_elo = []
    l_elo = []
    w_selo = []
    l_selo = []

    id_to_name: Dict[str, str] = {}

    for _, row in df.iterrows():
        wid = str(row.get("winner_id", ""))
        lid = str(row.get("loser_id", ""))
        surf = str(row.get("surface", "Unknown"))

        # zapamiętaj ostatnio widzianą nazwę dla id (ułatwia predykcje po nazwie)
        wname = str(row.get("winner_name", ""))
        lname = str(row.get("loser_name", ""))
        if wid:
            id_to_name[wid] = wname
        if lid:
            id_to_name[lid] = lname

        # global
        r_w = elo_global.get(wid, base)
        r_l = elo_global.get(lid, base)

        # surface
        r_ws = elo_surface.get((surf, wid), base)
        r_ls = elo_surface.get((surf, lid), base)

        # zapis "przed meczem"
        w_elo.append(r_w)
        l_elo.append(r_l)
        w_selo.append(r_ws)
        l_selo.append(r_ls)

        # update global
        p_w = _elo_expected(r_w, r_l)
        elo_global[wid] = r_w + k * (1.0 - p_w)
        elo_global[lid] = r_l + k * (0.0 - (1.0 - p_w))

        # update surface
        p_ws = _elo_expected(r_ws, r_ls)
        elo_surface[(surf, wid)] = r_ws + k * (1.0 - p_ws)
        elo_surface[(surf, lid)] = r_ls + k * (0.0 - (1.0 - p_ws))

    df["winner_elo"] = np.array(w_elo, dtype=np.float32)
    df["loser_elo"] = np.array(l_elo, dtype=np.float32)
    df["winner_surface_elo"] = np.array(w_selo, dtype=np.float32)
    df["loser_surface_elo"] = np.array(l_selo, dtype=np.float32)

    # Snapshot Elo po ostatnim meczu w df (czyli "stan na koniec end_year")
    all_ids = sorted(elo_global.keys())
    surfaces = ["Hard", "Clay", "Grass", "Carpet"]

    rows = []
    for pid in all_ids:
        row = {
            "player_id": pid,
            "player_name": id_to_name.get(pid, ""),
            "elo_global": float(elo_global.get(pid, base)),
        }
        for s in surfaces:
            row[f"elo_{s.lower()}"] = float(elo_surface.get((s, pid), base))
        rows.append(row)

    snapshot_df = pd.DataFrame(rows)

    return df, snapshot_df


def build_examples_from_matches(df: pd.DataFrame, rng: np.random.RandomState, use_elo: bool, elo_mode: str = "both") -> pd.DataFrame:
    # Jeśli czegoś brakuje w danym roku, tworzymy kolumnę z NaN
    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Konwersje liczbowe
    num_cols = [
        "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points",
        "winner_age", "loser_age",
        "winner_ht", "loser_ht",
        "winner_seed", "loser_seed",
        "best_of",
    ]
    for c in num_cols:
        df[c] = _to_numeric(df[c])

    # swap=True => A=winner => y=1
    swap = rng.rand(len(df)) < 0.5

    def pick(a_w, a_l):
        return np.where(swap, df[a_w].to_numpy(), df[a_l].to_numpy())

    def pick_name(n_w, n_l):
        return np.where(swap, df[n_w].astype(str).to_numpy(), df[n_l].astype(str).to_numpy())

    A_rank = pick("winner_rank", "loser_rank")
    B_rank = pick("loser_rank", "winner_rank")

    A_points = pick("winner_rank_points", "loser_rank_points")
    B_points = pick("loser_rank_points", "winner_rank_points")

    A_age = pick("winner_age", "loser_age")
    B_age = pick("loser_age", "winner_age")

    A_ht = pick("winner_ht", "loser_ht")
    B_ht = pick("loser_ht", "winner_ht")

    A_seed = pick("winner_seed", "loser_seed")
    B_seed = pick("loser_seed", "winner_seed")

    out = pd.DataFrame(
        {
            "tourney_date": df["tourney_date"],
            "year": _to_numeric(df["tourney_date"].astype(str).str.slice(0, 4)),
            "A_name": pick_name("winner_name", "loser_name"),
            "B_name": pick_name("loser_name", "winner_name"),
            "surface": df["surface"].fillna("Unknown").astype(str),
            "tourney_level": df["tourney_level"].fillna("Unknown").astype(str),
            "best_of": df["best_of"],

            "rank_diff": A_rank - B_rank,
            "rank_points_diff": A_points - B_points,
            "age_diff": A_age - B_age,
            "height_diff": A_ht - B_ht,
            "seed_diff": A_seed - B_seed,

            "y": swap.astype(int),
        }
    )

    if use_elo:
        elo_mode = (elo_mode or "both").lower()

        if elo_mode in ("both", "global"):
            A_elo = pick("winner_elo", "loser_elo")
            B_elo = pick("loser_elo", "winner_elo")
            out["elo_diff"] = A_elo - B_elo

        if elo_mode in ("both", "surface"):
            A_selo = pick("winner_surface_elo", "loser_surface_elo")
            B_selo = pick("loser_surface_elo", "winner_surface_elo")
            out["surface_elo_diff"] = A_selo - B_selo

    return out


def make_splits(examples: pd.DataFrame, train_end: int, val_year: int, test_year: int):
    train = examples[examples["year"].between(2010, train_end)].copy()
    val = examples[examples["year"] == val_year].copy()
    test = examples[examples["year"] == test_year].copy()
    return train, val, test


def fit_preprocessor(train_df: pd.DataFrame, use_elo: bool, elo_mode: str = "both") -> Tuple[ColumnTransformer, list]:
    numeric_features = ["rank_diff", "rank_points_diff", "age_diff", "height_diff", "seed_diff", "best_of"]
    if use_elo:
        elo_mode = (elo_mode or "both").lower()
        if elo_mode in ("both", "global"):
            numeric_features += ["elo_diff"]
        if elo_mode in ("both", "surface"):
            numeric_features += ["surface_elo_diff"]

    categorical_features = ["surface", "tourney_level"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    preprocessor.fit(train_df)

    # Feature names po transformacji
    num_names = numeric_features
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    if hasattr(cat_encoder, "get_feature_names_out"):
        cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    else:
        cat_names = cat_encoder.get_feature_names(categorical_features).tolist()

    feature_names = num_names + cat_names
    return preprocessor, feature_names


def transform_and_save(preprocessor: ColumnTransformer, df: pd.DataFrame, out_npz: Path, out_meta_csv: Path) -> None:
    y = df["y"].to_numpy().astype(np.int64)
    X = preprocessor.transform(df).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta_csv.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_npz, X=X, y=y)

    meta_cols = ["tourney_date", "year", "A_name", "B_name", "surface", "tourney_level", "best_of", "y"]
    df[meta_cols].to_csv(out_meta_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Przygotowanie danych (features + split) z atp_matches_YYYY.csv")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--train_end", type=int, default=2022)
    parser.add_argument("--val_year", type=int, default=2023)
    parser.add_argument("--test_year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=42)

    # Opcje dodatkowe
    parser.add_argument("--surface_filter", type=str, default="", help="Np. 'Hard' -> tylko ta nawierzchnia")
    parser.add_argument("--use_elo", action="store_true", help="Dodaj cechy Elo (global + surface)")
    parser.add_argument("--elo_k", type=float, default=32.0, help="Parametr K w Elo (szybkość reakcji)")
    parser.add_argument("--elo_mode",type=str,default="both",choices=["both", "global", "surface"],help="Jakie cechy Elo dodać: both=global+surface, global=tylko elo_diff, surface=tylko surface_elo_diff",)
    parser.add_argument("--final_fit", action="store_true", help="Przygotuj dane do final fit (train=2010–end_year, bez val/test)")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    models_dir = Path(args.models_dir)
    
    if args.final_fit:
        out_dir = out_dir / "final"

    rng = np.random.RandomState(args.seed)

    dfs = []
    for year in range(args.start_year, args.end_year + 1):
        path = raw_dir / "atp_matches_{}.csv".format(year)
        if not path.exists():
            raise FileNotFoundError("Brak pliku: {}. Najpierw uruchom download_data.py".format(path))
        dfs.append(_safe_read_csv(path))

    matches = pd.concat(dfs, ignore_index=True)

    # Elo (musi być policzone PRZED budową examples)
    if args.use_elo:
        matches, elo_snapshot = add_elo_columns(matches, k=float(args.elo_k))
        print("Dodano Elo: K={}".format(args.elo_k))

        snap_path = models_dir / f"elo_snapshot_{args.end_year}.csv"
        elo_snapshot.to_csv(snap_path, index=False)
        print("Zapisano snapshot Elo:", snap_path)

    examples = build_examples_from_matches(matches, rng=rng, use_elo=args.use_elo, elo_mode=args.elo_mode)

    # Opcjonalny filtr po nawierzchni na etapie examples
    if args.surface_filter:
        sf = args.surface_filter.strip().lower()
        examples = examples[examples["surface"].astype(str).str.lower() == sf].copy()
        print("Użyto surface_filter='{}' -> N={}".format(args.surface_filter, len(examples)))

    if args.final_fit:
        # Final fit: bierzemy wszystko do końca end_year
        train_df = examples[examples["year"].between(args.start_year, args.end_year)].copy()
        val_df = examples.iloc[0:0].copy()
        test_df = examples.iloc[0:0].copy()
    else:
        train_df, val_df, test_df = make_splits(
            examples,
            train_end=args.train_end,
            val_year=args.val_year,
            test_year=args.test_year,
        )

    if len(train_df) == 0:
        raise RuntimeError("Train jest pusty. Sprawdź years / dane.")
    if (not args.final_fit) and (len(val_df) == 0 or len(test_df) == 0):
        raise RuntimeError("Jeden ze splitów jest pusty. Sprawdź years / dane.")

    preprocessor, feature_names = fit_preprocessor(train_df, use_elo=args.use_elo, elo_mode=args.elo_mode)

    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_name = "preprocessor_final.joblib" if args.final_fit else "preprocessor.joblib"
    dump(preprocessor, models_dir / prep_name)

    feat_name = "feature_names_final.json" if args.final_fit else "feature_names.json"
    with open(out_dir / feat_name, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    transform_and_save(preprocessor, train_df, out_dir / "train.npz", out_dir / "train_meta.csv")
    if not args.final_fit:
        transform_and_save(preprocessor, val_df, out_dir / "val.npz", out_dir / "val_meta.csv")
        transform_and_save(preprocessor, test_df, out_dir / "test.npz", out_dir / "test_meta.csv")

    print("OK. Zapisano:")
    print("- {}".format(out_dir / "train.npz"))
    print("- {}".format(out_dir / "val.npz"))
    print("- {}".format(out_dir / "test.npz"))
    print("- {}".format(out_dir / "feature_names.json"))
    print("- {}".format(models_dir / "preprocessor.joblib"))


if __name__ == "__main__":
    main()
