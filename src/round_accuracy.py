import argparse
from pathlib import Path

import pandas as pd


def pick_actual_col(df: pd.DataFrame):
    for c in ["actual_winner", "actual winner", "actualWinner"]:
        if c in df.columns:
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-round prediction accuracy for a tournament.")
    parser.add_argument("--tournament", type=str, required=True, help="Tournament folder name, e.g. RG2026 or WIM2026")
    parser.add_argument("--examples_dir", type=str, default="examples")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    tournament_dir = Path(args.examples_dir) / args.tournament

    files = [
        ("R1", tournament_dir / "predictions_r1.csv"),
        ("R2", tournament_dir / "predictions_r2.csv"),
        ("R3", tournament_dir / "predictions_r3.csv"),
        ("R4", tournament_dir / "predictions_r4.csv"),
        ("R5", tournament_dir / "predictions_r5.csv"),
        ("R6", tournament_dir / "predictions_r6.csv"),
        ("R7", tournament_dir / "predictions_r7.csv"),
    ]

    rows = []
    total_correct = 0
    total_matches = 0

    for rnd, path in files:
        if not path.exists():
            print(f"[SKIP] Brak pliku: {path}")
            continue

        df = pd.read_csv(path)

        if "pred_winner" not in df.columns:
            raise RuntimeError(f"{path}: brak kolumny pred_winner")

        actual_col = pick_actual_col(df)
        if actual_col is None:
            raise RuntimeError(f"{path}: brak kolumny actual_winner / actual winner")

        sub = df.dropna(subset=[actual_col]).copy()
        sub = sub[sub[actual_col].astype(str).str.strip() != ""]
        sub = sub[sub[actual_col].astype(str).str.lower().str.strip() != "nan"]

        correct = int(
            (
                sub["pred_winner"].astype(str).str.strip()
                == sub[actual_col].astype(str).str.strip()
            ).sum()
        )
        total = len(sub)
        accuracy = correct / total if total > 0 else 0.0

        total_correct += correct
        total_matches += total

        rows.append(
            {
                "round": rnd,
                "correct": correct,
                "total": total,
                "accuracy_pct": round(accuracy * 100, 2),
            }
        )

    overall_accuracy = total_correct / total_matches if total_matches > 0 else 0.0
    rows.append(
        {
            "round": "TOTAL",
            "correct": total_correct,
            "total": total_matches,
            "accuracy_pct": round(overall_accuracy * 100, 2),
        }
    )

    out_df = pd.DataFrame(rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = reports_dir / f"{args.tournament.lower()}_round_accuracy.csv"

    out_df.to_csv(out_path, index=False)

    print(out_df.to_string(index=False))
    print()
    print("OK. Zapisano:", out_path)


if __name__ == "__main__":
    main()