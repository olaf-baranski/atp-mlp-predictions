import pandas as pd
from pathlib import Path

FILES = [
    ("R1", "examples/RG2026/predictions_r1.csv"),
    ("R2", "examples/RG2026/predictions_r2.csv"),
    ("R3", "examples/RG2026/predictions_r3.csv"),
    ("R4", "examples/RG2026/predictions_r4.csv"),
    ("R5", "examples/RG2026/predictions_r5.csv"),
    ("R6", "examples/RG2026/predictions_r6.csv"),
    ("R7", "examples/RG2026/predictions_r7.csv"),
]

def pick_actual_col(df):
    for c in ["actual_winner"]:
        if c in df.columns:
            return c
    return None

rows = []
total_correct = 0
total_matches = 0

for rnd, path in FILES:
    p = Path(path)
    if not p.exists():
        continue

    df = pd.read_csv(p)

    actual_col = pick_actual_col(df)
    if actual_col is None:
        raise RuntimeError(f"{path}: brak kolumny actual_winner")

    sub = df.dropna(subset=[actual_col]).copy()
    sub = sub[sub[actual_col].astype(str).str.strip() != ""]

    correct = int(
        (
            sub["pred_winner"].astype(str).str.strip()
            == sub[actual_col].astype(str).str.strip()
        ).sum()
    )
    total = len(sub)
    acc = correct / total if total else 0

    total_correct += correct
    total_matches += total

    rows.append([rnd, correct, total, round(acc * 100, 2)])

out = pd.DataFrame(rows, columns=["round", "correct", "total", "accuracy_pct"])

overall_acc = total_correct / total_matches if total_matches else 0
summary = pd.DataFrame(
    [["TOTAL", total_correct, total_matches, round(overall_acc * 100, 2)]],
    columns=["round", "correct", "total", "accuracy_pct"],
)

final = pd.concat([out, summary], ignore_index=True)

print(final.to_string(index=False))

Path("reports").mkdir(exist_ok=True)
final.to_csv("reports/rg2026_round_accuracy.csv", index=False)
print("\nOK. Zapisano: reports/rg2026_round_accuracy.csv")