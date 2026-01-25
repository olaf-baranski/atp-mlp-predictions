import argparse
from pathlib import Path
import pandas as pd


# =========================
# Pomocnicze
# =========================
def norm_name(s: str) -> str:
    """Normalizacja nazw do dopasowań (małe litery, pojedyncze spacje)."""
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    return s


def build_player_map(initial_input_csv: Path) -> dict:
    """
    Buduje mapę: name_norm -> atrybuty gracza (rank/points/age/height/seed/id)
    na podstawie examples/input_matches.csv (R1).
    Dzięki temu NIE musisz uzupełniać ręcznie danych graczy w kolejnych rundach.
    """
    df = pd.read_csv(initial_input_csv)

    # Kolumny, które przechowujemy dla gracza
    fields = ["rank", "rank_points", "age", "height", "seed", "id"]

    pmap = {}

    def add_player(side: str, row):
        name = str(row.get(f"{side}_name", "")).strip()
        if not name:
            return
        key = norm_name(name)

        rec = {
            "name": name,  # oryginalna nazwa do wypisywania
            "rank": row.get(f"{side}_rank", ""),
            "rank_points": row.get(f"{side}_rank_points", ""),
            "age": row.get(f"{side}_age", ""),
            "height": row.get(f"{side}_height", ""),
            "seed": row.get(f"{side}_seed", ""),
            "id": row.get(f"{side}_id", "") if f"{side}_id" in df.columns else "",
        }

        # Jeśli gracz już jest w mapie, zostawiamy pierwsze niepuste wartości
        if key in pmap:
            old = pmap[key]
            for f in fields:
                if (old.get(f, "") in ["", None]) and (rec.get(f, "") not in ["", None]):
                    old[f] = rec[f]
        else:
            pmap[key] = rec

    for _, r in df.iterrows():
        add_player("A", r)
        add_player("B", r)

    return pmap


def pick_winner(row, winners_col: str, fallback_col: str) -> str:
    """
    Zwraca nazwę zwycięzcy:
    - jeśli winners_col jest wypełnione -> to bierzemy
    - inaczej fallback_col (np. pred_winner)
    """
    w = str(row.get(winners_col, "")).strip() if winners_col in row else ""
    if w:
        return w
    fb = str(row.get(fallback_col, "")).strip() if fallback_col in row else ""
    return fb


def main() -> None:
    ap = argparse.ArgumentParser(description="Buduje input do kolejnej rundy na podstawie faktycznych zwycięzców.")
    ap.add_argument("--prev_csv", type=str, required=True, help="Plik z poprzedniej rundy (np. predictions_r1.csv) z kolumną actual_winner.")
    ap.add_argument("--initial_input", type=str, default="examples/input_matches.csv", help="R1 input z pełnymi danymi graczy.")
    ap.add_argument("--out_csv", type=str, required=True, help="Wyjściowy input CSV dla kolejnej rundy (np. examples/input_r2.csv).")
    ap.add_argument("--winners_col", type=str, default="actual_winner", help="Kolumna z prawdziwym zwycięzcą (domyślnie actual_winner).")
    ap.add_argument("--fallback_col", type=str, default="pred_winner", help="Fallback gdy winners_col puste (domyślnie pred_winner).")
    args = ap.parse_args()

    prev_csv = Path(args.prev_csv)
    initial_input = Path(args.initial_input)
    out_csv = Path(args.out_csv)

    if not prev_csv.exists():
        raise FileNotFoundError(f"Brak prev_csv: {prev_csv}")
    if not initial_input.exists():
        raise FileNotFoundError(f"Brak initial_input: {initial_input}")

    prev = pd.read_csv(prev_csv)
    pmap = build_player_map(initial_input)

    # Stałe meczowe bierzemy z poprzedniej rundy (powinny być w pliku)
    surface = str(prev.iloc[0].get("surface", "Hard"))
    tourney_level = str(prev.iloc[0].get("tourney_level", "G"))
    best_of = prev.iloc[0].get("best_of", 5)

    # Zbieramy zwycięzców w KOLEJNOŚCI meczu
    winners = []
    missing_players = []

    for _, r in prev.iterrows():
        w = pick_winner(r, args.winners_col, args.fallback_col)
        if not w:
            raise RuntimeError(
                f"Nie mogę wyznaczyć zwycięzcy (puste {args.winners_col} i {args.fallback_col}) w wierszu: {r.to_dict()}"
            )
        winners.append(w)

    if len(winners) % 2 != 0:
        raise RuntimeError("Nieparzysta liczba zwycięzców — sprawdź kolejność / kompletność danych.")

    # Budujemy mecze kolejnej rundy: (winner1 vs winner2), (winner3 vs winner4), ...
    rows = []
    for i in range(0, len(winners), 2):
        A = winners[i]
        B = winners[i + 1]

        Arec = pmap.get(norm_name(A))
        Brec = pmap.get(norm_name(B))

        if Arec is None:
            missing_players.append(A)
            Arec = {"name": A, "rank": "", "rank_points": "", "age": "", "height": "", "seed": "", "id": ""}
        if Brec is None:
            missing_players.append(B)
            Brec = {"name": B, "rank": "", "rank_points": "", "age": "", "height": "", "seed": "", "id": ""}

        rows.append(
            {
                "A_name": Arec["name"],
                "B_name": Brec["name"],
                "A_rank": Arec["rank"],
                "B_rank": Brec["rank"],
                "A_rank_points": Arec["rank_points"],
                "B_rank_points": Brec["rank_points"],
                "A_age": Arec["age"],
                "B_age": Brec["age"],
                "A_height": Arec["height"],
                "B_height": Brec["height"],
                "A_seed": Arec["seed"],
                "B_seed": Brec["seed"],
                "surface": surface,
                "tourney_level": tourney_level,
                "best_of": best_of,
                "A_id": Arec.get("id", ""),
                "B_id": Brec.get("id", ""),
            }
        )

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")

    # Dodatkowo: szybki raport, ile predykcji było trafionych (jeśli masz actual_winner)
    if args.winners_col in prev.columns and "pred_winner" in prev.columns:
        mask = prev[args.winners_col].astype(str).str.strip() != ""
        if mask.any():
            acc = (prev.loc[mask, "pred_winner"].astype(str).str.strip() == prev.loc[mask, args.winners_col].astype(str).str.strip()).mean()
            print(f"INFO: Accuracy poprzedniej rundy (tylko tam gdzie wpisano {args.winners_col}): {acc:.4f}")

    if missing_players:
        uniq = sorted(set(missing_players))
        print("UWAGA: Nie znalazłem tych graczy w initial_input (zostawiłem puste pola liczbowe):")
        for n in uniq:
            print(" -", n)

    print("OK. Zapisano:", out_csv)


if __name__ == "__main__":
    main()
