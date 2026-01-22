import argparse
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def _download_file(url: str, out_path: Path, timeout: int = 60) -> None:
    """Pobiera plik z URL i zapisuje pod out_path (z plikiem tymczasowym .part)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=out_path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        tmp_path.replace(out_path)


def _build_raw_url(owner: str, repo: str, branch: str, filename: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"


def download_matches(
    start_year: int,
    end_year: int,
    out_dir: Path,
    owner: str,
    repo: str,
    branches_to_try: list[str],
    force: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        filename = f"atp_matches_{year}.csv"
        out_path = out_dir / filename

        if out_path.exists() and not force:
            print(f"[SKIP] {out_path} już istnieje")
            continue

        last_err: Optional[Exception] = None
        downloaded = False

        for branch in branches_to_try:
            url = _build_raw_url(owner, repo, branch, filename)
            try:
                print(f"[GET] {url}")
                _download_file(url, out_path)
                downloaded = True
                break
            except Exception as e:
                last_err = e
                continue

        if not downloaded:
            raise RuntimeError(
                f"Nie udało się pobrać {filename}. Ostatni błąd: {repr(last_err)}"
            )

    print(f"\nOK. Pliki są w: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pobiera atp_matches_YYYY.csv z JeffSackmann/tennis_atp do data/raw/"
    )
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--owner", type=str, default="JeffSackmann")
    parser.add_argument("--repo", type=str, default="tennis_atp")
    parser.add_argument(
        "--branches",
        type=str,
        default="master,main",
        help="Lista branchy do sprawdzenia po kolei, np. 'master,main'",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Jeśli ustawione, nadpisuje istniejące pliki",
    )
    args = parser.parse_args()

    branches_to_try = [b.strip() for b in args.branches.split(",") if b.strip()]
    download_matches(
        start_year=args.start_year,
        end_year=args.end_year,
        out_dir=Path(args.out_dir),
        owner=args.owner,
        repo=args.repo,
        branches_to_try=branches_to_try,
        force=args.force,
    )


if __name__ == "__main__":
    main()
