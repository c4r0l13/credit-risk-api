"""Download the UCI 'Default of Credit Card Clients' dataset (30,000 rows, ~2.8 MB).

Original source: https://archive.ics.uci.edu/dataset/350 (Yeh & Lien, 2009).
A CSV copy on GitHub is used for convenience.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/"
    "main/dataset/default_of_credit_card_clients.csv"
)


def main() -> None:
    out = Path("data/raw/default_of_credit_card_clients.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        urllib.request.urlretrieve(URL, out)
    print(f"Data ready: {out.resolve()}")


if __name__ == "__main__":
    main()
