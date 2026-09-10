"""
Downloads the real NSL-KDD intrusion-detection dataset (train + test).

NSL-KDD is the benchmark referenced in the source papers ("Experimental
results on the NSL-KDD and UNSW-NB15 datasets ..."). It fixes the
duplicate-record bias of the original KDD'99 Cup dataset and is the
standard offline benchmark for network intrusion / insider-threat
detection research.

Run:
    python data/download_data.py

This writes KDDTrain+.txt (125,973 rows) and KDDTest+.txt (22,544 rows)
into this directory. If you already have these files (e.g. this repo
ships with them pre-downloaded), the script is a no-op.
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]
HERE = os.path.dirname(os.path.abspath(__file__))


def download():
    for fname in FILES:
        dest = os.path.join(HERE, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[skip] {fname} already present ({os.path.getsize(dest):,} bytes)")
            continue
        url = BASE + fname.replace("+", "%2B")
        print(f"[download] {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"[done] {fname}: {os.path.getsize(dest):,} bytes")


if __name__ == "__main__":
    download()
