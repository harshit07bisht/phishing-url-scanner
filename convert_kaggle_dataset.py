"""
convert_kaggle_dataset.py
---------------------------
Converts the Kaggle "Phishing Site URLs" dataset (columns: URL, Label with
values 'good'/'bad') into the format train_model.py expects (columns:
url, label with values 0/1, where 1 = phishing).

Usage:
    python convert_kaggle_dataset.py

Reads:  data/phishing_site_urls.csv   (the file you downloaded from Kaggle)
Writes: data/urls.csv                 (overwrites the old synthetic demo data)

By default it takes a balanced SAMPLE of the data (not all ~550,000 rows),
so training stays fast on a normal laptop. Increase SAMPLE_PER_CLASS below
for a more accurate (but slower to train) model.
"""

import csv
import random

random.seed(42)

INPUT_FILE = "data/phishing_site_urls.csv"
OUTPUT_FILE = "data/urls.csv"

# How many rows of EACH class (safe / phishing) to keep.
# 10,000 + 10,000 = 20,000 rows trains in well under a minute on a laptop.
# Set this higher (e.g. 50000) for a more accurate model if you don't mind
# training taking a few minutes longer.
SAMPLE_PER_CLASS = 10000

LABEL_MAP = {"good": 0, "bad": 1}


def main():
    legit_rows = []
    phish_rows = []

    with open(INPUT_FILE, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        # Kaggle's header is "URL,Label" — handle case differences safely.
        fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
        url_col = fieldnames.get("url")
        label_col = fieldnames.get("label")
        if not url_col or not label_col:
            raise SystemExit(
                f"Could not find URL/Label columns. Found columns: {reader.fieldnames}"
            )

        for row in reader:
            url = (row.get(url_col) or "").strip()
            label_raw = (row.get(label_col) or "").strip().lower()
            if not url or label_raw not in LABEL_MAP:
                continue
            label = LABEL_MAP[label_raw]
            if label == 0:
                legit_rows.append((url, 0))
            else:
                phish_rows.append((url, 1))

    print(f"Found {len(legit_rows)} safe URLs and {len(phish_rows)} phishing URLs in the dataset.")

    random.shuffle(legit_rows)
    random.shuffle(phish_rows)

    n = min(SAMPLE_PER_CLASS, len(legit_rows), len(phish_rows))
    sampled = legit_rows[:n] + phish_rows[:n]
    random.shuffle(sampled)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(sampled)

    print(f"Wrote {len(sampled)} rows ({n} safe / {n} phishing) to {OUTPUT_FILE}")
    print("Now run: python train_model.py")


if __name__ == "__main__":
    main()