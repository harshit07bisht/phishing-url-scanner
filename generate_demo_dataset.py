"""
generate_demo_dataset.py
-------------------------
Builds a small LABELED demo dataset (url, label) so the whole pipeline can
be trained and tested end-to-end offline.

IMPORTANT: This is synthetic/rule-generated data meant only to prove the
pipeline works. For a real deployment, replace `data/urls.csv` with a real
labeled dataset such as:
  - PhishTank (https://phishtank.org) verified phishing feed
  - OpenPhish (https://openphish.com)
  - UCI ML Repository "Phishing Websites" dataset
  - Kaggle "Phishing Site URLs" dataset (Malicious_URLs.csv, etc.)
The rest of the pipeline (feature_extraction.py, train_model.py, app.py)
does not need to change — only swap the CSV.
"""

import csv
import random

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "amazon.com", "microsoft.com",
    "apple.com", "nytimes.com", "bbc.com", "linkedin.com", "stackoverflow.com",
    "python.org", "reddit.com", "spotify.com", "netflix.com", "dropbox.com",
    "cloudflare.com", "mozilla.org", "wordpress.org", "adobe.com", "salesforce.com",
]
LEGIT_PATHS = [
    "/", "/about", "/search?q=news", "/docs/api", "/blog/2026/updates",
    "/user/settings", "/products/list", "/help/faq", "/pricing", "/contact",
]

PHISH_BRANDS = ["paypal", "amazon", "apple", "bankofamerica", "netflix", "microsoft", "chase", "wellsfargo"]
PHISH_WORDS = ["login", "verify", "secure", "update", "confirm", "account", "webscr", "signin", "recover"]
PHISH_TLDS = [".info", ".xyz", ".top", ".click", ".ru", ".tk", ".buzz", ".gq"]
SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly"]


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"
    sub = random.choice(["", "www.", "docs.", "www.", ""])
    return f"{scheme}://{sub}{domain}{path}"


def make_phish_url():
    style = random.choice(["ip", "fake_subdomain", "shortener", "hyphen_brand", "long_query"])
    brand = random.choice(PHISH_BRANDS)
    word = random.choice(PHISH_WORDS)
    tld = random.choice(PHISH_TLDS)

    if style == "ip":
        return f"http://{random_ip()}/{brand}/{word}.php?session={random.randint(1000,9999)}"
    if style == "fake_subdomain":
        return f"http://{brand}-{word}.{random.choice(['secure-check','account-center','verify-now'])}{tld}/{word}"
    if style == "shortener":
        return f"http://{random.choice(SHORTENERS)}/{random.choice(['a1B2c3','xY9zQ','Pk3mN'])}"
    if style == "hyphen_brand":
        return f"http://{brand}-{word}-{random.choice(['secure','support','online'])}.com-{word}{tld}/{word}.html"
    # long_query
    return f"http://{brand}{word}{tld}/index.php?cmd=_{word}&user=user{random.randint(1,999)}&token={random.randint(10000,99999)}"


def build(n_per_class=250):
    rows = []
    for _ in range(n_per_class):
        rows.append((make_legit_url(), 0))
    for _ in range(n_per_class):
        rows.append((make_phish_url(), 1))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build(n_per_class=300)
    with open("data/urls.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to data/urls.csv "
          f"({sum(r[1] for r in rows)} phishing / {len(rows) - sum(r[1] for r in rows)} legit)")
