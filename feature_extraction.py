"""
feature_extraction.py
----------------------
LAYER 2: Feature Extraction Layer

Converts a raw URL (string) into a fixed-length numeric feature vector that
the Random Forest model can consume.

Design notes:
- All features here are LEXICAL / STRUCTURAL — computed purely from the URL
  string itself, with no network calls (no DNS, no WHOIS, no HTTP fetch).
  This keeps the layer fast, stateless, and safe to run inside a serverless
  function (AWS Lambda / GCP Cloud Function / Azure Function) on every
  request without adding latency or outbound-network risk.
- If you want stronger signals later (domain age, SSL certificate issuer,
  WHOIS registrar, Google Safe Browsing hits, page content features), add
  them as a *second* optional feature block behind a flag — see
  `extract_features(url, enrich=False)`.
"""

import re
import math
import ipaddress
from urllib.parse import urlparse
from collections import Counter

# Ordered list of feature names — order MUST match the vector produced by
# `extract_features`. The model is trained on this exact ordering.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_ampersands",
    "num_digits",
    "digit_ratio",
    "num_subdomains",
    "has_ip_address",
    "has_https_scheme",
    "has_port",
    "has_double_slash_in_path",
    "has_hyphen_in_domain",
    "has_https_token_in_domain",  # e.g. "https-login.badsite.com"
    "num_query_params",
    "is_shortening_service",
    "suspicious_word_count",
    "url_entropy",
    "tld_length",
    "path_depth",
]

# Known URL-shortening services often abused to hide the real phishing target
SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc",
}

# Words commonly used in phishing URLs to create urgency / trust
SUSPICIOUS_WORDS = {
    "login", "signin", "verify", "verification", "update", "secure",
    "account", "bank", "confirm", "password", "pay", "billing",
    "suspend", "unlock", "webscr", "ebayisapi", "recover", "wallet",
    "security", "alert", "invoice",
}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _is_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    host = hostname.split(":")[0]  # strip port if present
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _registrable_domain_parts(hostname: str):
    """Lightweight subdomain/domain/TLD split without external TLD lists.
    Good enough for feature purposes (not for exact eTLD+1 resolution)."""
    if not hostname:
        return [], "", ""
    parts = hostname.split(".")
    if len(parts) <= 2:
        return [], (parts[0] if parts else ""), (parts[-1] if parts else "")
    return parts[:-2], parts[-2], parts[-1]


def extract_features(url: str) -> dict:
    """
    Parse `url` and return a dict of {feature_name: numeric_value}.
    Never raises on malformed input — falls back to safe defaults so the
    inference layer can always get a usable vector.
    """
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        # No scheme provided (e.g. "example.com/login") -> assume http for parsing
        parse_target = "http://" + url
    else:
        parse_target = url

    try:
        parsed = urlparse(parse_target)
    except Exception:
        parsed = urlparse("http://invalid")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    subdomains, domain, tld = _registrable_domain_parts(hostname)

    digits = sum(c.isdigit() for c in url)
    url_len = len(url) if len(url) > 0 else 1

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_digits": digits,
        "digit_ratio": round(digits / url_len, 4),
        "num_subdomains": len(subdomains),
        "has_ip_address": int(_is_ip_address(hostname)),
        "has_https_scheme": int(parsed.scheme == "https"),
        "has_port": int(parsed.port is not None) if hostname else 0,
        "has_double_slash_in_path": int("//" in path),
        "has_hyphen_in_domain": int("-" in domain),
        "has_https_token_in_domain": int("https" in hostname.replace(".", "") or "ssl" in hostname),
        "num_query_params": len([p for p in query.split("&") if p]) if query else 0,
        "is_shortening_service": int(hostname in SHORTENING_SERVICES),
        "suspicious_word_count": sum(1 for w in SUSPICIOUS_WORDS if w in url.lower()),
        "url_entropy": round(_shannon_entropy(url), 4),
        "tld_length": len(tld),
        "path_depth": len([p for p in path.split("/") if p]),
    }
    return features


def extract_feature_vector(url: str) -> list:
    """Return the feature values in FEATURE_NAMES order (what the model expects)."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]


if __name__ == "__main__":
    samples = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/login.php?verify=account",
        "http://secure-paypal-login.com-update.info/webscr?cmd=confirm",
        "https://bit.ly/3xYzAbc",
    ]
    for s in samples:
        print(s)
        for k, v in extract_features(s).items():
            print(f"   {k:28s} {v}")
        print()
