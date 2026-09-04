"""
client_example.py
-------------------
LAYER 1: Client Layer (example)

This simulates what a browser extension, mobile app, or any backend
service would do: take a URL the user is about to visit (or a URL found
in an email/SMS) and call the cloud classifier's /predict endpoint.

In a real browser extension you'd call this from a background script
using fetch(); in a mobile app, from your HTTP client (OkHttp/URLSession);
this Python version is just a stand-in so the whole pipeline is runnable.
"""

import sys
import json
import urllib.request
import urllib.error

API_URL = "http://localhost:8080/predict"  # replace with your API Gateway / Cloud Run URL


def check_url(url: str, api_url: str = API_URL) -> dict:
    body = json.dumps({"url": url, "explain": True}).encode("utf-8")
    req = urllib.request.Request(
        api_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e)}


def main():
    urls = sys.argv[1:] or [
        "https://www.wikipedia.org/",
        "http://paypal-verify.account-secure.info/login",
        "http://23.94.51.12/bank/confirm.php?id=8871",
    ]
    for url in urls:
        result = check_url(url)
        level = result.get("level", "?").upper()
        score = result.get("score", "?")
        print(f"[{level:10s}] score={score}  {url}")
        if "explanation" in result:
            top = result["explanation"][0]
            print(f"             top signal: {top['feature']} = {top['value']}")


if __name__ == "__main__":
    main()
