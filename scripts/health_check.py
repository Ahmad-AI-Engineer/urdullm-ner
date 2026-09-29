import json
import os
import urllib.request

port = os.getenv("PORT", "8005")
url = f"http://127.0.0.1:{port}/api/health"

with urllib.request.urlopen(url, timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))

if payload.get("status") != "ok":
    raise SystemExit("Health endpoint did not report status=ok")
