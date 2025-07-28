import requests
import json
import base64
import os
# CONFIG
GITHUB_USERNAME = "Lgreen02"
REPO_NAME = "SlideSite"
BRANCH = "main"
FILE_PATH = "ngrok_url.json"
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def get_ngrok_url():
    try:
        res = requests.get("http://localhost:4040/api/tunnels")
        tunnels = res.json()["tunnels"]
        for t in tunnels:
            if t["proto"] == "https":
                return t["public_url"]
    except Exception as e:
        print("Error fetching Ngrok URL:", e)
    return None

def get_file_sha():
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.json()["sha"]
    return None

def update_github_file(ngrok_url):
    content = json.dumps({"ngrok_url": ngrok_url}, indent=2)
    encoded_content = base64.b64encode(content.encode()).decode()
    file_sha = get_file_sha()

    payload = {
        "message": "Update ngrok_url.json",
        "content": encoded_content,
        "branch": BRANCH
    }
    if file_sha:
        payload["sha"] = file_sha  # Required for updating

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    res = requests.put(url, headers=headers, json=payload)

    if res.status_code in [200, 201]:
        print("Successfully updated ngrok_url.json on GitHub.")
    else:
        print("GitHub API error:", res.status_code, res.text)

# MAIN
ngrok_url = get_ngrok_url()
if ngrok_url:
    update_github_file(ngrok_url)
else:
    print("Ngrok not running or tunnel not available.")
