import requests
import json
import time
import subprocess
import os

def get_ngrok_url():
    try:
        res = requests.get("http://localhost:8080/api/tunnels")
        tunnels = res.json()["tunnels"]
        for t in tunnels:
            if t["proto"] == "https":
                return t["public_url"]
    except Exception as e:
        print("Error getting Ngrok URL:", e)
    return None

def save_to_json(url, path):
    with open(path, "w") as f:
        json.dump({"ngrok_url": url}, f, indent=2)

def push_to_github(commit_message="Update ngrok_url"):
    try:
        subprocess.run(["git", "add", "ngrok_url.json"], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("Changes pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print("Git error:", e)

# Full flow
json_path = "ngrok_url.json"
ngrok_url = get_ngrok_url()
if ngrok_url:
    print("Updating ngrok_url.json:", ngrok_url)
    save_to_json(ngrok_url, json_path)
    push_to_github()
else:
    print("Ngrok not active?")
