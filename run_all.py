import subprocess
import time
import requests
import os
from pathlib import Path

# CONFIG
API_PORT = 8080

# Step 1: Start the local Flask API
print("[+] Starting local API (local_api.py)...")
api_proc = subprocess.Popen(["python", "local_api.py"])

# Step 2: Start ngrok
print("[+] Starting ngrok...")
ngrok_proc = subprocess.Popen(["ngrok", "http", str(API_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Step 3: Wait for ngrok to start
def wait_for_ngrok(timeout=15):
    print("[*] Waiting for ngrok tunnel to be ready...")
    for _ in range(timeout):
        try:
            res = requests.get("http://localhost:4040/api/tunnels")
            if res.status_code == 200 and res.json()["tunnels"]:
                print("[✓] Ngrok is running.")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    print("[✗] Ngrok did not start in time.")
    return False

# Step 4: Push ngrok URL to GitHub
def run_update_script():
    if Path("get_api_url.py").exists():
        print("[*] Running get_api_url.py to update GitHub...")
        subprocess.run(["python", "get_api_url.py"])
    else:
        print("[!] Could not find get_api_url.py.")

# Run update only if ngrok is up
if wait_for_ngrok():
    run_update_script()
else:
    print("[!] Skipping GitHub update because ngrok failed.")

# Step 5: Keep alive (Ctrl+C to stop)
print("[*] Local API and Ngrok are running. Press Ctrl+C to exit.")
try:
    ngrok_proc.wait()
except KeyboardInterrupt:
    print("\n[!] Shutting down...")

    ngrok_proc.terminate()
    api_proc.terminate()
    print("[✓] Processes terminated.")
