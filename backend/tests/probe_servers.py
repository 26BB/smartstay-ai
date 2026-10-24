"""
SmartStay AI — Live Server Prober
Validates runtime HTTP responses from FastAPI and Streamlit servers.
"""

import subprocess
import time
import urllib.request
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

def probe_fastapi():
    print(f"[Probe] Starting FastAPI daemon from {BASE_DIR}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR
    proc = subprocess.Popen([
        PYTHON_EXE, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"
    ], cwd=BASE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        connected = False
        start_time = time.time()
        url = "http://127.0.0.1:8000/api/v1/health"

        print("[Probe] Waiting for FastAPI server to bind on port 8000...")
        while time.time() - start_time < 20:
            if proc.poll() is not None:
                out, err = proc.communicate()
                print(f"Uvicorn terminated early!\nstdout:\n{out.decode('utf-8', errors='ignore')}\nstderr:\n{err.decode('utf-8', errors='ignore')}")
                return

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SmartStay-Probe/1.0"})
                with urllib.request.urlopen(req, timeout=2) as response:
                    status_code = response.getcode()
                    body = response.read().decode("utf-8")
                    data = json.loads(body)
                    print(f"\n[SUCCESS] Live FastAPI HTTP Server Verified!")
                    print(f"Status: HTTP {status_code} OK")
                    print(f"Payload: {data}")
                    assert status_code == 200
                    assert data["status"] == "online"
                    connected = True
                    break
            except Exception:
                time.sleep(1)

        if not connected:
            print("[Probe Error] Could not connect to FastAPI server within timeout.")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
        print("[Probe] FastAPI process cleanly terminated.")

def probe_streamlit():
    print(f"\n[Probe] Starting Streamlit daemon from {BASE_DIR}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR
    proc = subprocess.Popen([
        PYTHON_EXE, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"
    ], cwd=BASE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        connected = False
        start_time = time.time()
        url = "http://127.0.0.1:8501/_stcore/health"

        print("[Probe] Waiting for Streamlit server to bind on port 8501...")
        while time.time() - start_time < 20:
            if proc.poll() is not None:
                out, err = proc.communicate()
                print(f"Streamlit terminated early!\nstdout:\n{out.decode('utf-8', errors='ignore')}\nstderr:\n{err.decode('utf-8', errors='ignore')}")
                return

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SmartStay-Probe/1.0"})
                with urllib.request.urlopen(req, timeout=2) as response:
                    status_code = response.getcode()
                    body = response.read().decode("utf-8")
                    print(f"\n[SUCCESS] Live Streamlit UI Server Verified!")
                    print(f"Status: HTTP {status_code} OK")
                    print(f"Health Response: {body.strip()}")
                    assert status_code == 200
                    connected = True
                    break
            except Exception:
                time.sleep(1)

        if not connected:
            print("[Probe Error] Could not connect to Streamlit server within timeout.")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
        print("[Probe] Streamlit process cleanly terminated.")

if __name__ == "__main__":
    probe_fastapi()
    probe_streamlit()
