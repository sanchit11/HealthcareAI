# HealthcareAI
# Synapta_AI

---

# Azure VM Deployment Guide

**Target audience:** DevOps Engineer  
**Application:** SynaptaEMR AI — FastAPI + Uvicorn  
**Entry point:** `main.py`

---

## 1. Provision the Azure VM

> **⚠️ OS Requirement: Windows VM is required.**  
> The application depends on `pywin32`, which is a Windows-only package. Deploy on a **Windows Server** Azure VM — Linux VMs are not supported.

1. Create a **Windows Server 2022 Datacenter** VM in Azure Portal (or CLI).  
   Recommended size: **Standard_D4s_v3** (4 vCPUs, 16 GB RAM) minimum — the app loads TensorFlow, PyTorch, spaCy, and Whisper models into memory.
2. Open inbound ports: **3389** (RDP), **80** (HTTP), **443** (HTTPS), and **8000** (FastAPI direct, for testing only).
3. Connect to the VM via RDP:
   - Open **Remote Desktop Connection** and enter `<VM_PUBLIC_IP>`
   - Log in with the admin credentials set during VM creation.

---

## 2. System Dependencies

Run all commands in **PowerShell (Run as Administrator)**.

```powershell
# Install Chocolatey (Windows package manager) if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python 3.11
choco install python311 -y

# Install Git
choco install git -y

# Install FFmpeg (required by Whisper audio processing)
choco install ffmpeg -y

# Reload environment so python/pip are on PATH
refreshenv

# Verify Python version
python --version
```

---

## 3. Clone the Repository

```powershell
cd C:\
git clone <YOUR_REPO_URL> synaptaemr
cd C:\synaptaemr
```

> If deploying from a zip/artifact instead of Git, extract files to `C:\synaptaemr\`.

---

## 4. Create Python Virtual Environment

```powershell
cd C:\synaptaemr

# Create venv
python -m venv venv

# Activate venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip
```

> If PowerShell blocks script execution, run first:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## 5. Install Dependencies

```powershell
# Install all dependencies (this will take several minutes — large ML stack)
pip install -r requirements.txt
```

> `pywin32` is included in `requirements.txt` and installs correctly on Windows — no modifications needed.  
> The spaCy model `en_core_web_lg` is installed directly from a GitHub release URL already listed in `requirements.txt` — no extra step needed.

**Verify key packages loaded correctly:**
```powershell
python -c "import fastapi, uvicorn, tensorflow, torch, spacy; print('All core packages OK')"
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('spaCy model OK')"
```

---

## 6. Configure Environment Variables

The app uses `.env` via `python-dotenv`. Create the file in the project root:

```powershell
# If an example file exists:
copy .env.example .env

# Otherwise create it directly:
notepad C:\synaptaemr\.env
```

Populate the required keys (adjust to match your actual config):

```ini
# OpenAI
OPENAI_API_KEY=sk-...

# App settings
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000

# Add any other keys your application requires
```

Secure the file — restrict access via Windows file permissions:
```powershell
icacls C:\synaptaemr\.env /inheritance:r /grant:r "$env:USERNAME:(R)"
```

---

## 7. Test the Application

Run manually first to confirm it starts without errors:

```powershell
cd C:\synaptaemr
.\venv\Scripts\Activate.ps1

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://<VM_PUBLIC_IP>:8000/docs` in a browser to verify the Swagger UI loads.  
Stop the process (`Ctrl+C`) before proceeding.

---

## 8. Run as a Windows Service

Use **NSSM** (Non-Sucking Service Manager) to run uvicorn as a Windows service that starts automatically and restarts on failure.

```powershell
# Install NSSM
choco install nssm -y

# Register the service
nssm install SynaptaEMR "C:\synaptaemr\venv\Scripts\uvicorn.exe"
nssm set SynaptaEMR AppParameters "main:app --host 0.0.0.0 --port 8000 --workers 2 --log-level info"
nssm set SynaptaEMR AppDirectory "C:\synaptaemr"
nssm set SynaptaEMR AppEnvironmentExtra "DOTENV_PATH=C:\synaptaemr\.env"
nssm set SynaptaEMR Start SERVICE_AUTO_START

# Start the service
nssm start SynaptaEMR

# Check status
nssm status SynaptaEMR

# View logs (NSSM captures stdout/stderr to files — configure output path)
nssm set SynaptaEMR AppStdout "C:\synaptaemr\logs\service.log"
nssm set SynaptaEMR AppStderr "C:\synaptaemr\logs\error.log"
New-Item -ItemType Directory -Force -Path C:\synaptaemr\logs
nssm restart SynaptaEMR
```

You can also manage the service from **Services** (`services.msc`) in Windows.

---

## 9. Configure Nginx as Reverse Proxy

This proxies port 80 → uvicorn on port 8000, and handles WebSocket upgrades (the app uses WebSockets).

```powershell
# Install Nginx for Windows
choco install nginx -y

# Default config location: C:\tools\nginx\conf\nginx.conf
# Edit the config:
notepad C:\tools\nginx\conf\nginx.conf
```

Replace the `server { }` block with:

```nginx
server {
    listen 80;
    server_name <VM_PUBLIC_IP_OR_DOMAIN>;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
```

Start Nginx:

```powershell
cd C:\tools\nginx
# Test config
.\nginx.exe -t

# Start
Start-Process .\nginx.exe

# To reload config after changes
.\nginx.exe -s reload
```

To run Nginx as a Windows service automatically:
```powershell
nssm install Nginx "C:\tools\nginx\nginx.exe"
nssm set Nginx AppDirectory "C:\tools\nginx"
nssm start Nginx
```

---

## 10. (Optional) Enable HTTPS with Let's Encrypt

```powershell
# Install win-acme (ACME client for Windows)
choco install win-acme -y

# Run the wizard — it will auto-configure Nginx and renew certs
wacs
```

> Requires a domain name pointed at the VM's public IP.

---

## 11. Verify Deployment

| Check | Command |
|---|---|
| Service running | `nssm status SynaptaEMR` |
| Logs | `Get-Content C:\synaptaemr\logs\service.log -Tail 50` |
| HTTP health | `curl http://localhost:8000/docs` |
| Nginx status | `nssm status Nginx` |

---

## Troubleshooting

**Service fails to start**  
→ Check logs: `Get-Content C:\synaptaemr\logs\error.log -Tail 100`  
→ Confirm `.env` exists and all required keys are set  
→ Re-run `pip install -r requirements.txt` inside the venv and check for errors  

**Out of memory**  
→ The ML stack (TensorFlow + PyTorch + spaCy + Whisper) is heavy. Upgrade to a larger VM size (e.g. `Standard_D8s_v3`). Windows manages its own page file automatically, but you can increase it manually via *System Properties → Advanced → Performance → Virtual Memory*.

**spaCy model not found**  
→ Re-run inside the venv:
```powershell
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
```

**WebSocket connections dropping**  
→ Increase Nginx timeouts (`proxy_read_timeout`) and confirm the `Upgrade` headers are set correctly (see Step 9).

**Windows Firewall blocking port 8000 or 80**  
→ Open the ports:
```powershell
New-NetFirewallRule -DisplayName "FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
New-NetFirewallRule -DisplayName "HTTP 80" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow
```
