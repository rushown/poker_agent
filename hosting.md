# VPS Hosting Details

## Provider
- **Company:** InterServer
- **Plan:** KVM Linux VPS Slice (1 Slice)
- **Cost:** $3.00/month
- **Location:** Los Angeles (KVM14.lax1)
- **Billing:** Monthly, next invoice July 7, 2026

## Server Info
- **Hostname:** vps3431843.trouble-free.net
- **IP:** 153.75.235.189
- **OS:** Ubuntu 26.04 64bit
- **RAM:** 2048 MB
- **Disk:** 40 GB
- **Bandwidth:** 2000 GB/month
- **VPS ID:** 3431843

## Account
- **Account email:** nabindada22@gmail.com
- **Invoice ID:** 46112677

## SSH Access
```bash
ssh root@153.75.235.189
```

## Setup Commands (run once after VPS is ready)
```bash
apt update && apt install -y python3-venv python3-pip git tmux
git clone https://github.com/<your-username>/poker_agent.git
cd poker_agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
nano .env  # paste API keys
```

## Running Bots
```bash
# Start a bot in a tmux session
tmux new -s devil
.venv/bin/python run_devil.py
# Ctrl+B then D to detach

tmux new -s aggro
.venv/bin/python run_aggro.py
# Ctrl+B then D to detach
```

## Reconnecting
```bash
ssh root@153.75.235.189
tmux attach -t devil
tmux attach -t aggro
tmux ls  # list all sessions
```
