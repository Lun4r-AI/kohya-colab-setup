"""
Colab-friendly Kohya installer / starter script
- Clones kohya repositories you want (kohya_ss for local-style, kohya-colab for colab helper)
- Creates standard folders under /content/kohya
- Saves CivitAI / HF read tokens to a JSON file for later use
- Shows live output in notebook and avoids re-cloning / re-installing when already present

Usage in Colab (example):
%run /content/kohya_colab_installer.py --install --civitai_key YOUR_KEY --hf_read_token YOUR_HF_TOKEN

Notes:
- This script does NOT force CUDA / torch installs. Installing torch/xformers inside Colab often triggers version conflicts.
  I leave install commands visible and optional so you can run them when you're ready.
- The script focuses on reproducible file layout and bootstrapping; run installs manually once you confirm GPU/runtime.

Author: modified for Kohya by ChatGPT
"""

from IPython.display import display, Image, clear_output
from IPython import get_ipython
from ipywidgets import widgets
from pathlib import Path
import subprocess
import argparse
import shlex
import json
import sys
import os
import time

# ---------- Helpers ----------

def run(cmd, stream_output=True):
    """Run a shell command and stream stdout/stderr"""
    print(f"\n⚡ {cmd}\n")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            print(line, end="")
    rc = proc.poll()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def safe_clone(repo_url, dest: Path, branch=None):
    """Clone repo_url into dest if not already present. If present, do a git pull if it is a git repo."""
    dest = Path(dest)
    if dest.exists():
        git_dir = dest / '.git'
        if git_dir.exists():
            print(f"✔ {dest} exists — pulling latest")
            try:
                run(f"cd {dest} && git fetch --all --prune && git pull --rebase")
            except Exception as e:
                print("⚠ pull failed:", e)
        else:
            print(f"✔ {dest} exists — not a git repo, skipping clone")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"git clone {'-b '+branch if branch else ''} {repo_url} {dest}"
    run(cmd)


def write_tokens(output_path: Path, civitai_key: str, hf_read_token: str):
    data = {
        'civitai_key': civitai_key or '',
        'hf_read_token': hf_read_token or ''
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"✔ Saved tokens to {output_path}")


# ---------- Main installer logic ----------

# Default paths
BASE = Path('/content/kohya')
KOHYA_SS = BASE / 'kohya_ss'             # local-style trainer repo (bmaltais / official forks)
KOHYA_COLAB = BASE / 'kohya-colab'       # colab helper (hollowstrawberry / kohya-colab)
MODELS = BASE / 'models'
DATASET = BASE / 'dataset'
OUTPUT = BASE / 'outputs'
KEYFILE = BASE / 'api-keys.json'

parser = argparse.ArgumentParser(description='Colab Kohya bootstrapper')
parser.add_argument('--install', action='store_true', help='Clone repos & create folders')
parser.add_argument('--clone_kohya_ss', action='store_true', help='Clone bmaltais/kohya_ss (local-style kohya)')
parser.add_argument('--clone_kohya_colab', action='store_true', help='Clone hollowstrawberry/kohya-colab (colab helper)')
parser.add_argument('--civitai_key', default='', help='CivitAI API key (optional)')
parser.add_argument('--hf_read_token', default='', help='HuggingFace READ token (optional)')
parser.add_argument('--skip_installs', action='store_true', help='Do not run pip installs (recommended)')
parser.add_argument('--show_installs', action='store_true', help='Print recommended pip install commands')
args, unknown = parser.parse_known_args()

# If --install passed, default to cloning both unless specific flags set
if args.install:
    if not (args.clone_kohya_ss or args.clone_kohya_colab):
        args.clone_kohya_ss = True
        args.clone_kohya_colab = True

# Create base tree
for p in (BASE, MODELS, DATASET, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)
print(f"✔ Created base directories under {BASE}")

# Save tokens
if args.civitai_key or args.hf_read_token:
    write_tokens(KEYFILE, args.civitai_key, args.hf_read_token)
else:
    if KEYFILE.exists():
        print(f"✔ Found existing key file: {KEYFILE}")
    else:
        print("⚠ No tokens provided and no existing api-keys.json found. Private model downloads may fail without keys.")

# Clone repos
if args.clone_kohya_ss:
    # recommended repo: bmaltais/kohya_ss (fork that many use). You can swap URL to your preferred kohya clone.
    repo = 'https://github.com/bmaltais/kohya_ss.git'
    print('\n--- kohya_ss ---')
    safe_clone(repo, KOHYA_SS)

if args.clone_kohya_colab:
    repo2 = 'https://github.com/hollowstrawberry/kohya-colab.git'
    print('\n--- kohya-colab ---')
    safe_clone(repo2, KOHYA_COLAB)

# Provide optional install commands (left commented so you run them consciously)
install_notes = []
install_notes.append("# Recommended: verify Colab runtime is GPU (Runtime -> Change runtime type -> GPU)")
install_notes.append("# Example safe install sequence (may cause package version churn):")
install_notes.append("# pip install -U torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121")
install_notes.append("# pip install -r /content/kohya-colab/requirements.txt")
install_notes.append("# pip install -e /content/kohya-colab")
install_notes.append("")

if args.show_installs:
    print('\n'.join(install_notes))

if not args.skip_installs and (args.clone_kohya_colab or args.clone_kohya_ss):
    print('\n⚠ No automatic pip installs were run (safe default). Use --show_installs to reveal recommended commands.')

# Final status summary
print('\n===== SUMMARY =====')
print('Base folder: ', BASE)
print('kohya_ss:    ', 'present' if KOHYA_SS.exists() else 'missing')
print('kohya-colab: ', 'present' if KOHYA_COLAB.exists() else 'missing')
print('models dir:  ', MODELS)
print('dataset dir: ', DATASET)
print('outputs dir: ', OUTPUT)
print('api keys:    ', KEYFILE if KEYFILE.exists() else 'none')

print('\nNext steps:')
print(' - Put your training images in', DATASET)
print(' - If you cloned kohya-colab, open /content/kohya-colab and inspect the colab launcher scripts (they include accelerate launch helpers).')
print(' - Run installs manually using the shown pip commands when you are ready. Installing different torch versions may conflict with preinstalled packages in Colab.')
print('\nExample quick start to test environment:')
print("  cd /content/kohya-colab && python3 train_network.py --help  # will error unless deps installed, but shows files")

# Display a small UI card explaining what was done
output = widgets.Output()
with output:
    display(Image(url='https://raw.githubusercontent.com/hollowstrawberry/kohya-colab/main/docs/logo.png')) if True else None
    print('\nBootstrapping complete. Check the notebook output above for clone/install messages.')

display(output)

# Save a small manifest for notebook helpers to read later
manifest = {
    'base': str(BASE),
    'kohya_ss': str(KOHYA_SS) if KOHYA_SS.exists() else '',
    'kohya_colab': str(KOHYA_COLAB) if KOHYA_COLAB.exists() else '',
    'models': str(MODELS),
    'dataset': str(DATASET),
    'outputs': str(OUTPUT),
    'keys': str(KEYFILE) if KEYFILE.exists() else ''
}
(Path(BASE) / 'bootstrap_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Manifest written to', BASE / 'bootstrap_manifest.json')

# EOF
