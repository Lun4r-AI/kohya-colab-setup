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
import re

SyS = get_ipython().system
CD = os.chdir
iRON = os.environ

REPO = {
    'Kohya': 'https://github.com/bmaltais/kohya_ss'
}

WEBUI_LIST = ['Kohya']

def getENV():
    env = {
        'Colab': ('/content', '/content', 'COLAB_JUPYTER_TOKEN'),
        'Kaggle': ('/kaggle', '/kaggle/working', 'KAGGLE_DATA_PROXY_TOKEN')
    }
    for name, (base, home, var) in env.items():
        if var in iRON:
            return name, base, home
    return None, None, None

def getArgs():
    """
    Parse and validate args for Kohya installer.
    Returns: (selected_ui, civitai_key, hf_read_token)
    selected_ui will be the string "kohya".
    Exits with non-zero code on fatal validation errors.
    """
    parser = argparse.ArgumentParser(description='Kohya Installer Script for Kaggle and Google Colab')
    parser.add_argument('--model_family', required=True,
                        help='available model_family: Flux, Illustrious, SD15, SDXL, Pony')
    parser.add_argument('--civitai_key', required=True, help='your CivitAI API key')
    parser.add_argument('--hf_read_token', default=None, help='your Huggingface READ Token (optional)')

    args, unknown = parser.parse_known_args()

    # validate model_family against known list (case-insensitive)
    MODEL_FAMILIES = ['flux', 'illustrious', 'sd15', 'sdxl', 'pony']
    model_family = args.model_family.strip().lower()
    if model_family not in MODEL_FAMILIES:
        print(f"[ERROR] invalid model_family: \"{args.model_family}\"")
        print("Available model_family options:", ", ".join(MODEL_FAMILIES))
        sys.exit(2)

    civitai_key = args.civitai_key.strip()
    hf_read_token = args.hf_read_token.strip() if args.hf_read_token else ''

    # validate civitai_key (basic checks)
    if not civitai_key:
        print("[ERROR] CivitAI API key is missing.")
        sys.exit(2)
    if re.search(r'\s+', civitai_key):
        print(f"[ERROR] CivitAI API key contains spaces \"{civitai_key}\" - not allowed.")
        sys.exit(2)
    if len(civitai_key) < 32:
        # a conservative minimum length - adjust if your key is shorter/longer
        print("[ERROR] CivitAI API key must be at least 32 characters long.")
        sys.exit(2)

    # sanitize HF token
    if re.search(r'\s+', hf_read_token):
        hf_read_token = ''

    # For the rest of the script we return "kohya" as the selected UI/flow marker
    selected_ui = "kohya"
    return selected_ui, civitai_key, hf_read_token

def getPython():
    hao = selected_ui in ['Kohya']
    v = '3.11' if hao else '3.10'
    BIN = str(PY / 'bin')
    PKG = str(PY / f'lib/python{v}/site-packages')

    tar = {
        **dict.fromkeys(['ComfyUI', 'SwarmUI'], 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-ComfyUI-SwarmUI-Python310-Torch260-cu124.tar.lz4'),
       
    }

    url = tar.get(selected_ui, 'https://huggingface.co/gutris1/webui/resolve/main/env/KC-Python310-Torch260-cu124.tar.lz4')

    fn = Path(url).name

    CD(Path(ENVBASE).parent)
    print(f"\n{ARROW} installing Python Portable {'3.11.13' if hao else '3.10.15'}")

    SyS('sudo apt-get -qq -y install aria2 pv lz4 > /dev/null 2>&1')

    aria = f'aria2c --console-log-level=error --stderr=true -c -x16 -s16 -k1M -j5 {url} -o {fn}'
    p = subprocess.Popen(shlex.split(aria), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()

    SyS(f'pv {fn} | lz4 -d | tar -xf -')
    Path(f'/{fn}').unlink()

    sys.path.insert(0, PKG)
    if BIN not in iRON['PATH']:
        iRON['PATH'] = BIN + ':' + iRON['PATH']
    if PKG not in iRON['PYTHONPATH']:
        iRON['PYTHONPATH'] = PKG + ':' + iRON['PYTHONPATH']

    if ENVNAME == 'Kaggle':
        for cmd in [
            'pip install ipywidgets jupyterlab_widgets --upgrade',
            'rm -f /usr/lib/python3.10/sitecustomize.py'
        ]:
            SyS(f'{cmd} > /dev/null 2>&1')

def marking(base_path: Path, filename: str, ui: str):
    """
    Create/patch a small JSON marker file at base_path/filename with ui info.
    Example: marking(Path('/content/kohya'), 'launcher.json', 'kohya')
    """
    base_path = Path(base_path)
    t = base_path / filename
    v = {'ui': ui, 'launch_args': '', 'tunnel': ''}
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(json.dumps(v, indent=4))
    else:
        try:
            d = json.loads(t.read_text())
            if not isinstance(d, dict):
                d = {}
        except Exception:
            d = {}
        d.update(v)
        t.write_text(json.dumps(d, indent=4))


def key_inject(target_file: Path, civitai_key: str, hf_read_token: str):
    """
    Replace placeholder token strings in a target script/file with provided keys.
    The function looks for placeholders TOKET and TOBRUT in the file contents and substitutes them.
    Example: key_inject(Path('/content/kohya/some_script.py'), 'CIV_KEY', 'HF_TOKEN')
    """
    p = Path(target_file)
    if not p.exists():
        raise FileNotFoundError(f"key_inject: target file not found: {p}")
    v = p.read_text()
    # basic safe replacements (escape single quotes inside tokens)
    civ_safe = civitai_key.replace("'", "\\'")
    hf_safe = hf_read_token.replace("'", "\\'")
    v = v.replace("TOKET = ''", f"TOKET = '{civ_safe}'")
    v = v.replace("TOBRUT = ''", f"TOBRUT = '{hf_safe}'")
    p.write_text(v)


def install_tunnel(user_bin: Path):
    """
    Install small tunnel helpers to user_bin (Path).
    This function is generic and no longer relies on hidden globals.
    Example: install_tunnel(Path('/content/kohya/bin'))
    """
    user_bin = Path(user_bin)
    user_bin.mkdir(parents=True, exist_ok=True)

    # cloudflared
    cl_path = user_bin / 'cloudflared'
    try:
        SyS(f'wget -qO {cl_path} https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64')
        SyS(f'chmod +x {cl_path}')
    except Exception:
        print("[WARN] failed to fetch cloudflared; continuing")

    bins = {
        'zrok': {
            'bin': user_bin / 'zrok',
            'url': 'https://github.com/openziti/zrok/releases/download/v1.0.6/zrok_1.0.6_linux_amd64.tar.gz'
        },
        'ngrok': {
            'bin': user_bin / 'ngrok',
            'url': 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz'
        }
    }

    for n, b in bins.items():
        try:
            if b['bin'].exists():
                b['bin'].unlink()
            url = b['url']
            name = Path(url).name
            SyS(f'wget -qO {name} {url}')
            SyS(f'tar -xzf {name} -C {user_bin}')
            SyS(f'rm -f {name}')
        except Exception:
            print(f"[WARN] failed to install {n}; continuing")

def sym_link(U, M):
    configs = {
        'A1111': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet')
            ]
        },

        'Forge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'}",
                f"rm -rf {M / 'diffusion_models'} {M / 'text_encoder'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet'),
                (TMP / 'z123', M / 'z123'),
                (TMP / 'svd', M / 'svd'),
                (TMP / 'clip', M / 'clip'),
                (TMP / 'clip_vision', M / 'clip_vision'),
                (TMP / 'diffusers', M / 'diffusers'),
                (TMP / 'diffusion_models', M / 'diffusion_models'),
                (TMP / 'text_encoders', M / 'text_encoder'),
                (TMP / 'unet', M / 'unet')
            ]
        },

        'ReForge': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}",
                f"rm -rf {M / 'svd'} {M / 'z123'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet'),
                (TMP / 'z123', M / 'z123'),
                (TMP / 'svd', M / 'svd')
            ]
        },

        'Forge-Classic': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet')
            ]
        },

        'Forge-Neo': {
            'sym': [
                f"rm -rf {M / 'Stable-diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'ControlNet'}"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'ControlNet')
            ]
        },

        'ComfyUI': {
            'sym': [
                f"rm -rf {M / 'checkpoints/tmp_ckpt'} {M / 'loras/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'clip_vision'} {M / 'diffusers'} {M / 'diffusion_models'}",
                f"rm -rf {M / 'text_encoders'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'checkpoints/tmp_ckpt'),
                (TMP / 'lora', M / 'loras/tmp_lora'),
                (TMP / 'controlnet', M / 'controlnet'),
                (TMP / 'clip', M / 'clip'),
                (TMP / 'clip_vision', M / 'clip_vision'),
                (TMP / 'diffusers', M / 'diffusers'),
                (TMP / 'diffusion_models', M / 'diffusion_models'),
                (TMP / 'text_encoders', M / 'text_encoders'),
                (TMP / 'unet', M / 'unet')
            ]
        },

        'SwarmUI': {
            'sym': [
                f"rm -rf {M / 'Stable-Diffusion/tmp_ckpt'} {M / 'Lora/tmp_lora'} {M / 'controlnet'}",
                f"rm -rf {M / 'clip'} {M / 'unet'} {TMP}/*"
            ],
            'links': [
                (TMP / 'ckpt', M / 'Stable-Diffusion/tmp_ckpt'),
                (TMP / 'lora', M / 'Lora/tmp_lora'),
                (TMP / 'controlnet', M / 'controlnet'),
                (TMP / 'clip', M / 'clip'),
                (TMP / 'unet', M / 'unet')
            ]
        }
    }

    cfg = configs.get(U)
    [SyS(f'{cmd}') for cmd in cfg['sym']]
    if U not in ['ComfyUI', 'SwarmUI']: [(M / d).mkdir(parents=True, exist_ok=True) for d in ['Lora', 'ESRGAN']]
    [SyS(f'ln -s {src} {tg}') for src, tg in cfg['links']]

def kohya_requirements(base_path):
    """
    Install only the requirements needed for Kohya sd-scripts.
    No WebUI, no folders, no useless junk.
    """

    # Move into sd-scripts directory
    kohya_dir = base_path / "sd-scripts"
    CD(kohya_dir)

    # Install core requirements
    say("<br><b>Installing Kohya requirements…</b>")
    SyS("pip install --upgrade pip")

    # Install the official sd-scripts deps
    req_file = kohya_dir / "requirements.txt"
    if req_file.exists():
        SyS(f"pip install -r {req_file}")
    else:
        say("requirements.txt not found — installing known dependencies manually…")

        # fallback minimal list (validated from kohya repo)
        SyS("pip install accelerate==0.30.0")
        SyS("pip install bitsandbytes")
        SyS("pip install transformers==4.41.0")
        SyS("pip install ftfy")
        SyS("pip install safetensors")
        SyS("pip install sentencepiece")
        SyS("pip install torchvision")
        SyS("pip install tensorboard")
        SyS("pip install einops")
        SyS("pip install opencv-python")
        SyS("pip install tqdm")
      
    say("<br><b>Kohya requirements installed successfully.</b>")


def kohya_installation(base_path: Path):
    """
    Minimal installation step for Kohya.
    - Only ensures requirements are installed.
    - No extra folders or scripts are downloaded.
    """
    # Define main Kohya paths
    sd_scripts_dir = base_path / "sd-scripts"
    kohya_ss_dir = base_path / "kohya_ss"
    kohya_colab_dir = base_path / "kohya-colab"

    # Install requirements for sd-scripts
    kohya_requirements(base_path)

    print("\n✔ Kohya installation complete.")
    print("Next steps:")
    print(f" - Put your training images in {base_path / 'sd-scripts/dataset'}")
    print(f" - Put your models in {base_path / 'sd-scripts/models'}")
    print(f" - Output will be saved to {base_path / 'sd-scripts/outputs'}")


def notebook_scripts():
    z = [
        (STR / '00-startup.py', f'wget -qO {STR}/00-startup.py https://github.com/gutris1/segsmaker/raw/main/script/KC/00-startup.py'),
        (nenen, f'wget -qO {nenen} https://github.com/gutris1/segsmaker/raw/main/script/nenen88.py'),
        (melon, f'wget -qO {melon} https://github.com/gutris1/segsmaker/raw/main/script/melon00.py'),
        (STR / 'cupang.py', f'wget -qO {STR}/cupang.py https://github.com/gutris1/segsmaker/raw/main/script/cupang.py'),
        (MRK, f'wget -qO {MRK} https://github.com/gutris1/segsmaker/raw/main/script/marking.py')
    ]

    [SyS(y) for x, y in z if not Path(x).exists()]

    j = {'ENVNAME': ENVNAME, 'HOMEPATH': HOME, 'TEMPPATH': TMP, 'BASEPATH': Path(ENVBASE)}
    text = '\n'.join(f"{k} = '{v}'" for k, v in j.items())
    Path(KANDANG).write_text(text)

    key_inject(nenen, civitai_key, hf_read_token)
    marking(SRC, MARKED, selected_ui)
    sys.path.append(str(STR))

    for scripts in [nenen, melon, KANDANG, MRK]: get_ipython().run_line_magic('run', str(scripts))

ENVNAME, ENVBASE, ENVHOME = getENV()

if not ENVNAME:
    print('You are not in Kaggle or Google Colab.\nExiting.')
    sys.exit()

RESET = '\033[0m'
RED = '\033[31m'
PURPLE = '\033[38;5;135m'
ORANGE = '\033[38;5;208m'
ARROW = f'{ORANGE}▶{RESET}'
ERROR = f'{PURPLE}[{RESET}{RED}ERROR{RESET}{PURPLE}]{RESET}'
IMG = 'https://github.com/gutris1/segsmaker/raw/main/script/loading.png'

HOME = Path(ENVHOME)
TMP = Path(ENVBASE) / 'temp'

PY = Path('/GUTRIS1')
SRC = HOME / 'gutris1'
MRK = SRC / 'marking.py'
KEY = SRC / 'api-key.json'
MARKED = SRC / 'marking.json'

USR = Path('/usr/bin')
STR = Path('/root/.ipython/profile_default/startup')
nenen = STR / 'nenen88.py'
melon = STR / 'melon00.py'
KANDANG = STR / 'KANDANG.py'

TMP.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

output = widgets.Output()
loading = widgets.Output()

selected_ui, civitai_key, hf_read_token = getArgs()
if civitai_key is None: sys.exit()

display(output, loading)
with loading: display(Image(url=IMG))
with output: PY.exists() or getPython()
notebook_scripts()

from pathlib import Path
import subprocess

kohya_dir = HOME / "sd-scripts"

if not kohya_dir.exists():
    print("Cloning Kohya sd-scripts repository…")
    subprocess.run(["git", "clone", REPO['Kohya'], str(kohya_dir)], check=True)
else:
    print("Kohya sd-scripts already cloned, pulling latest changes…")
    subprocess.run(["git", "-C", str(kohya_dir), "pull"], check=True)


from pathlib import Path
import subprocess
import sys # <-- Ensure sys is imported here if you move the definition

kohya_dir = HOME / "sd-scripts"

if not kohya_dir.exists():
    print("Cloning Kohya sd-scripts repository…")
    subprocess.run(["git", "clone", REPO['Kohya'], str(kohya_dir)], check=True)
else:
    print("Kohya sd-scripts already cloned, pulling latest changes…")
    subprocess.run(["git", "-C", str(kohya_dir), "pull"], check=True)

# -------------------------------------------------------------
# STEP 1: DEFINE THE CUSTOM PIP WRAPPER FUNCTION
# This function is necessary for running pip commands safely.
# It MUST be defined before it is used.
def pip_install_wrapper(cmd):
    # This uses subprocess to call the python interpreter's pip module
    subprocess.check_call([sys.executable, "-m", "pip"] + cmd.split())
# -------------------------------------------------------------

from nenen88 import clone, say, download, tempe, pull
kohya_installation(HOME)
