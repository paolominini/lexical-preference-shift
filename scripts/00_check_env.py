#!/usr/bin/env python3
"""
00_check_env.py - can this machine run the experiment, and are the models there?

Checks (each failure is reported, none stops the script):
  A. Python packages and versions
  B. Hardware: RAM, free disk, GPU (CUDA or Apple MPS) and its memory budget
  C. Model inventory for config `run.size`: found locally? size on disk? does the repo exist on the Hub?
  D. Memory fit: one row per size in config (weights, estimated GPU need, disk for six checkpoints, fits here?)
  E. Tokenizer facts for the models found locally (BOS, role tokens, chat template)
  F. Optional (--time): load the smallest local model and time one forward pass

Usage (from the project root):
    python scripts/00_check_env.py [--time] [--models-dir PATH]
Writes reports/step_00_env.md and prints a short summary.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["torch", "transformers", "accelerate", "huggingface_hub", "safetensors", "pandas", "pyarrow", "numpy",
            "scipy", "statsmodels", "matplotlib", "seaborn", "openpyxl", "yaml", "tqdm", "detoxify", "pytest"]
GIB = 1024 ** 3  # all sizes in this script are GiB; the Hugging Face website shows decimal GB (about 7% larger)
EST_GIB = {"1.4b": 2.73, "2.8b": 5.46, "6.9b": 12.9, "12b": 22.2}   # largest checkpoint per size, measured on the Hub (offline fallback)


def load_config() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def pkg_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            out[name] = f"MISSING ({type(e).__name__})"
    return out


def total_ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / GIB
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
    except Exception:  # noqa: BLE001
        return None


def gpu_info() -> dict:
    """CUDA first, then Apple MPS: device kind, name and the memory PyTorch can use on it."""
    info = {"kind": None, "name": None, "budget_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(kind="cuda", name=f"{props.name} (x{torch.cuda.device_count()})", budget_gb=props.total_memory / GIB)
        elif torch.backends.mps.is_available():
            info.update(kind="mps", name="Apple GPU (MPS)")
            fn = getattr(torch.mps, "recommended_max_memory", None)
            if fn:
                info["budget_gb"] = fn() / GIB
    except Exception:  # noqa: BLE001
        pass
    return info


def find_local(repo_id: str, models_dir: str | None) -> tuple[Path | None, float | None]:
    """Locate a downloaded model: explicit folder first, then the Hugging Face cache."""
    name = repo_id.split("/")[-1]
    if models_dir:
        for cand in (Path(models_dir) / name, Path(models_dir) / repo_id.replace("/", "--"),
                     Path(models_dir) / f"models--{repo_id.replace('/', '--')}"):
            if cand.exists():
                return cand, folder_size_gb(cand)
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                return Path(repo.repo_path), repo.size_on_disk / GIB
    except Exception:  # noqa: BLE001
        pass
    return None, None


def resolve_dir(p: Path) -> str:
    """If p is a Hugging Face cache folder (models--org--name), return its newest snapshot folder."""
    snaps = p / "snapshots"
    if snaps.exists():
        subs = sorted((d for d in snaps.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
        if subs:
            return str(subs[-1])
    return str(p)


def folder_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / GIB  # HF snapshots link to blobs


def hub_info(repo_id: str) -> tuple[str, float | None]:
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / GIB
        return f"exists on the Hub (weights {size:.2f} GiB)", size
    except Exception as e:  # noqa: BLE001
        return f"could not verify ({type(e).__name__}: no internet, or repo not found)", None


def models_cache(models_dir: str | None) -> Path:
    """Folder the checkpoints are (or will be) downloaded to; the first existing parent if it does not exist yet."""
    from huggingface_hub import constants
    p = Path(models_dir or constants.HF_HUB_CACHE)
    while not p.exists():
        p = p.parent
    return p


def fit_verdict(weights_gb: float, budget_gb: float | None, ram_gb: float | None) -> str:
    budget = budget_gb or (0.6 * ram_gb if ram_gb else None)
    if budget is None:
        return "unknown (could not read memory budget)"
    ratio = weights_gb / budget
    tag = "OK" if ratio < 0.6 else "TIGHT" if ratio < 0.9 else "DOES NOT FIT"
    return f"{tag}: weights {weights_gb:.2f} GiB vs usable budget ~{budget:.1f} GiB (ratio {ratio:.2f})"


def tokenizer_facts(path_or_id: str, local_only: bool) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_id, local_files_only=local_only)
    test = "The quick brown fox"
    ids = tok(test)["input_ids"]
    def single(t: str) -> str:
        i = tok.convert_tokens_to_ids(t)
        return "no" if i is None or i == tok.unk_token_id else f"yes (id {i})"
    return {
        "vocab_size": len(tok), "bos_token": tok.bos_token, "eos_token": tok.eos_token,
        "ids('" + test + "')": ids[:6], "first_id_is_bos": bool(ids and tok.bos_token_id is not None and ids[0] == tok.bos_token_id),
        "<|user|> single token": single("<|user|>"), "<|assistant|> single token": single("<|assistant|>"),
        "chat_template": "yes" if getattr(tok, "chat_template", None) else "no",
    }


def time_forward(model_dir: str, device: str, dtype: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    try:  # transformers >= 4.56 / 5.x
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, local_files_only=True)
    except TypeError:  # older versions
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dt, local_files_only=True)
    model = model.to(device).eval()
    load_s = time.time() - t0
    x = tok(["The quick brown fox jumps over the lazy dog. " * 4] * 8, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**x)  # warm-up
        t1 = time.time()
        for _ in range(3):
            logits = model(**x).logits
            logits.float().log_softmax(-1)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        step_s = (time.time() - t1) / 3
    mem = (torch.mps.current_allocated_memory() / GIB if device == "mps"
           else torch.cuda.max_memory_allocated() / GIB if device == "cuda" else None)
    return {"load_s": round(load_s, 1), "forward_s_batch8x%dtok" % x["input_ids"].shape[1]: round(step_s, 2),
            "gpu_memory_gib": None if mem is None else round(mem, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--time", action="store_true", help="load the smallest local model and time a forward pass")
    ap.add_argument("--no-hub", action="store_true", help="skip Hub lookups")
    a = ap.parse_args()
    cfg = load_config()
    size = cfg["run"]["size"]
    models_dir = a.models_dir or cfg["paths"].get("models_dir")
    L: list[str] = [f"# Environment check ({size})", "", f"Run at {time.strftime('%Y-%m-%d %H:%M')}", ""]
    summary: list[str] = []

    # A
    vers = pkg_versions()
    missing = [k for k, v in vers.items() if v.startswith("MISSING")]
    L += ["## A. Packages", "", f"Python {platform.python_version()} on {platform.platform()} ({platform.machine()})", "",
          "| package | version |", "|---|---|"] + [f"| {k} | {v} |" for k, v in vers.items()] + [""]
    summary.append("packages: " + ("all present" if not missing else "MISSING " + ", ".join(missing)))

    # B
    ram = total_ram_gb()
    disk_free = shutil.disk_usage(ROOT).free / GIB
    cache = models_cache(models_dir)
    cache_free = shutil.disk_usage(cache).free / GIB
    gpu = gpu_info()
    L += ["## B. Hardware", "", f"- RAM: {ram:.1f} GiB" if ram else "- RAM: unknown",
          f"- Free disk (project drive): {disk_free:.0f} GiB",
          f"- Free disk where the models go ({cache}): {cache_free:.0f} GiB",
          f"- GPU: {gpu['name'] or 'none (CPU only)'}",
          f"- GPU memory budget reported by PyTorch: {gpu['budget_gb']:.1f} GiB" if gpu["budget_gb"] else
          "- GPU memory budget: not reported (torch missing, or no GPU)", ""]
    summary.append(f"RAM {ram:.0f} GB, models disk free {cache_free:.0f} GB, GPU {gpu['name']}" if ram else "hardware: partial")

    # C + D
    L += [f"## C. Models ({size})", "", "| key | repo | local | on disk (GiB) | Hub |", "|---|---|---|---|---|"]
    found: dict[str, Path] = {}
    to_download = 0.0
    hub_sizes: list[float] = []
    hub: dict[str, tuple[str, float | None]] = {}
    for s in cfg["models"]:
        for repo in cfg["models"][s].values():
            hub[repo] = ("skipped", None) if a.no_hub else hub_info(repo)
    default_gib = EST_GIB.get(size, 5.46)
    for key, repo in cfg["models"][size].items():
        path, gb = find_local(repo, models_dir)
        hub_txt, hub_gib = hub[repo]
        if hub_gib:
            hub_sizes.append(hub_gib)
        if path:
            found[key] = path
        else:
            to_download += hub_gib or default_gib
        gb_txt = f"{gb:.2f}" if gb else "-"
        L.append(f"| {key} | `{repo}` | {'yes' if path else 'NO'} | {gb_txt} | {hub_txt} |")
    L.append("")
    summary.append(f"models found locally: {len(found)}/6")
    if to_download:
        L.append(f"To download: about {to_download:.0f} GiB (free disk {disk_free:.0f} GiB: "
                 f"{'enough' if cache_free > to_download * 1.2 else 'NOT ENOUGH'}). Run `python scripts/00b_download_models.py`.\n")
    weights = max(hub_sizes) if hub_sizes else default_gib
    L += ["## D. Memory fit", "", f"- fp16 weights of the largest {size} checkpoint (config run.size): {weights:.2f} GiB"
          + (" (from the Hub)" if hub_sizes else " (estimate: Hub not reachable)"),
          f"- Verdict: {fit_verdict(weights, gpu['budget_gb'], ram)}", "",
          "All sizes in config (GPU need = largest checkpoint / 0.6, the margin used by the verdict; "
          "disk = six checkpoints, +20%). Set `run.size` in config.yaml to the largest size that fits.", "",
          "| size | largest checkpoint (GiB) | GPU need (GiB) | disk for six (GiB) | fits GPU here | fits disk here |",
          "|---|---|---|---|---|---|"]
    for s, repos in cfg["models"].items():
        sizes = [hub[r][1] for r in repos.values() if hub[r][1]]
        big = max(sizes) if sizes else EST_GIB.get(s, float("nan"))
        six = sum(sizes) if len(sizes) == len(repos) else 6 * big
        need = big / 0.6
        gpu_ok = "unknown" if not gpu["budget_gb"] else "yes" if gpu["budget_gb"] >= need else "no"
        disk_ok = "yes" if cache_free >= six * 1.2 else "no"
        L.append(f"| {s} | {big:.1f}{'' if sizes else ' (estimate)'} | {need:.0f} | {six * 1.2:.0f} | {gpu_ok} | {disk_ok} |")
        summary.append(f"size {s}: GPU need ~{need:.0f} GiB ({gpu_ok}), disk ~{six * 1.2:.0f} GiB ({disk_ok})")
    L.append("")
    summary.append("fit: " + fit_verdict(weights, gpu["budget_gb"], ram))

    # E
    L += ["## E. Tokenizer facts (models found locally)", ""]
    if not found:
        L += ["No local model found: nothing to inspect.", ""]
    for key, path in found.items():
        try:
            repo = cfg["models"][size][key]
            facts = tokenizer_facts(repo if not models_dir else resolve_dir(path), local_only=True)
            L += [f"**{key}**", ""] + [f"- {k}: {v}" for k, v in facts.items()] + [""]
        except Exception as e:  # noqa: BLE001
            L += [f"**{key}**: could not load tokenizer ({type(e).__name__}: {str(e)[:120]})", ""]

    # F
    if a.time and found:
        key = next(iter(found))
        device = cfg["run"]["device"]
        if device == "auto":
            device = gpu["kind"] or "cpu"
        try:
            res = time_forward(cfg["models"][size][key] if not models_dir else resolve_dir(found[key]), device, cfg["run"]["dtype"])
            L += [f"## F. Forward-pass timing ({key}, device {device}, {cfg['run']['dtype']})", ""] + [f"- {k}: {v}" for k, v in res.items()] + [""]
            summary.append(f"timing OK on {device}: {res}")
        except Exception as e:  # noqa: BLE001
            L += [f"## F. Timing FAILED on {device}: {type(e).__name__}: {str(e)[:200]}", ""]
            summary.append(f"timing FAILED ({type(e).__name__})")

    out = ROOT / cfg["paths"]["reports"] / "step_00_env.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join("- " + s for s in summary))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
