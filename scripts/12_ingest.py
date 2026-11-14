"""Ingest of the server results: checks, environment audit, reproduction check, working copies.

Needs the raw record of what the server returned (results/server_raw/), which is not public: it holds the job files with
personal data and cluster paths. In the public repository the outputs of this script (the working copies in results/ and the
tables in reports/step_12_tables/) are committed, so 12_scale.py and 11d_scale_assets.py run without it.
Reads (never writes) that untouched record and the Mac 1.4B main run
(results/scores_1.4b*.parquet, results/delta_1.4b*.parquet, reports/step_07_tables/h1_tests.csv).
Writes:
  reports/step_12_tables/server_raw_manifest.csv  sha256 and size of every received file
  reports/step_12_tables/ingest_checks.csv        per size: files, zip == unzipped, rows, pair set, finiteness, n_tokens vs Mac,
                                                   provenance columns, server consolidation checks / spot check
  reports/step_12_tables/env_audit.csv            server environment vs Mac / server instructions (slurm, versions, GPU, job log)
  reports/step_12_tables/revisions.csv            model snapshot hashes used on the server (1.4B also vs the Mac cache)
  reports/step_12_tables/reproduction_gate.csv    server 1.4B vs Mac 1.4B: mean D per model within 0.04 nat (blocking)
  reports/step_12_tables/reproduction_rows.csv    row-level max / mean |diff| and Pearson r of logp and D
  reports/step_12_tables/h1_cuda_vs_mac.csv       H1 on the server 1.4B vs the main-run table (same conclusions?)
  only if the reproduction check passes, never overwriting:
  results/scores_1.4b_cuda[_base_nobos].parquet, results/scores_<size>[_base_nobos].parquet (2.8b, 6.9b, 12b): byte copies
  results/delta_1.4b_cuda[_base_nobos].parquet, results/delta_<size>[_base_nobos].parquet: D via 05_derived.py functions
The Mac 1.4B files are never touched. Usage: .venv/bin/python scripts/12_ingest.py
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config, local_dir  # noqa: E402
from probing.derived import identity_error  # noqa: E402
from probing.scale import compare_column, h1_verdict, load_script, mean_d_gate, sha256  # noqa: E402

RAW = ROOT / "results/server_raw/2026-10-14_tutor"
TAB = ROOT / "reports/step_12_tables"
SIZES = ["1.4b", "2.8b", "6.9b", "12b"]
TOL = 0.04                      # nat, reproduction check on mean D per model (difference between two runs of the same models)
ALLOWED_CONFIG_KEYS = {"run.size", "paths.models_dir"}   # the only config edits allowed by the server instructions
PAIR = ["prefix_id", "model", "word"]
TABLES = ["gate_checks.csv", "provisional_D_by_model_category.csv", "spot_check.csv", "summary_by_model.csv"]

d05 = load_script(ROOT / "scripts/05_derived.py")
d07 = load_script(ROOT / "scripts/07_rq1.py")


def expected_files(size: str) -> list[str]:
    """Files every size must contain, relative to risultati_<size>/ (same list as the zip command of the server instructions)."""
    return ([f"results/scores_{size}.parquet", f"results/scores_{size}_base_nobos.parquet", "reports/step_00_env.md"]
            + [f"reports/step_04_tables/{t}" for t in TABLES])


def manifest() -> pd.DataFrame:
    """sha256 and size of every received file (macOS .DS_Store excluded)."""
    rows = [dict(path=str(p.relative_to(RAW)), bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != ".DS_Store"]
    return pd.DataFrame(rows)


def zip_matches(size: str) -> tuple[bool, str]:
    """True if the original zip holds exactly the unzipped files, byte for byte."""
    zpath = RAW / "zip" / f"risultati_{size}.zip"
    folder = RAW / f"risultati_{size}"
    with zipfile.ZipFile(zpath) as z:
        names = {n for n in z.namelist() if not n.endswith("/")}
        same = [z.read(n) == (folder / n).read_bytes() if (folder / n).exists() else False for n in sorted(names)]
    unzipped = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name != ".DS_Store"}
    ok = names == unzipped and all(same)
    return ok, f"{sum(same)}/{len(names)} members identical; extra unzipped files: {sorted(unzipped - names)}"


def size_checks(size: str, cfg: dict, mac: pd.DataFrame, mac_nobos: pd.DataFrame) -> list[dict]:
    """Ingest checks for one size: one dict per check (size, set, check, passed, detail)."""
    folder = RAW / f"risultati_{size}"
    rows = []

    def add(set_: str, check: str, passed: bool, detail: object = "") -> None:
        rows.append(dict(size=size, set=set_, check=check, passed=bool(passed), detail=str(detail)))

    missing = [f for f in expected_files(size) if not (folder / f).exists()]
    add("files", "expected files present (2 parquet, step_00_env.md, 4 step_04 tables)", not missing, missing or "7/7")
    ok, detail = zip_matches(size)
    add("files", "original zip == unzipped files (bytes)", ok, detail)

    repos = cfg["models"][size]
    for set_, path, ref, n_exp, bos in [("main", f"results/scores_{size}.parquet", mac, 24_600, True),
                                        ("nobos", f"results/scores_{size}_base_nobos.parquet", mac_nobos, 4_100, False)]:
        df = pd.read_parquet(folder / path)
        add(set_, "row count", len(df) == n_exp, f"{len(df)} (expected {n_exp})")
        add(set_, "pair set (prefix_id, model, word) identical to Mac 1.4B",
            set(map(tuple, df[PAIR].to_numpy())) == set(map(tuple, ref[PAIR].to_numpy())) and not df.duplicated(PAIR).any(), "")
        add(set_, "non-finite logp = 0", np.isfinite(df["logp"]).all(), int((~np.isfinite(df["logp"])).sum()))
        add(set_, "logp <= 0", (df["logp"] <= 0).all(), f"max {df['logp'].max():.4f}")
        m = df.merge(ref, on=PAIR, suffixes=("", "_mac"), validate="one_to_one")
        n_bad = int((m["n_tokens"] != m["n_tokens_mac"]).sum())
        add(set_, "n_tokens per pair == Mac 1.4B (tokenizer check)", n_bad == 0, f"{n_bad} mismatches")
        lab_bad = int(((m["category"] != m["category_mac"]) | (m["polarity_word"] != m["polarity_word_mac"])).sum())
        add(set_, "category and polarity_word per pair == Mac 1.4B", lab_bad == 0, f"{lab_bad} mismatches")
        add(set_, "bos column", (df["bos"] == bos).all(), f"all {bos}" if (df["bos"] == bos).all() else sorted(df["bos"].unique()))
        add(set_, "dtype column float16", (df["dtype"] == "float16").all(), sorted(df["dtype"].unique()))
        fmt_ok = all((df.loc[df["model"] == k, "input_format"] == ("raw" if k == "base" else "chat")).all() for k in df["model"].unique())
        add(set_, "input_format (base raw, aligned chat)", fmt_ok, "")
        repo_ok = all((df.loc[df["model"] == k, "model_repo"] == repos[k]).all() for k in df["model"].unique())
        add(set_, f"model_repo == config models[{size}]", repo_ok, "")

    gates = pd.read_csv(folder / "reports/step_04_tables/gate_checks.csv")
    warn = gates[~gates["passed"].astype(bool)]
    add("server", "gate_checks.csv: all rows passed", warn.empty,
        f"{len(gates)} rows" if warn.empty else "; ".join(f"{r.check}: {r.detail}" for r in warn.itertuples()))
    spot = pd.read_csv(folder / "reports/step_04_tables/spot_check.csv")
    add("server", "spot check: re-scored rows bit-identical", spot["bit_identical"].all(),
        f"{int(spot['bit_identical'].sum())}/{len(spot)}; max |stored - manual| {spot['manual_abs_diff'].max():.2e}")
    return rows


def env_table(path: Path) -> tuple[str, dict]:
    """Python line and package table (section A) of a step_00_env.md report."""
    text = path.read_text()
    py = re.search(r"^Python (\S+)", text, re.M).group(1)
    pk = dict(re.findall(r"^\| (\S+) \| (\S+) \|$", text, re.M))
    pk.pop("package", None)
    return py, pk


def requirements(path: Path) -> dict:
    """name -> version from a pip requirements file (comments skipped, names lowercased)."""
    out = {}
    for line in path.read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            n, v = line.strip().split("==")
            out[n.lower()] = v
    return out


def simulate_config_edit(cfg_path: Path, keys: list[str]) -> bool:
    """Replay the slurm `set_config_value` (yaml load -> set key -> safe_dump -> load) in memory; True if every other value is unchanged."""
    orig = yaml.safe_load(cfg_path.read_text())
    cfg = yaml.safe_load(cfg_path.read_text())
    for k in keys:
        section, field = k.split(".", 1)
        cfg[section][field] = "placeholder"
    again = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
    for k in keys:
        section, field = k.split(".", 1)
        again[section][field] = orig[section][field]
    return again == orig


def job_durations(log: str) -> dict:
    """Minutes per size from the 'Starting size' / 'Finished size' lines of the job log."""
    def when(s: str) -> datetime:
        return datetime.strptime(s.replace(" CEST", ""), "%a %b %d %H:%M:%S %Y")
    start = {m[0]: when(m[1]) for m in re.findall(r"Starting size: (\S+) at (.+)", log)}
    end = {m[0]: when(m[1]) for m in re.findall(r"Finished size: (\S+) at (.+)", log)}
    return {s: round((end[s] - start[s]).total_seconds() / 60, 1) for s in start if s in end}


def env_audit(cfg: dict) -> pd.DataFrame:
    """Server environment and commands vs the Mac run and the server instructions: one row per item (item, reference, server, same, note)."""
    rows = []

    def add(item: str, ref: object, srv: object, same: bool, note: str = "") -> None:
        rows.append(dict(item=item, reference=str(ref), server=str(srv), same=bool(same), note=note))

    mac_py, mac_pk = env_table(ROOT / "reports/step_00_env.md")
    srv_py, srv_pk = env_table(RAW / "risultati_1.4b/reports/step_00_env.md")
    add("Python", mac_py, srv_py, mac_py == srv_py, "server provides Python 3.11 only (tutor's email)")
    for k in mac_pk:
        add(f"package {k} (step_00_env)", mac_pk[k], srv_pk.get(k), mac_pk[k] == srv_pk.get(k))
    lock, srv_req = requirements(ROOT / "requirements-lock.txt"), requirements(RAW / "requirements-cuda.txt")
    ours = requirements(ROOT / "requirements-cuda.txt")
    for k in sorted(set(srv_req) | set(ours)):
        if srv_req.get(k) != ours.get(k) or k in {"torch", "transformers", "tokenizers", "numpy", "safetensors"}:
            add(f"requirement {k}", f"lock {lock.get(k)} / our cuda file {ours.get(k)}", srv_req.get(k), srv_req.get(k) == lock.get(k))
    env_hash = {s: sha256(RAW / f"risultati_{s}/reports/step_00_env.md") for s in SIZES}
    add("step_00_env.md identical in the four zips", "-", f"{len(set(env_hash.values()))} distinct file(s)", len(set(env_hash.values())) == 1,
        "00_check_env ran once, before the size loop (as instructed: once)")
    gpu = re.search(r"^- GPU: (.+)$", (RAW / "risultati_1.4b/reports/step_00_env.md").read_text(), re.M).group(1)
    add("device", "mps (Mac, Apple M1)", f"cuda: {gpu}", False, "config run.device = auto (cuda > mps > cpu); not a parquet column")

    slurm = (RAW / "run_all_sizes.slurm").read_text()
    keys = sorted(set(re.findall(r"^\s*set_config_value (\S+) ", slurm, re.M)))
    add("config.yaml keys edited by the slurm script", sorted(ALLOWED_CONFIG_KEYS), keys, set(keys) <= ALLOWED_CONFIG_KEYS,
        "run.batch_size (1), run.dtype, run.device, bos, chat not edited")
    add("config.yaml rewrite (yaml.safe_dump) keeps every other value", "-", simulate_config_edit(ROOT / "config.yaml", keys),
        simulate_config_edit(ROOT / "config.yaml", keys), "file re-serialised: comments lost, values unchanged (replayed in memory)")
    cmds = sorted(set(re.sub(r"\s+", " ", c).strip() for c in re.findall(r"^\s*((?:srun )?python scripts/[^\n]+)", slurm, re.M)))
    add("commands run", "00_check_env; per size 00b, 04 --smoke, 04, 04b", " | ".join(cmds), True,
        "00b gets --size (same as run.size); 04 full run wrapped in srun")
    order = re.search(r"for size in (.+); do", slurm).group(1).replace('"', "")
    add("size order", "1.4b first, then the larger size", order, order.split()[0] == "1.4b")
    idx = re.findall(r"--index-url (\S+)", slurm)
    add("torch wheel", "PyPI (CUDA 13) or cu126 if driver < 13", idx, True, "cu126 index used unconditionally (allowed by the instructions)")

    log = (RAW / "job_1374680.txt").read_text()
    bad = [ln for ln in log.splitlines() if re.search(r"warn|error|traceback|exception", ln, re.I)]
    add("job log: lines with warning / error / traceback / exception", 0, len(bad), not bad)
    add("job log: minutes per size", "-", job_durations(log), True)
    return pd.DataFrame(rows)


def revisions(cfg: dict) -> pd.DataFrame:
    """Snapshot hash of each checkpoint used on the server (from the 00b lines of the job log); 1.4B also vs the Mac cache."""
    log = (RAW / "job_1374680.txt").read_text()
    rows = []
    for size, block in re.findall(r"Starting size: (\S+) at .+?\n(.+?)Finished size", log, re.S):
        for key, repo, h in re.findall(r"== (\w+): (\S+)\n\s+ready: \S+/snapshots/(\w+)", block):
            mac = local_dir(repo).name if size == "1.4b" else None
            rows.append(dict(size=size, model=key, repo=repo, repo_matches_config=repo == cfg["models"][size][key],
                             server_snapshot=h, mac_snapshot=mac, same_as_mac=(h == mac) if mac else None))
    return pd.DataFrame(rows)


def build_deltas(scores: pd.DataFrame, nobos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefix-level D tables (main and no-BOS reference) with the 05_derived.py functions, plus their internal checks."""
    dm = d05.build_delta_main(scores)
    dn = d05.build_delta_nobos(scores[scores["model"] != "base"], nobos)
    assert len(dm) == 3000 and len(dn) == 2500
    assert identity_error(dm.dropna(subset=["D"])) < 1e-9 and identity_error(dn) < 1e-9
    assert dm.loc[dm["model"] != "base", ["delta_pos", "delta_neg", "D"]].notna().all().all()
    return dm, dn


def copy_new(src: Path, dst: Path) -> None:
    """Byte copy that never overwrites: an existing destination must already be identical."""
    if dst.exists():
        if sha256(dst) != sha256(src):
            raise FileExistsError(f"{dst.name} exists and differs from {src}")
        print(f"{dst.name} already exists and is identical: kept")
        return
    shutil.copyfile(src, dst)


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    seed = int(cfg["project"]["seed"])
    aligned = list(cfg["models"]["1.4b"])[1:]
    TAB.mkdir(parents=True, exist_ok=True)
    mac = pd.read_parquet(res / "scores_1.4b.parquet")
    mac_nobos = pd.read_parquet(res / "scores_1.4b_base_nobos.parquet")

    manifest().to_csv(TAB / "server_raw_manifest.csv", index=False)
    checks = pd.DataFrame([r for s in SIZES for r in size_checks(s, cfg, mac, mac_nobos)])
    checks.to_csv(TAB / "ingest_checks.csv", index=False)
    audit = env_audit(cfg)
    audit.to_csv(TAB / "env_audit.csv", index=False)
    rev = revisions(cfg)
    rev.to_csv(TAB / "revisions.csv", index=False)

    # Reproduction check: server 1.4B vs Mac 1.4B (the Mac D tables are those of 05_derived.py).
    raw14 = RAW / "risultati_1.4b/results"
    cu, cu_nobos = pd.read_parquet(raw14 / "scores_1.4b.parquet"), pd.read_parquet(raw14 / "scores_1.4b_base_nobos.parquet")
    cu_dm, cu_dn = build_deltas(cu, cu_nobos)
    mac_dm, mac_dn = pd.read_parquet(res / "delta_1.4b.parquet"), pd.read_parquet(res / "delta_1.4b_base_nobos.parquet")
    gate = pd.concat([mean_d_gate(mac_dm, cu_dm, aligned, TOL).assign(reference="base_bos"),
                      mean_d_gate(mac_dn, cu_dn, aligned, TOL).assign(reference="base_nobos")], ignore_index=True)
    gate = gate[["reference"] + [c for c in gate.columns if c != "reference"]]
    gate.to_csv(TAB / "reproduction_gate.csv", index=False)
    rows = [dict(quantity="logp, main (6 models)", **compare_column(mac, cu, PAIR, "logp")),
            dict(quantity="logp, Base without BOS", **compare_column(mac_nobos, cu_nobos, PAIR, "logp")),
            dict(quantity="Delta per prefix, main (6 models)", **compare_column(mac_dm, cu_dm, ["model", "prefix_id"], "delta")),
            dict(quantity="D vs Base with BOS (5 models)", **compare_column(mac_dm.dropna(subset=["D"]), cu_dm.dropna(subset=["D"]),
                                                                           ["model", "prefix_id"], "D")),
            dict(quantity="D vs Base without BOS (5 models)", **compare_column(mac_dn, cu_dn, ["model", "prefix_id"], "D"))]
    pd.DataFrame(rows).to_csv(TAB / "reproduction_rows.csv", index=False)
    h1_mac = pd.read_csv(ROOT / "reports/step_07_tables/h1_tests.csv")
    h1_cu = d07.h1_table(cu_dm, cu_dn, aligned, seed)
    h1_mac["verdict"], h1_cu["verdict"] = h1_verdict(h1_mac), h1_verdict(h1_cu)
    cols = ["reference", "model", "mean", "ci_boot_lo", "ci_boot_hi", "d_z", "share_pos", "wilcoxon_p_bonf5", "mcnemar_p", "verdict"]
    h1 = h1_mac[cols].merge(h1_cu[cols], on=["reference", "model"], suffixes=("_mac", "_cuda"), validate="one_to_one")
    h1["same_verdict"] = h1["verdict_mac"] == h1["verdict_cuda"]
    h1.to_csv(TAB / "h1_cuda_vs_mac.csv", index=False)

    pd.set_option("display.width", 200)
    n_fail = int((~checks["passed"]).sum())
    print(f"ingest checks: {len(checks) - n_fail}/{len(checks)} passed over {len(SIZES)} sizes")
    if n_fail:
        print(checks[~checks["passed"]].to_string(index=False))
    print(f"env audit: {int((~audit['same']).sum())} items differ (see env_audit.csv)")
    print(f"revisions 1.4B same as Mac: {rev['same_as_mac'].dropna().astype(bool).sum()}/6; repos match config: {rev['repo_matches_config'].all()}")
    print("\nreproduction check (server 1.4B - Mac 1.4B, mean D per model):")
    print(gate[["reference", "model", "mean_D_ref", "mean_D_new", "diff", "passed"]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"H1 verdicts identical: {int(h1['same_verdict'].sum())}/{len(h1)}")

    if not gate["passed"].all() or n_fail:
        print("\nGATE FAILED (reproduction or ingest): no working copies written, no analysis of other sizes.")
        sys.exit(1)

    # Working copies (byte copies of the record) and D tables; the Mac 1.4B files keep their names and are never touched.
    for size in SIZES:
        tag = "1.4b_cuda" if size == "1.4b" else size
        src = RAW / f"risultati_{size}/results"
        copy_new(src / f"scores_{size}.parquet", res / f"scores_{tag}.parquet")
        copy_new(src / f"scores_{size}_base_nobos.parquet", res / f"scores_{tag}_base_nobos.parquet")
        dm, dn = (cu_dm, cu_dn) if size == "1.4b" else build_deltas(pd.read_parquet(res / f"scores_{tag}.parquet"),
                                                                    pd.read_parquet(res / f"scores_{tag}_base_nobos.parquet"))
        d05.write_new(dm, res / f"delta_{tag}.parquet")
        d05.write_new(dn, res / f"delta_{tag}_base_nobos.parquet")
    print("\nreproduction check passed: working copies and D tables written in results/")


if __name__ == "__main__":
    main()
