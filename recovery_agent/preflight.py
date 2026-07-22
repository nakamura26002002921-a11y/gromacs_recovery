# recovery_agent/preflight.py
import os
import sys
import shutil
import logging
import requests


def preflight(config, pdb_id):
    """実行前に全ての前提条件を検証する。1つでも失敗したら即終了。"""
    errors = []

    if shutil.which("gmx") is None:
        errors.append("gmx (GROMACS) がPATHに見つかりません。")

    rf = config.get("rfdiffusion", {})
    script = rf.get("script_path", "")
    if script and not os.path.isfile(script):
        errors.append(f"RFdiffusion script_path が存在しません: {script}")
    model_dir = rf.get("model_directory_path", "")
    if model_dir and not os.path.isdir(model_dir):
        errors.append(f"RFdiffusion model_directory_path が存在しません: {model_dir}")

    mod = config.get("modeller", {})
    if mod.get("enabled", True):
        key = mod.get("license_key", "")
        if not key or key == "YOUR-MODELLER-LICENSE-KEY":
            errors.append("MODELLER license_key が未設定です。config.yaml に設定するか enabled: false にしてください。")

    try:
        r = requests.head(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=10)
        if r.status_code != 200:
            errors.append(f"RCSB から PDB {pdb_id} を取得できません (HTTP {r.status_code})。")
    except requests.RequestException as e:
        errors.append(f"RCSB への接続に失敗しました: {e}")

    if errors:
        logging.error("=" * 60)
        logging.error("  Preflight check FAILED")
        logging.error("=" * 60)
        for e in errors:
            logging.error(f"  ✗ {e}")
        sys.exit(1)

    logging.info("Preflight check: ALL PASSED ✓")
