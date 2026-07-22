# main.py
import os
import sys
import shutil
import logging
import argparse
import requests
from datetime import datetime
import yaml

from recovery_agent.graph import build_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "recovery.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)


def download_pdb(pdb_id, save_dir):
    out_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(out_path, "w") as f:
        f.write(resp.text)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="GROMACS Recovery Agent")
    parser.add_argument("--pdb-id", default="1XYZ", help="RCSB PDB ID (default: 1XYZ)")
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper().strip()

    with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    work_dir = os.path.join(LOG_DIR, f"work_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        initial_pdb = download_pdb(pdb_id, work_dir)
    except Exception as e:
        logging.error(f"PDB download failed: {e}")
        sys.exit(1)

    logging.info(f"PDB ID: {pdb_id} | Work dir: {work_dir}")

    app = build_graph(config)
    final_state = app.invoke({
        "pdb_path": initial_pdb,
        "pdb_id": pdb_id,
        "work_dir": work_dir,
        "attempt": 0,
        "repair_history": [],
        "extra_flags": [],
    }, config={"recursion_limit": 100})

    if final_state.get("success"):
        out_dir = config["agent"].get("output_dir", "results")
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, f"{pdb_id}_final.pdb")
        shutil.copy2(final_state["pdb_path"], dest)
        logging.info(f"✅ Success: {dest}")
    else:
        logging.error(f"❌ Failed: {final_state.get('status')}")

    if not config["agent"].get("keep_work_dir", False):
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
