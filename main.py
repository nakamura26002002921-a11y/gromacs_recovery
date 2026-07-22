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

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb_from_rcsb(pdb_id, save_dir, timeout=60):
    """RCSBからPDBファイルをダウンロードする。"""
    pdb_id = pdb_id.upper().strip()
    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    out_path = os.path.join(save_dir, f"{pdb_id}.pdb")

    if os.path.exists(out_path):
        logging.info(f"PDB already cached: {out_path}")
        return out_path

    logging.info(f"Downloading PDB from RCSB: {url}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to download PDB {pdb_id} from RCSB: {e}")
        raise RuntimeError(
            f"PDB {pdb_id} のダウンロードに失敗しました。"
            f"PDB IDが正しいか、ネットワーク接続を確認してください。"
        ) from e

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    logging.info(f"Downloaded PDB saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="GROMACS Recovery Agent - PDB修復パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py --pdb-id 1AON
  python main.py --pdb-id 1AON --pdb local_file.pdb
  python main.py                          # デフォルト: 1XYZ
        """,
    )
    parser.add_argument(
        "--pdb-id", default="1XYZ",
        help="RCSB PDB ID (デフォルト: 1XYZ)。RCSBからPDBとFASTAをダウンロードする。",
    )
    parser.add_argument(
        "--pdb", default=None,
        help="ローカルPDBファイルのパス。指定するとダウンロードせずにこのファイルを使用。",
    )
    parser.add_argument(
        "--config", default=None,
        help="設定ファイルのパス (デフォルト: ./config.yaml)",
    )
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper().strip()

    logging.info("=" * 60)
    logging.info("  GROMACS Recovery Agent started.")
    logging.info(f"  PDB ID: {pdb_id}")
    logging.info("=" * 60)

    # --- 設定読み込み ---
    config_path = args.config or os.path.join(BASE_DIR, "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # --- ワークディレクトリ作成 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(LOG_DIR, f"work_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)
    logging.info(f"Working directory: {work_dir}")

    # --- 入力PDBの決定 ---
    if args.pdb:
        if not os.path.exists(args.pdb):
            logging.error(f"Local PDB file not found: {args.pdb}")
            sys.exit(1)
        initial_pdb = os.path.join(work_dir, "input.pdb")
        shutil.copy2(args.pdb, initial_pdb)
        logging.info(f"Using local PDB: {args.pdb} → {initial_pdb}")
    else:
        try:
            initial_pdb = download_pdb_from_rcsb(pdb_id, work_dir)
        except RuntimeError as e:
            logging.error(str(e))
            sys.exit(1)

    # --- グラフ構築・実行 ---
    app = build_graph(config)

    init_state = {
        "pdb_path": initial_pdb,
        "pdb_id": pdb_id,
        "work_dir": work_dir,
        "attempt": 0,
        "repair_history": [],
        "extra_flags": [],
    }

    try:
        logging.info("Starting recovery process...")
        final_state = app.invoke(init_state, config={"recursion_limit": 100})

        if final_state.get("success"):
            out_dir = config["agent"].get("output_dir", "results")
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, f"{pdb_id}_final.pdb")
            shutil.copy2(final_state["pdb_path"], dest)
            logging.info(f"✅ Success! Saved final PDB to: {dest}")
        else:
            logging.error(f"❌ Failed: {final_state.get('status')}")
            if "stderr" in final_state:
                logging.error(f"Last stderr: {final_state['stderr'][-500:]}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

    finally:
        if not config["agent"].get("keep_work_dir", False):
            logging.info(f"Cleaning up working directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            logging.info(f"Keeping working directory: {work_dir}")


if __name__ == "__main__":
    main()
