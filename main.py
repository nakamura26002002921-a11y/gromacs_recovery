import os
import sys
import shutil
import logging
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

def main():
    logging.info("GROMACS Recovery Agent started.")
    config_path = os.path.join(BASE_DIR, "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    initial_pdb = "broken_test.pdb"
    pdb_id = config.get("rfdiffusion", {}).get("pdb_id")
    if not pdb_id:
        logging.error(
            " 正しいPDB IDをconfig.yamlで明示指定してください。"
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(LOG_DIR, f"work_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)
    logging.info(f"Working directory created: {work_dir}")
    
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
            dest = os.path.join(out_dir, os.path.basename(initial_pdb).replace(".pdb", "_final.pdb"))
            shutil.copy2(final_state["pdb_path"], dest)
            logging.info(f"Success! Saved final PDB to: {dest}")
        else:
            logging.error(f"Failed: {final_state.get('status')}")
            if "stderr" in final_state:
                logging.error(f"Last stderr: {final_state['stderr']}")
                
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        
    finally:
        if not config["agent"].get("keep_work_dir", False):
            logging.info(f"Cleaning up working directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            logging.info(f"Keeping working directory as requested: {work_dir}")

if __name__ == "__main__":
    main()
