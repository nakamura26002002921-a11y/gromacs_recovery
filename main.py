import os
import shutil
import tempfile

import yaml

from recovery_agent.graph import build_graph


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    initial_pdb = "broken_test.pdb"
    work_dir = tempfile.mkdtemp(prefix="recovery_")
    app = build_graph(config)

    init_state = {
        "pdb_path": initial_pdb,
        "work_dir": work_dir,
        "attempt": 0,
        "repair_history": [],
        "extra_flags": [],
    }

    try:
        final_state = app.invoke(init_state, config={"recursion_limit": 100})

        if final_state.get("success"):
            out_dir = config["agent"].get("output_dir", "results")
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, os.path.basename(initial_pdb).replace(".pdb", "_final.pdb"))
            shutil.copy2(final_state["pdb_path"], dest)
            print(f">> Success! Saved final PDB to: {dest}")
        else:
            print(f">> Failed: {final_state.get('status')}")
    finally:
        if not config["agent"].get("keep_work_dir", False):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
