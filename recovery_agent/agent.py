import json
import os
import time
from .observation import ObservationModule
from .diagnosis import diagnose_error
from .repair import get_repair_candidates, pdbfixer_add_missing_atoms

class RecoveryAgent:
    def __init__(self, config):
        self.max_attempts = config["agent"]["max_attempts"]
        self.log_dir = config["agent"]["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.obs_module = ObservationModule(
            force_field=config["gromacs"]["force_field"],
            water_model=config["gromacs"]["water_model"]
        )

    def run(self, initial_pdb):
        current_pdb = initial_pdb
        attempt = 0
        repair_history = []
        state_logs = []

        print(f"--- Starting Recovery for {initial_pdb} ---")

        while attempt < self.max_attempts:
            print(f"\n[Attempt {attempt}] Observing...")
            start_time = time.time()
            obs_result = self.obs_module.run_pdb2gmx(current_pdb)
            
            # [2] State Representation (simple ver)
            state = {
                "attempt": attempt,
                "current_pdb": current_pdb,
                "success": obs_result["success"],
                "repair_history": list(repair_history)
            }

            if obs_result["success"]:
                print(">> Success! pdb2gmx completed.")
                state["status"] = "success"
                self._log_step(state_logs, state, time.time() - start_time)
                break

            # [3] Diagnosis
            category = diagnose_error(obs_result["stderr"])
            state["diagnosis_category"] = category
            print(f">> Diagnosis: {category}")

            # [4] Repair Strategy Selector
            candidates = get_repair_candidates(category)
            selected_repair = None
            
            for candidate in candidates:
                if candidate not in repair_history:
                    selected_repair = candidate
                    break
            
            state["selected_repair"] = selected_repair

            if not selected_repair:
                print(">> No viable repair candidates left. Terminating.")
                state["status"] = "failed_no_candidates"
                self._log_step(state_logs, state, time.time() - start_time)
                break

            if selected_repair in repair_history:
                print(">> ERROR: Logic fault. Duplicate repair suggested. Force stopping.")
                state["status"] = "error_duplicate_repair"
                self._log_step(state_logs, state, time.time() - start_time)
                break

            # [5] Repair Execution
            print(f">> Executing Repair: {selected_repair}")
            if selected_repair == "pdbfixer_add_missing_atoms":
                current_pdb, op_name = pdbfixer_add_missing_atoms(current_pdb, attempt)
                repair_history.append(op_name)

            state["status"] = "repaired_and_continuing"
            self._log_step(state_logs, state, time.time() - start_time)
            attempt += 1

        else:
            print(">> Max attempts exceeded.")
            state["status"] = "max_attempts_exceeded"
            self._log_step(state_logs, state, 0)

        self._save_jsonlines(initial_pdb, state_logs)
        return state_logs

    def _log_step(self, logs, state, duration):
        state["duration_sec"] = round(duration, 2)
        logs.append(state)

    def _save_jsonlines(self, initial_pdb, logs):
        filename = os.path.basename(initial_pdb).replace(".pdb", "_recovery.jsonl")
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            for log in logs:
                f.write(json.dumps(log) + "\n")
