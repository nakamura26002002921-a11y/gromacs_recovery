# recovery_agent/agent.py
import json
import os
import time
from .observation import ObservationModule
from .diagnosis import diagnose_error, extract_fatal_error
from .repair import get_repair_candidates

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
        repair_history = []  # op_name(文字列)のリスト
        state_logs = []
        extra_flags = None  # 前回の修復で-ignhなどのフラグが指定された場合に保持

        print(f"--- Starting Recovery for {initial_pdb} ---")

        while attempt < self.max_attempts:
            print(f"\n[Attempt {attempt}] Observing...")
            start_time = time.time()
            obs_result = self.obs_module.run_pdb2gmx(current_pdb, additional_flags=extra_flags)

            # [2] State Representation
            state = {
                "attempt": attempt,
                "current_pdb": current_pdb,
                "success": obs_result["success"],
                "repair_history": list(repair_history),
                # ★追加: 診断の根拠となったFatal errorテキストをそのまま残す
                "fatal_error_text": extract_fatal_error(obs_result["stderr"]),
                # 参考情報として先頭1000文字も残す
                "stderr_head": obs_result["stderr"][:1000], 
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
            selected_fn = None

            for candidate_fn in candidates:
                if candidate_fn.__name__ not in repair_history:
                    selected_fn = candidate_fn
                    break

            state["selected_repair"] = selected_fn.__name__ if selected_fn else None

            if selected_fn is None:
                print(">> No viable repair candidates left. Terminating.")
                state["status"] = "failed_no_candidates"
                self._log_step(state_logs, state, time.time() - start_time)
                break

            # [5] Repair Execution
            print(f">> Executing Repair: {selected_fn.__name__}")
            try:
                result = selected_fn(current_pdb, attempt)
            except Exception as e:
                print(f">> ERROR during repair execution: {e}")
                state["status"] = "error_repair_execution_failed"
                state["error_detail"] = str(e)
                self._log_step(state_logs, state, time.time() - start_time)
                break

            repair_history.append(result["op_name"])

            if result.get("new_pdb_path"):
                current_pdb = result["new_pdb_path"]

            extra_flags = result.get("extra_flags")

            state["repair_extra_flags"] = extra_flags
            state["structure_altered"] = result.get("structure_altered", False)
            state["status"] = "repaired_and_continuing"
            self._log_step(state_logs, state, time.time() - start_time)
            attempt += 1

        else:
            print(">> Max attempts exceeded.")
            state = {
                "attempt": attempt,
                "current_pdb": current_pdb,
                "repair_history": list(repair_history),
                "status": "max_attempts_exceeded",
            }
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
