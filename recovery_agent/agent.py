# recovery_agent/agent.py
import json
import os
import time
import tempfile
import re
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

    def _extract_repair_context(self, state):
        """
        現在の状態（エラーログ）から、修復関数に渡すべき追加情報を抽出する。
        現在は remove_residue_as_last_resort 用の residue_id 抽出のみを行う。
        """
        context = {}
        fatal_text = state.get("fatal_error_text", "")
        if fatal_text:
            # "Residue 196 named ARG ..." から "196" を抽出
            match = re.search(r"Residue (\d+) named", fatal_text)
            if match:
                context["residue_id"] = match.group(1)
        return context

    def run(self, initial_pdb):
        # ★ケースごとに一意な作業ディレクトリを作成
        work_dir = tempfile.mkdtemp(prefix=f"recovery_{os.path.basename(initial_pdb)}_")
        
        current_pdb = initial_pdb
        attempt = 0
        repair_history = []
        state_logs = []
        extra_flags = None

        print(f"--- Starting Recovery for {initial_pdb} ---")
        print(f"Work directory: {work_dir}")

        while attempt < self.max_attempts:
            print(f"\n[Attempt {attempt}] Observing...")
            start_time = time.time()
            
            # work_dirを渡して実行
            obs_result = self.obs_module.run_pdb2gmx(
                current_pdb, 
                work_dir, 
                additional_flags=extra_flags
            )

            state = {
                "attempt": attempt,
                "current_pdb": current_pdb,
                "work_dir": work_dir,
                "success": obs_result["success"],
                "repair_history": list(repair_history),
                "fatal_error_text": extract_fatal_error(obs_result["stderr"]),
                "stderr_head": obs_result["stderr"][:1000],
            }

            if obs_result["success"]:
                print(">> Success! pdb2gmx completed.")
                state["status"] = "success"
                self._log_step(state_logs, state, time.time() - start_time)
                break

            category = diagnose_error(obs_result["stderr"])
            state["diagnosis_category"] = category
            print(f">> Diagnosis: {category}")

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

            print(f">> Executing Repair: {selected_fn.__name__}")
            try:
                # ★コンテキスト（residue_id等）を抽出して渡す
                context = self._extract_repair_context(state)
                
                # work_dirとcontextを渡して修復実行
                result = selected_fn(current_pdb, attempt, work_dir, **context)
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
                "work_dir": work_dir,
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
