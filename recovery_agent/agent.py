# recovery_agent/agent.py
import json
import os
import time
import tempfile
import re
import shutil
from .observation import ObservationModule
from .diagnosis import diagnose_error, extract_fatal_error
from .repair import get_repair_candidates
from .utils import run_with_timeout

class RecoveryAgent:
    def __init__(self, config):
        self.config = config
        self.max_attempts = config["agent"]["max_attempts"]
        self.log_dir = config["agent"]["log_dir"]
        self.keep_work_dir = config["agent"].get("keep_work_dir", False)
        self.repair_timeout = config["agent"].get("repair_timeout_sec", 300)
        
        # ★追加: 完成したPDBの保存先ディレクトリ
        self.output_dir = config["agent"].get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        os.makedirs(self.log_dir, exist_ok=True)

        self.obs_module = ObservationModule(
            force_field=config["gromacs"]["force_field"],
            water_model=config["gromacs"]["water_model"]
        )

    def _extract_repair_context(self, state):
        # ... (前回と同じ) ...
        context = {}
        fatal_text = state.get("fatal_error_text", "")
        if not fatal_text: return context
        match_res = re.search(r"Residue (\d+) named", fatal_text)
        if match_res: context["residue_id"] = match_res.group(1)
        else:
            match_res2 = re.search(r"residue [A-Z]+ (\d+)", fatal_text)
            if match_res2: context["residue_id"] = match_res2.group(1)
        match_chain = re.search(r"Chain ([A-Z])", fatal_text)
        if match_chain: context["chain_id"] = match_chain.group(1)
        return context

    # ★追加: 最終PDBを保存するメソッド
    def _save_final_pdb(self, current_pdb, initial_pdb):
        """成功した最終PDBをoutput_dirにコピーする"""
        base_name = os.path.splitext(os.path.basename(initial_pdb))[0]
        dest_filename = f"{base_name}_final.pdb"
        dest_path = os.path.join(self.output_dir, dest_filename)
        
        # work_dir内のファイルでも絶対パスに変換してコピー
        shutil.copy2(os.path.abspath(current_pdb), dest_path)
        print(f">> Saved final PDB to: {dest_path}")

    def run(self, initial_pdb):
        work_dir = tempfile.mkdtemp(prefix=f"recovery_{os.path.basename(initial_pdb)}_")
        
        current_pdb = initial_pdb
        attempt = 0
        repair_history = []
        state_logs = []
        extra_flags = None
        previous_fatal_error = None

        print(f"--- Starting Recovery for {initial_pdb} ---")
        print(f"Work directory: {work_dir}")

        try:
            while attempt < self.max_attempts:
                print(f"\n[Attempt {attempt}] Observing...")
                start_time = time.time()
                
                obs_result = self.obs_module.run_pdb2gmx(
                    current_pdb, work_dir, additional_flags=extra_flags
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
                    
                    # ★成功時に最終PDBをoutput_dirに保存
                    self._save_final_pdb(current_pdb, initial_pdb)
                    break

                current_fatal_error = state.get("fatal_error_text")

                if attempt > 0 and current_fatal_error and current_fatal_error == previous_fatal_error:
                    print(">> ERROR: Last repair had no effect. Terminating.")
                    state["status"] = "failed_no_progress"
                    self._log_step(state_logs, state, time.time() - start_time)
                    break
                
                previous_fatal_error = current_fatal_error

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
                    context = self._extract_repair_context(state)
                    result = run_with_timeout(
                        selected_fn, 
                        args=(current_pdb, attempt, work_dir), 
                        kwargs=context, 
                        timeout_sec=self.repair_timeout
                    )
                except Exception as e:
                    print(f">> ERROR during repair execution: {e}")
                    state["status"] = "error_repair_execution_failed"
                    state["error_detail"] = str(e)
                    self._log_step(state_logs, state, time.time() - start_time)
                    break

                if result.get("status") in ["repair_timeout", "repair_error"]:
                    print(f">> Repair {result['status']}")
                    state["status"] = result["status"]
                    state["error_detail"] = result.get("error")
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

        finally:
            if not self.keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

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
