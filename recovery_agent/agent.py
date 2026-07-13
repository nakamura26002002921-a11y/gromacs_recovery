# recovery_agent/agent.py
import json
import os
import time
import tempfile
import re
import shutil
from .observation import ObservationModule
from .diagnosis import diagnose_error, extract_fatal_error
from .repair import (
    get_repair_candidates,
    count_missing_residues,
    rfdiffusion_rebuild_missing_loops,
    RFDIFFUSION_MISSING_RESIDUE_THRESHOLD,
)
from .utils import run_with_timeout

class RecoveryAgent:
    def __init__(self, config):
        self.config = config
        self.max_attempts = config["agent"]["max_attempts"]
        self.log_dir = config["agent"]["log_dir"]
        self.keep_work_dir = config["agent"].get("keep_work_dir", False)
        self.repair_timeout = config["agent"].get("repair_timeout_sec", 300)

        # RFdiffusionによるループ再構築の設定
        rf_conf = config.get("rfdiffusion", {}) or {}
        self.rfdiffusion_threshold = rf_conf.get(
            "missing_residue_threshold", RFDIFFUSION_MISSING_RESIDUE_THRESHOLD
        )
        self.rfdiffusion_script = rf_conf.get("run_inference_script")
        self.rfdiffusion_timeout = rf_conf.get("timeout_sec")
        
        # ★追加: 完成したPDBの保存先ディレクトリ
        self.output_dir = config["agent"].get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        os.makedirs(self.log_dir, exist_ok=True)

        self.obs_module = ObservationModule(
            force_field=config["gromacs"]["force_field"],
            water_model=config["gromacs"]["water_model"]
        )

    def _extract_repair_context(self, state):
        context = {}
        fatal_text = state.get("fatal_error_text", "")
        if not fatal_text: 
            return context
        match_res = re.search(r"Residue (\d+) named", fatal_text)
        if match_res: 
            context["residue_id"] = match_res.group(1)
        else:
            match_res2 = re.search(r"residue [A-Z]+ (\d+)", fatal_text)
            if match_res2: 
                context["residue_id"] = match_res2.group(1)
        match_chain = re.search(r"Chain ([A-Z])", fatal_text)
        if match_chain: 
            context["chain_id"] = match_chain.group(1)
        match_res_name = re.search(r"Residue '(\w+)' not found in residue topology database", fatal_text)
        if match_res_name:
            context["missing_residue_name"] = match_res_name.group(1)
            
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

                category = diagnose_error(obs_result["stderr"])
                state["diagnosis_category"] = category
                print(f">> Diagnosis: {category}")

                # 診断カテゴリに関わらず、まず構造そのものに大きな欠損
                # (ループ全体の欠落)がないかをチェックする。
                # pdbfixerの単純な幾何学的補間は6残基以上のギャップでは
                # 立体構造が破綻しやすく、巨大な複合体では処理時間も
                # 膨れ上がる(実測: 1AONでタイムアウト)ため、閾値以上の
                # 欠損はRFdiffusionでの再構築を優先する。
                try:
                    total_missing, _ = count_missing_residues(current_pdb)
                except Exception as e:
                    total_missing = 0
                    print(f">> WARNING: missing-residue check failed: {e}")

                state["missing_residue_count"] = total_missing

                selected_fn = None
                if (total_missing >= self.rfdiffusion_threshold
                        and rfdiffusion_rebuild_missing_loops.__name__ not in repair_history):
                    selected_fn = rfdiffusion_rebuild_missing_loops
                    print(
                        f">> {total_missing} missing residues detected "
                        f"(>= {self.rfdiffusion_threshold}); routing to RFdiffusion"
                    )
                else:
                    candidates = get_repair_candidates(category)
                    for candidate_fn in candidates:
                        if candidate_fn.__name__ not in repair_history:
                            selected_fn = candidate_fn
                            break

                state["selected_repair"] = selected_fn.__name__ if selected_fn else None

                # 試すべき候補がまだ残っている限りは、同じエラー文言が
                # 連続していても諦めずに次の候補を試す。
                # (repair_historyにより同じ関数が二度呼ばれることはないため、
                #  候補は必ず有限個で尽きるので無限ループにはならない)
                if selected_fn is None:
                    print(">> No viable repair candidates left. Terminating.")
                    state["status"] = "failed_no_candidates"
                    self._log_step(state_logs, state, time.time() - start_time)
                    break

                print(f">> Executing Repair: {selected_fn.__name__}")
                try:
                    context = self._extract_repair_context(state)
                    outer_timeout = self.repair_timeout
                    if selected_fn is rfdiffusion_rebuild_missing_loops:
                        context["run_inference_script"] = self.rfdiffusion_script
                        context["timeout_sec"] = self.rfdiffusion_timeout
                        # RFdiffusionは通常のpdbfixer系修復より遥かに時間が
                        # かかるため、外側(プロセス強制終了用)のタイムアウトも
                        # rfdiffusion.timeout_secに合わせて延長する。
                        # そうしないと内部のsubprocess.runより先に、この外側の
                        # multiprocessingタイムアウトで強制終了されてしまう。
                        outer_timeout = max(
                            self.repair_timeout,
                            self.rfdiffusion_timeout or 1800,
                        ) + 60  # サブプロセス終了処理の余裕を持たせる
                    result = run_with_timeout(
                        selected_fn, 
                        args=(current_pdb, attempt, work_dir), 
                        kwargs=context, 
                        timeout_sec=outer_timeout
                    )
                except Exception as e:
                    print(f">> ERROR during repair execution: {e}")
                    state["status"] = "error_repair_execution_failed"
                    state["error_detail"] = str(e)
                    self._log_step(state_logs, state, time.time() - start_time)
                    break

                if result.get("status") in ["repair_timeout", "repair_error"]:
                    print(f">> Repair {result['status']}: {result.get('error')}")
                    # この修復方法は使えなかったとして履歴に記録し、
                    # 次のattemptで他の候補(pdbfixer等)にフォールバックする。
                    # RFdiffusionが未インストール/GPU無し等で失敗しても、
                    # そこで全体を諦めず、既存の修復手段を試せるようにするため。
                    repair_history.append(selected_fn.__name__)
                    state["status"] = result["status"]
                    state["error_detail"] = result.get("error")
                    self._log_step(state_logs, state, time.time() - start_time)
                    attempt += 1
                    continue

                repair_history.append(result["op_name"])
                if result.get("new_pdb_path"):
                    current_pdb = result["new_pdb_path"]
                # 既存のextra_flags(-ignhなど)を消さずに積み重ねる。
                # 上書きすると、以前の修復で有効になったフラグが
                # 後続の(フラグを返さない)修復によって失われてしまう。
                new_flags = result.get("extra_flags")
                if new_flags:
                    merged = list(dict.fromkeys((extra_flags or []) + new_flags))
                    extra_flags = merged

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
