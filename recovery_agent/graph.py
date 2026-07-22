# recovery_agent/graph.py
import os
import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from .observation import ObservationModule
from .diagnosis import diagnose_error, extract_fatal_error, extract_local_residue_info
from .repair import get_repair_candidates
from .utils import run_with_timeout
from .missing_residues import count_missing_residues
from .rfdiffusion_repair import run_rfdiffusion
from .sequence_recovery import apply_sequence_recovery
from .modeller_minimize import minimize_with_modeller


class RecoveryState(TypedDict, total=False):
    pdb_path: str
    pdb_id: str
    work_dir: str
    attempt: int
    repair_history: List[str]
    extra_flags: List[str]
    stderr: str
    success: bool
    status: str
    original_pdb_path: str
    missing_count: int


def _extract_context(fatal_text):
    context = {}
    if not fatal_text:
        return context
    m = re.search(r"Residue (\d+) named", fatal_text) or re.search(r"residue [A-Z]+ (\d+)", fatal_text)
    if m:
        context["residue_id"] = m.group(1)
    mc = re.search(r"Chain ([A-Z])", fatal_text)
    if mc:
        context["chain_id"] = mc.group(1)
    mn = re.search(r"Residue '(\w+)' not found in residue topology database", fatal_text)
    if mn:
        context["missing_residue_name"] = mn.group(1)
    is_local, local_info = extract_local_residue_info(fatal_text)
    if is_local:
        context["res_name"] = local_info["res_name"]
        context["res_id"] = local_info["res_id"]
    return context


def build_graph(config):
    obs = ObservationModule(config["gromacs"]["force_field"], config["gromacs"]["water_model"])
    rf_config = config.get("rfdiffusion", {})
    rf_threshold = rf_config.get("min_residues_for_rfdiffusion", 6)
    modeller_config = config.get("modeller", {})
    max_attempts = config["agent"]["max_attempts"]
    repair_timeout = config["agent"].get("repair_timeout_sec", 300)

    # --- ノード ---

    def check_missing(state):
        return {
            "missing_count": count_missing_residues(state["pdb_path"]),
            "original_pdb_path": state["pdb_path"],
        }

    def _fill_missing_atoms(pdb_path, work_dir, out_name):
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        out_path = os.path.join(work_dir, out_name)
        with open(out_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        return out_path

    def rfdiffusion_node(state):
        """
        【3段階パイプライン】
        1. RFdiffusion: バックボーン生成 (新規残基はGLY)
        2. sequence_recovery: GLY → 正しいアミノ酸名に置換
        3. PDBFixer: 側鎖原子の補完
        """
        original_pdb_path = state["pdb_path"]
        work_dir = state["work_dir"]
        pdb_id = state.get("pdb_id")

        # --- Step 1: RFdiffusion ---
        print(f"[rfdiffusion_node] Step 1: RFdiffusion実行中...")
        backbone_pdb = run_rfdiffusion(original_pdb_path, work_dir, rf_config)
        print(f"[rfdiffusion_node] Step 1 完了: {backbone_pdb}")

        # --- Step 2: 配列復元 ---
        # 【修正】reassign_sequence_from_fasta が未設定でも pdb_id があれば実行する
        should_recover = rf_config.get("reassign_sequence_from_fasta", True)
        if should_recover and pdb_id:
            print(f"[rfdiffusion_node] Step 2: 配列復元実行中 (pdb_id={pdb_id})...")
            recovered_pdb = apply_sequence_recovery(
                original_pdb_path=original_pdb_path,
                rfdiffusion_pdb_path=backbone_pdb,
                work_dir=work_dir,
                pdb_id=pdb_id,
                cache_dir=rf_config.get("fasta_cache_dir"),
            )
            if recovered_pdb != backbone_pdb:
                print(f"[rfdiffusion_node] Step 2 完了: {recovered_pdb}")
            else:
                print(f"[rfdiffusion_node] Step 2: 配列復元スキップ (欠損なし or pdb_id未指定)")
                recovered_pdb = backbone_pdb
        else:
            print(f"[rfdiffusion_node] Step 2: 配列復元スキップ "
                  f"(reassign={should_recover}, pdb_id={pdb_id})")
            recovered_pdb = backbone_pdb

        # --- Step 3: 側鎖補完 ---
        print(f"[rfdiffusion_node] Step 3: 側鎖原子補完中...")
        filled_pdb = _fill_missing_atoms(recovered_pdb, work_dir, "rfdiffusion_filled.pdb")
        print(f"[rfdiffusion_node] Step 3 完了: {filled_pdb}")

        return {"pdb_path": filled_pdb}

    def pdbfixer_node(state):
        out_path = _fill_missing_atoms(state["pdb_path"], state["work_dir"], "pdbfixer_filled.pdb")
        return {"pdb_path": out_path}

    def modeller_minimize_node(state):
        """
        MODELLERによる局所エネルギー極小化。
        config.yaml で modeller.enabled: false の場合はスキップ。
        """
        # 【修正】enabled チェックを追加
        if not modeller_config.get("enabled", True):
            print("[modeller_minimize] スキップ (modeller.enabled: false)")
            return {"pdb_path": state["pdb_path"]}

        # ライセンスキーが未設定の場合はスキップ
        license_key = modeller_config.get("license_key", "")
        if not license_key or license_key == "YOUR-MODELLER-LICENSE-KEY":
            print("[modeller_minimize] スキップ (MODELLERライセンスキー未設定)")
            return {"pdb_path": state["pdb_path"]}

        print(f"[modeller_minimize] 局所エネルギー極小化実行中...")
        print(f"  元PDB: {state['original_pdb_path']}")
        print(f"  入力PDB: {state['pdb_path']}")

        timeout_sec = modeller_config.get("timeout_sec", 600)
        try:
            result = run_with_timeout(
                minimize_with_modeller,
                args=(state["original_pdb_path"], state["pdb_path"],
                      state["work_dir"], modeller_config),
                kwargs={},
                timeout_sec=timeout_sec,
            )
        except Exception as e:
            print(f"[modeller_minimize] 例外発生: {e}")
            print(f"[modeller_minimize] 極小化なしで続行します。")
            return {"pdb_path": state["pdb_path"]}

        if isinstance(result, dict) and result.get("status") in ("repair_timeout", "repair_error"):
            print(f"[modeller_minimize] 失敗/タイムアウト: {result.get('error')}")
            print(f"[modeller_minimize] 極小化なしで続行します。")
            return {"pdb_path": state["pdb_path"]}

        if isinstance(result, str) and os.path.exists(result):
            print(f"[modeller_minimize] 完了: {result}")
            return {"pdb_path": result}

        # 予期しない戻り値
        print(f"[modeller_minimize] 予期しない戻り値: {type(result)}")
        return {"pdb_path": state["pdb_path"]}

    def pdb2gmx_node(state):
        result = obs.run_pdb2gmx(state["pdb_path"], state["work_dir"],
                                 additional_flags=state.get("extra_flags"))
        return {
            "success": result["success"],
            "stderr": result["stderr"],
            "attempt": state.get("attempt", 0) + 1,
            "status": "success" if result["success"] else state.get("status"),
        }

    def diagnosis_node(state):
        fatal_text = extract_fatal_error(state["stderr"])
        category = diagnose_error(state["stderr"])
        history = state.get("repair_history", [])
        candidates = get_repair_candidates(category)
        selected = next((fn for fn in candidates if fn.__name__ not in history), None)
        if selected is None:
            return {"status": "failed_no_candidates"}
        result = run_with_timeout(
            selected,
            args=(state["pdb_path"], state["attempt"], state["work_dir"]),
            kwargs=_extract_context(fatal_text),
            timeout_sec=repair_timeout,
        )
        if result.get("status") in ("repair_timeout", "repair_error"):
            return {"status": result["status"]}
        update = {
            "repair_history": history + [result["op_name"]],
            "status": "repaired",
        }
        if result.get("new_pdb_path"):
            update["pdb_path"] = result["new_pdb_path"]
        new_flags = result.get("extra_flags") or []
        if new_flags:
            update["extra_flags"] = list(dict.fromkeys((state.get("extra_flags") or []) + new_flags))
        return update

    # --- 分岐条件 ---

    def route_missing(state):
        n = state["missing_count"]
        if n >= rf_threshold:
            return "rfdiffusion"
        if n >= 1:
            return "pdbfixer"
        return "pdb2gmx"

    def route_pdb2gmx(state):
        if state["success"] or state["attempt"] >= max_attempts:
            return "end"
        return "diagnosis"

    def route_diagnosis(state):
        if state["status"] in ("failed_no_candidates", "repair_timeout", "repair_error"):
            return "end"
        return "pdb2gmx"

    # --- グラフ構築 ---

    graph = StateGraph(RecoveryState)
    graph.add_node("check_missing", check_missing)
    graph.add_node("rfdiffusion", rfdiffusion_node)
    graph.add_node("pdbfixer", pdbfixer_node)
    graph.add_node("modeller_minimize", modeller_minimize_node)
    graph.add_node("pdb2gmx", pdb2gmx_node)
    graph.add_node("diagnosis", diagnosis_node)

    graph.add_edge(START, "check_missing")
    graph.add_conditional_edges(
        "check_missing", route_missing,
        {"rfdiffusion": "rfdiffusion", "pdbfixer": "pdbfixer", "pdb2gmx": "pdb2gmx"},
    )
    graph.add_edge("rfdiffusion", "modeller_minimize")
    graph.add_edge("pdbfixer", "modeller_minimize")
    graph.add_edge("modeller_minimize", "pdb2gmx")
    graph.add_conditional_edges("pdb2gmx", route_pdb2gmx, {"end": END, "diagnosis": "diagnosis"})
    graph.add_conditional_edges("diagnosis", route_diagnosis, {"end": END, "pdb2gmx": "pdb2gmx"})

    return graph.compile()
