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
        # 極小化ノードで「どの残基が今回の修復で新規追加/補完されたか」を再判定するために、
        # 欠損情報がまだ残っている元PDBのパスをstateに保持しておく。
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
            # 【重要】keepIds=True を必ず指定すること。
            # デフォルト(keepIds=False)では、OpenMMが元のPDBのチェーンID・残基番号を
            # 完全に無視し、トポロジー内の出現順に基づいて A,B,C... / 1,2,3... と
            # 振り直してしまう(openmm/app/pdbfile.py の writeModel を参照)。
            # これにより、RFdiffusionでマージした際の残基番号(例: 2-548)が
            # 1-547 のような別の番号体系に化けてしまい、後続の MODELLER 残基選択
            # (_get_repaired_resnums は元の番号を前提に計算する)や、GROMACSトポロジー、
            # sequence_recovery結果との整合性が全て崩れる。
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        return out_path

    def rfdiffusion_node(state):
        # 【正しい2段階パイプライン】
        # 1. RFdiffusion: バックボーン(N,CA,C,O)のみを生成する構造生成モデル。
        #    公式仕様により、新規生成された残基は側鎖を持たず常にGLYとして出力される
        #    (側鎖予測には損失が適用されておらず信頼できないため)。
        # 2. sequence_recovery: RFdiffusionとは独立したステップとして、RCSB FASTAとの
        #    アラインメントにより、新規生成された各残基の「あるべきアミノ酸種」を推定し、
        #    GLYだった残基名をそのアミノ酸名に置き換える(座標自体はまだGLY相当のまま)。
        # 3. PDBFixerで、置き換え後の残基名に対応する側鎖原子を補完する。
        original_pdb_path = state["pdb_path"]

        backbone_pdb = run_rfdiffusion(
            original_pdb_path, state["work_dir"], rf_config,
        )

        recovered_pdb = apply_sequence_recovery(
            original_pdb_path=original_pdb_path,
            rfdiffusion_pdb_path=backbone_pdb,
            work_dir=state["work_dir"],
            pdb_id=state.get("pdb_id"),
            cache_dir=rf_config.get("fasta_cache_dir"),
        ) if rf_config.get("reassign_sequence_from_fasta") else backbone_pdb

        filled_pdb = _fill_missing_atoms(recovered_pdb, state["work_dir"], "rfdiffusion_filled.pdb")
        return {"pdb_path": filled_pdb}

    def pdbfixer_node(state):
        out_path = _fill_missing_atoms(state["pdb_path"], state["work_dir"], "pdbfixer_filled.pdb")
        return {"pdb_path": out_path}

    def modeller_minimize_node(state):
        # RFdiffusion経路(6残基以上)・PDBFixer経路(1〜5残基)のどちらから来た場合でも、
        # pdb2gmxに渡す直前に、新規生成/補完された残基とその近傍のみをMODELLERで
        # 局所的にエネルギー極小化する。側鎖の衝突や不自然な結合長/角度を緩和し、
        # pdb2gmxがクラッシュ原子で失敗する確率を下げるための最終仕上げステップ。
        #
        # MODELLERの最適化は入力構造によっては長時間かかる/収束しないことがあるため、
        # repair系ノードと同様に別プロセス+タイムアウトで保護する。
        timeout_sec = modeller_config.get("timeout_sec", 600)
        result = run_with_timeout(
            minimize_with_modeller,
            args=(state["original_pdb_path"], state["pdb_path"], state["work_dir"], modeller_config),
            kwargs={},
            timeout_sec=timeout_sec,
        )
        # run_with_timeoutは本来repairノード用の結果dict形式を期待するが、
        # minimize_with_modellerは文字列(パス)を直接返すため、成功時はそのまま使う。
        if isinstance(result, dict) and result.get("status") in ("repair_timeout", "repair_error"):
            print(f"[Warning] MODELLER minimization failed/timed out: {result.get('error')}. "
                  f"Falling back to un-minimized structure.")
            return {"pdb_path": state["pdb_path"]}
        return {"pdb_path": result}

    def pdb2gmx_node(state):
        result = obs.run_pdb2gmx(state["pdb_path"], state["work_dir"], additional_flags=state.get("extra_flags"))
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
    graph.add_edge("rfdiffusion", "modeller_minimize")   # G: PDB更新 -> 極小化 -> D
    graph.add_edge("pdbfixer", "modeller_minimize")      # G: PDB更新 -> 極小化 -> D
    graph.add_edge("modeller_minimize", "pdb2gmx")
    graph.add_conditional_edges("pdb2gmx", route_pdb2gmx, {"end": END, "diagnosis": "diagnosis"})
    graph.add_conditional_edges("diagnosis", route_diagnosis, {"end": END, "pdb2gmx": "pdb2gmx"})

    return graph.compile()
