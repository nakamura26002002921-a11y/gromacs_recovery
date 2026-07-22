# recovery_agent/graph.py
import os
import re
from typing import TypedDict, List
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
from .preflight import preflight


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
    ctx = {}
    if not fatal_text:
        return ctx
    m = re.search(r"Residue (\d+) named", fatal_text) or re.search(r"residue [A-Z]+ (\d+)", fatal_text)
    if m:
        ctx["residue_id"] = m.group(1)
    mc = re.search(r"Chain ([A-Z])", fatal_text)
    if mc:
        ctx["chain_id"] = mc.group(1)
    mn = re.search(r"Residue '(\w+)' not found in residue topology database", fatal_text)
    if mn:
        ctx["missing_residue_name"] = mn.group(1)
    is_local, info = extract_local_residue_info(fatal_text)
    if is_local:
        ctx.update(res_name=info["res_name"], res_id=info["res_id"])
    return ctx


def build_graph(config):
    obs = ObservationModule(config["gromacs"]["force_field"], config["gromacs"]["water_model"])
    rf_config = config.get("rfdiffusion", {})
    rf_threshold = rf_config.get("min_residues_for_rfdiffusion", 6)
    modeller_config = config.get("modeller", {})
    max_attempts = config["agent"]["max_attempts"]
    repair_timeout = config["agent"].get("repair_timeout_sec", 300)

    def _fill_atoms(pdb_path, work_dir, name):
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        out = os.path.join(work_dir, name)
        with open(out, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        return out

    def check_missing(state):
        preflight(config, state["pdb_id"])
        return {"missing_count": count_missing_residues(state["pdb_path"]),
                "original_pdb_path": state["pdb_path"]}

    def rfdiffusion_node(state):
        orig = state["pdb_path"]
        wd = state["work_dir"]
        pdb_id = state["pdb_id"]
        backbone = run_rfdiffusion(orig, wd, rf_config)
        recovered = apply_sequence_recovery(orig, backbone, wd, pdb_id,
                                            cache_dir=rf_config.get("fasta_cache_dir"))
        filled = _fill_atoms(recovered, wd, "rfdiffusion_filled.pdb")
        return {"pdb_path": filled}

    def pdbfixer_node(state):
        return {"pdb_path": _fill_atoms(state["pdb_path"], state["work_dir"], "pdbfixer_filled.pdb")}

    def modeller_minimize_node(state):
        out = minimize_with_modeller(
            state["original_pdb_path"], state["pdb_path"],
            state["work_dir"], modeller_config)
        return {"pdb_path": out}

    def pdb2gmx_node(state):
        r = obs.run_pdb2gmx(state["pdb_path"], state["work_dir"],
                            additional_flags=state.get("extra_flags"))
        return {"success": r["success"], "stderr": r["stderr"],
                "attempt": state.get("attempt", 0) + 1,
                "status": "success" if r["success"] else state.get("status")}

    def diagnosis_node(state):
        fatal = extract_fatal_error(state["stderr"])
        history = state.get("repair_history", [])
        candidates = get_repair_candidates(diagnose_error(state["stderr"]))
        fn = next((f for f in candidates if f.__name__ not in history), None)
        if fn is None:
            return {"status": "failed_no_candidates"}
        r = run_with_timeout(fn, args=(state["pdb_path"], state["attempt"], state["work_dir"]),
                             kwargs=_extract_context(fatal), timeout_sec=repair_timeout)
        if r.get("status") in ("repair_timeout", "repair_error"):
            return {"status": r["status"]}
        update = {"repair_history": history + [r["op_name"]], "status": "repaired"}
        if r.get("new_pdb_path"):
            update["pdb_path"] = r["new_pdb_path"]
        if r.get("extra_flags"):
            update["extra_flags"] = list(dict.fromkeys(
                (state.get("extra_flags") or []) + r["extra_flags"]))
        return update

    def route_missing(state):
        n = state["missing_count"]
        if n >= rf_threshold:
            return "rfdiffusion"
        return "pdbfixer" if n >= 1 else "pdb2gmx"

    def route_pdb2gmx(state):
        return "end" if state["success"] or state["attempt"] >= max_attempts else "diagnosis"

    def route_diagnosis(state):
        return "end" if state["status"] in ("failed_no_candidates", "repair_timeout", "repair_error") else "pdb2gmx"

    g = StateGraph(RecoveryState)
    g.add_node("check_missing", check_missing)
    g.add_node("rfdiffusion", rfdiffusion_node)
    g.add_node("pdbfixer", pdbfixer_node)
    g.add_node("modeller_minimize", modeller_minimize_node)
    g.add_node("pdb2gmx", pdb2gmx_node)
    g.add_node("diagnosis", diagnosis_node)

    g.add_edge(START, "check_missing")
    g.add_conditional_edges("check_missing", route_missing,
                            {"rfdiffusion": "rfdiffusion", "pdbfixer": "pdbfixer", "pdb2gmx": "pdb2gmx"})
    
    g.add_edge("rfdiffusion", "modeller_minimize")
    g.add_edge("pdbfixer", "modeller_minimize")
    g.add_edge("modeller_minimize", "pdb2gmx")

    g.add_conditional_edges("pdb2gmx", route_pdb2gmx, {"end": END, "diagnosis": "diagnosis"})
    g.add_conditional_edges("diagnosis", route_diagnosis, {"end": END, "pdb2gmx": "pdb2gmx"})

    return g.compile()
