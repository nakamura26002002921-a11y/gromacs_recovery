# recovery_agent/repair.py
import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from Bio.PDB import PDBParser, PDBIO

def _save_fixer_output(fixer, step_num, op_name, work_dir):
    filename = f"step_{step_num}_{op_name}.pdb"
    new_pdb_path = os.path.join(work_dir, filename)
    with open(new_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return new_pdb_path

def pdbfixer_add_missing_atoms(pdb_path, step_num, work_dir, **kwargs):
    op_name = "pdbfixer_add_missing_atoms"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name, work_dir)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}

def pdbfixer_add_missing_atoms_and_hydrogens(pdb_path, step_num, work_dir, ph=7.0, **kwargs):
    op_name = "pdbfixer_add_missing_atoms_and_hydrogens"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name, work_dir)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}

def pdbfixer_replace_nonstandard_residues(pdb_path, step_num, work_dir, **kwargs):
    op_name = "pdbfixer_replace_nonstandard_residues"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name, work_dir)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}

def rename_duplicate_chain_ids(pdb_path, step_num, work_dir, **kwargs):
    op_name = "rename_duplicate_chain_ids"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    available_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    used_ids = set()

    for model in structure:
        for chain in model:
            new_id = next(c for c in available_ids if c not in used_ids)
            used_ids.add(new_id)
            chain.id = new_id

    filename = f"step_{step_num}_{op_name}.pdb"
    new_pdb_path = os.path.join(work_dir, filename)
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb_path)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}

def pdb2gmx_with_ignh_flag(pdb_path, step_num, work_dir, **kwargs):
    op_name = "pdb2gmx_with_ignh_flag"
    return {"op_name": op_name, "new_pdb_path": pdb_path, "extra_flags": ["-ignh"]}

def pdb2gmx_with_explicit_ter_flag(pdb_path, step_num, work_dir, **kwargs):
    op_name = "pdb2gmx_with_explicit_ter_flag"
    return {"op_name": op_name, "new_pdb_path": pdb_path, "extra_flags": ["-ter"]}

def remove_residue_as_last_resort(pdb_path, step_num, work_dir, residue_id=None, chain_id=None, **kwargs):
    """
    最終手段: 問題のある残基を削除する。
    安全のため、鎖ID(chain_id)が特定できない場合は削除を実行しない。
    """
    op_name = "remove_residue_as_last_resort"
    
    if residue_id is None:
        return {"op_name": op_name, "new_pdb_path": None, "extra_flags": None}

    # ★鎖IDが不明な場合は、過剰削除を防ぐため実行しない
    if chain_id is None:
        print(f">> WARNING: Cannot identify chain ID for residue {residue_id}. Skipping removal to avoid over-deletion in multimeric structures.")
        return {
            "op_name": op_name, 
            "new_pdb_path": None, 
            "extra_flags": None,
            "error": "chain_id_not_specified"
        }

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    
    try:
        target_seq_id = int(residue_id)
    except ValueError:
        target_seq_id = residue_id

    removed_count = 0
    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            to_remove = [res for res in chain if res.id[1] == target_seq_id]
            for res in to_remove:
                chain.detach_child(res.id)
                removed_count += 1

    if removed_count == 0:
        return {"op_name": op_name, "new_pdb_path": None, "extra_flags": None, "error": "residue_not_found"}

    filename = f"step_{step_num}_{op_name}.pdb"
    new_pdb_path = os.path.join(work_dir, filename)
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb_path)
    
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None, "structure_altered": True}

REPAIR_CANDIDATES = {
    "MISSING_ATOM": [pdbfixer_add_missing_atoms],
    "MISSING_RESIDUE_DB_ENTRY": [pdbfixer_replace_nonstandard_residues],
    "MISSING_HYDROGEN": [pdb2gmx_with_ignh_flag, pdbfixer_add_missing_atoms_and_hydrogens],
    "CHAIN_SPLIT": [rename_duplicate_chain_ids],
    "TERMINUS_ISSUE": [pdb2gmx_with_explicit_ter_flag],
    "UNKNOWN": [],
}

def get_repair_candidates(category):
    return REPAIR_CANDIDATES.get(category, [])
