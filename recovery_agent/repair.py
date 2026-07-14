# recovery_agent/repair.py
import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from Bio.PDB import PDBParser, PDBIO, Select

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
    return {"op_name": "pdb2gmx_with_ignh_flag", "new_pdb_path": pdb_path, "extra_flags": ["-ignh"]}

def pdb2gmx_with_explicit_ter_flag(pdb_path, step_num, work_dir, **kwargs):
    return {"op_name": "pdb2gmx_with_explicit_ter_flag", "new_pdb_path": pdb_path, "extra_flags": ["-ter"]}

def remove_residue_as_last_resort(pdb_path, step_num, work_dir, residue_id=None, chain_id=None, **kwargs):
    op_name = "remove_residue_as_last_resort"
    if residue_id is None or chain_id is None:
        return {"op_name": op_name, "new_pdb_path": None, "extra_flags": None, "error": "id_not_specified"}

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

def strip_hetero_cofactors(pdb_path, step_num, work_dir, **kwargs):
    """標準アミノ酸・水以外のHETATM(イオン・補因子等)を除去する"""
    op_name = "strip_hetero_cofactors"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    for model in structure:
        for chain in model:
            # res.id[0] が ' ' (空白) なら標準残基、'W' なら水、その他はHETATM
            to_remove = [res for res in chain if res.id[0] != ' ' and res.resname != 'HOH']
            for res in to_remove:
                chain.detach_child(res.id)

    filename = f"step_{step_num}_{op_name}.pdb"
    new_pdb_path = os.path.join(work_dir, filename)
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb_path)
    # 補因子を消す=生物学的情報を失う、強い構造変更なので必ずフラグを立てる
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None, "structure_altered": True}

def strip_unknown_residue(pdb_path, step_num, work_dir, missing_residue_name=None, **kwargs):
    op_name = "strip_unknown_residue"
    if not missing_residue_name:
        return strip_hetero_cofactors(pdb_path, step_num, work_dir, **kwargs)
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            return residue.get_resname().strip() != missing_residue_name
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("temp", pdb_path)
    out_path = os.path.join(work_dir, f"step_{step_num}_{op_name}.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, select=ResidueSelect())
    return {"op_name": op_name, "new_pdb_path": out_path, "extra_flags": None, "structure_altered": True}


def pdbfixer_local_repair(pdb_path, attempt, work_dir, res_name=None, res_id=None, **kwargs):
    """既存座標は動かさず、エラーが出た残基周辺の欠損原子だけを局所的に補完する"""
    op_name = "pdbfixer_local_repair"
    if res_name and res_id:
        print(f"  -> Targeting local repair for residue: {res_name} {res_id}")
    fixer = PDBFixer(filename=pdb_path)
    # addMissingAtoms() は findMissingResidues()/findMissingAtoms() が設定する
    # self.missingResidues / self.missingAtoms を前提にしているため、必ず先に呼ぶ必要がある
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    new_pdb_path = _save_fixer_output(fixer, attempt, op_name, work_dir)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}



REPAIR_CANDIDATES = {
    "MISSING_ATOM": [pdbfixer_add_missing_atoms],
    "MISSING_RESIDUE_DB_ENTRY": [strip_unknown_residue, pdbfixer_replace_nonstandard_residues, strip_hetero_cofactors],
    "MISSING_HYDROGEN": [pdb2gmx_with_ignh_flag, pdbfixer_add_missing_atoms_and_hydrogens],
    "HETERO_CHAIN_TYPE_MISMATCH": [strip_hetero_cofactors],
    "CHAIN_SPLIT": [rename_duplicate_chain_ids],
    "TERMINUS_ISSUE": [pdb2gmx_with_explicit_ter_flag],
    "LOCAL_RESIDUE_ISSUE": [pdbfixer_local_repair],
    "UNKNOWN": [],
}

def get_repair_candidates(category):
    return REPAIR_CANDIDATES.get(category, [])
