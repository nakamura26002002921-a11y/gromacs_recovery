import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from Bio.PDB import PDBParser, PDBIO


def _save_fixer_output(fixer, step_num, op_name):
    new_pdb_path = f"step_{step_num}_{op_name}.pdb"
    with open(new_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return new_pdb_path


def pdbfixer_add_missing_atoms(pdb_path, step_num):
    """欠損している重原子のみを追加する(水素は触らない)"""
    op_name = "pdbfixer_add_missing_atoms"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}


def pdbfixer_add_missing_atoms_and_hydrogens(pdb_path, step_num, ph=7.0):
    """欠損重原子に加えて水素も明示的に付加する(MISSING_HYDROGEN用)"""
    op_name = "pdbfixer_add_missing_atoms_and_hydrogens"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}


def pdbfixer_replace_nonstandard_residues(pdb_path, step_num):
    """非標準残基を標準アミノ酸に変換する(MISSING_RESIDUE_DB_ENTRY用)"""
    op_name = "pdbfixer_replace_nonstandard_residues"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}


def rename_duplicate_chain_ids(pdb_path, step_num):
    """重複・分断しているchain IDをA, B, C...に振り直す(CHAIN_SPLIT用)"""
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

    new_pdb_path = f"step_{step_num}_{op_name}.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb_path)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}


def pdb2gmx_with_ignh_flag(pdb_path, step_num):
    """PDBは変更せず、pdb2gmx実行時に-ignhを付けて既存水素を無視する(MISSING_HYDROGEN用)"""
    op_name = "pdb2gmx_with_ignh_flag"
    # PDB自体は変更しないので同じパスを返す
    return {"op_name": op_name, "new_pdb_path": pdb_path, "extra_flags": ["-ignh"]}


def pdb2gmx_with_explicit_ter_flag(pdb_path, step_num):
    """末端残基の処理を明示的に指定する(TERMINUS_ISSUE用)"""
    op_name = "pdb2gmx_with_explicit_ter_flag"
    return {"op_name": op_name, "new_pdb_path": pdb_path, "extra_flags": ["-ter"]}


def remove_residue_as_last_resort(pdb_path, step_num, residue_id=None):
    """
    最終手段: 問題のある残基を削除する。
    residue_idが指定できない場合は、この関数は呼ばないこと(構造破壊のリスクが高いため)。
    """
    op_name = "remove_residue_as_last_resort"
    if residue_id is None:
        # 安全のため、residue_idがない場合は何もせず失敗扱いにする
        return {"op_name": op_name, "new_pdb_path": None, "extra_flags": None}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    for model in structure:
        for chain in model:
            to_remove = [res for res in chain if res.id == residue_id]
            for res in to_remove:
                chain.detach_child(res.id)

    new_pdb_path = f"step_{step_num}_{op_name}.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb_path)
    # 構造を変更した、という強いフラグを付ける
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
    """エラーカテゴリに基づく修復関数の優先順位リストを返す"""
    return REPAIR_CANDIDATES.get(category, [])
