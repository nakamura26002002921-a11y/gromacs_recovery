# recovery_agent/repair.py
import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from Bio.PDB import PDBParser, PDBIO

def _save_fixer_output(fixer, step_num, op_name, work_dir):
    """PDBFixerの結果をwork_dir内に保存する"""
    filename = f"step_{step_num}_{op_name}.pdb"
    new_pdb_path = os.path.join(work_dir, filename)
    with open(new_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return new_pdb_path

def pdbfixer_add_missing_atoms(pdb_path, step_num, work_dir, **kwargs):
    """欠損している重原子のみを追加する"""
    op_name = "pdbfixer_add_missing_atoms"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name, work_dir)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}

def pdbfixer_add_missing_atoms_and_hydrogens(pdb_path, step_num, work_dir, ph=7.0, **kwargs):
    """欠損重原子に加えて水素も明示的に付加する"""
    op_name = "pdbfixer_add_missing_atoms_and_hydrogens"
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    new_pdb_path = _save_fixer_output(fixer, step_num, op_name, work_dir)
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None}

def pdbfixer_replace_nonstandard_residues(pdb_path, step_num, work_dir, **kwargs):
    """非標準残基を標準アミノ酸に変換する"""
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
    """重複・分断しているchain IDをA, B, C...に振り直す"""
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
    """PDBは変更せず、pdb2gmx実行時に-ignhを付ける"""
    op_name = "pdb2gmx_with_ignh_flag"
    # PDB自体は変更しないが、work_dirは使わない
    return {"op_name": op_name, "new_pdb_path": pdb_path, "extra_flags": ["-ignh"]}

def pdb2gmx_with_explicit_ter_flag(pdb_path, step_num, work_dir, **kwargs):
    """末端残基の処理を明示的に指定する"""
    op_name = "pdb2gmx_with_explicit_ter_flag"
    return {"op_name": op_name, "new_pdb_path": pdb_path, "extra_flags": ["-ter"]}

def remove_residue_as_last_resort(pdb_path, step_num, work_dir, residue_id=None, **kwargs):
    """
    最終手段: 問題のある残基を削除する。
    agent.pyからkwargs経由でresidue_idが渡されることを想定。
    """
    op_name = "remove_residue_as_last_resort"
    
    # residue_idが特定できない場合は失敗扱い（構造破壊を防ぐため）
    if residue_id is None:
        return {"op_name": op_name, "new_pdb_path": None, "extra_flags": None}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    
    # PDBのresidue idは整数または文字列のタプル等形式の場合があるため、文字列として比較する準備をする
    # GROMACSのエラーログからは整数のシーケンス番号が得られるが、PDBファイル上のIDと一致するとは限らない
    # ここでは簡易的に、residue.id[1] (シーケンス番号) が一致するものを削除する
    try:
        target_seq_id = int(residue_id)
    except ValueError:
        target_seq_id = residue_id

    for model in structure:
        for chain in model:
            to_remove = [res for res in chain if res.id[1] == target_seq_id]
            for res in to_remove:
                chain.detach_child(res.id)

    filename = f"step_{step_num}_{op_name}.pdb"
    new_pdb_path = os.path.join(work_dir, filename)
    io = PDBIO()
    io.set_structure(structure)
    io.save(new_pdb_path)
    
    return {"op_name": op_name, "new_pdb_path": new_pdb_path, "extra_flags": None, "structure_altered": True}

# 候補リスト
# remove_residue_as_last_resort は最終手段として、特定のカテゴリの末尾に追加することを想定
REPAIR_CANDIDATES = {
    "MISSING_ATOM": [pdbfixer_add_missing_atoms],
    "MISSING_RESIDUE_DB_ENTRY": [pdbfixer_replace_nonstandard_residues], # 必要に応じて remove... を末尾に追加
    "MISSING_HYDROGEN": [pdb2gmx_with_ignh_flag, pdbfixer_add_missing_atoms_and_hydrogens],
    "CHAIN_SPLIT": [rename_duplicate_chain_ids],
    "TERMINUS_ISSUE": [pdb2gmx_with_explicit_ter_flag],
    "UNKNOWN": [],
}

def get_repair_candidates(category):
    return REPAIR_CANDIDATES.get(category, [])
