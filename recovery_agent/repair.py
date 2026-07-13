# recovery_agent/repair.py
import os
import subprocess
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Residue import Residue as _BioResidue
from Bio.PDB.Atom import Atom as _BioAtom

# 6残基以上の欠損(ループ)があるとpdbfixerの単純な幾何学的補間では
# 立体構造が破綻しやすく、また巨大な複合体では処理が非常に重くなる
# (実測: 1AONのような多鎖の巨大複合体でタイムアウトする)。
# そのため一定数以上の欠損はRFdiffusionによる再構築に振り分ける。
RFDIFFUSION_MISSING_RESIDUE_THRESHOLD = 6
RFDIFFUSION_DEFAULT_SCRIPT = os.environ.get(
    "RFDIFFUSION_RUN_INFERENCE", "/opt/RFdiffusion/scripts/run_inference.py"
)
RFDIFFUSION_DEFAULT_TIMEOUT_SEC = int(os.environ.get("RFDIFFUSION_TIMEOUT_SEC", "1800"))

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

def count_missing_residues(pdb_path):
    """
    PDBFixerのfindMissingResidues()でSEQRESとATOM座標を比較し、
    構造から丸ごと欠損している残基の総数とその位置を調べる。

    戻り値: (欠損残基の総数, gaps)
      gaps: {(chain_index, insert_index): [残基名, ...], ...}
            PDBFixer内部の表現。insert_indexは「そのギャップの直前に
            何番目の残基があるか」を示すchain内インデックス。
    """
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    gaps = fixer.missingResidues or {}
    total = sum(len(v) for v in gaps.values())
    return total, gaps


def _largest_gap(gaps):
    """最も長い欠損ループのキーと残基名リストを返す"""
    if not gaps:
        return None, None
    key = max(gaps, key=lambda k: len(gaps[k]))
    return key, gaps[key]


def _build_rfdiffusion_contig(pdb_path, gaps):
    """
    最大の欠損ギャップを対象に、RFdiffusionのcontig文字列
    (例: "A1-54/6-6/A61-120")を組み立てる。
    複数箇所に欠損がある場合は、まず最大のギャップから修復し、
    残りは再度pdb2gmxを流した後の次のattemptで処理する。
    """
    (chain_index, insert_index), missing_names = _largest_gap(gaps)
    gap_len = len(missing_names)

    fixer = PDBFixer(filename=pdb_path)
    chain = list(fixer.topology.chains())[chain_index]
    chain_id = chain.id
    residues = list(chain.residues())

    if insert_index == 0:
        # N末端側が丸ごと欠損 -> 全体をdiffuseし、C末側のみ固定
        after_start = int(residues[0].id)
        after_end = int(residues[-1].id)
        contig = f"{gap_len}-{gap_len}/{chain_id}{after_start}-{after_end}"
    elif insert_index >= len(residues):
        # C末端側が丸ごと欠損 -> N末側のみ固定
        before_start = int(residues[0].id)
        before_end = int(residues[-1].id)
        contig = f"{chain_id}{before_start}-{before_end}/{gap_len}-{gap_len}"
    else:
        before_start = int(residues[0].id)
        before_end = int(residues[insert_index - 1].id)
        after_start = int(residues[insert_index].id)
        after_end = int(residues[-1].id)
        contig = (
            f"{chain_id}{before_start}-{before_end}/"
            f"{gap_len}-{gap_len}/"
            f"{chain_id}{after_start}-{after_end}"
        )

    return contig, chain_id


def _clone_residue(residue):
    """Bio.PDBのResidueを、親構造から独立した形で複製する"""
    new_res = _BioResidue(residue.id, residue.resname, residue.segid)
    for atom in residue:
        new_atom = _BioAtom(
            atom.get_name(), atom.get_coord(), atom.get_bfactor(),
            atom.get_occupancy(), atom.get_altloc(), atom.get_fullname(),
            atom.get_serial_number(), element=atom.element,
        )
        new_res.add(new_atom)
    return new_res


def _merge_rfdiffusion_output(original_pdb_path, designed_pdb_path, out_path):
    """
    RFdiffusionが出力した骨格構造(backboneのみ)のうち、元の構造には
    存在しなかった(=欠損していた)残基だけを取り出し、正しい位置に
    挿入して元の構造とマージする。既存残基は実測座標を優先して保持する。
    側鎖原子・水素は後続のpdbfixer_add_missing_atoms等のステップで付加される。
    """
    parser = PDBParser(QUIET=True)
    original = parser.get_structure("orig", original_pdb_path)
    designed = parser.get_structure("designed", designed_pdb_path)

    model0 = original[0]
    d_model0 = designed[0]

    for d_chain in d_model0:
        chain_id = d_chain.id
        if chain_id not in model0:
            continue
        orig_chain = model0[chain_id]
        existing_ids = {res.id[1] for res in orig_chain if res.id[0] == ' '}

        for d_res in d_chain:
            if d_res.id[0] != ' ' or d_res.id[1] in existing_ids:
                continue  # 標準残基以外、または既存残基はスキップ(実測座標を優先)

            new_res = _clone_residue(d_res)
            # 残基番号順になる位置へ挿入する(末尾に追加すると
            # ファイル中の順序が崩れ、pdb2gmxが鎖を分断してしまうため)
            insert_pos = len(orig_chain.child_list)
            for i, existing_res in enumerate(orig_chain.child_list):
                if existing_res.id[0] == ' ' and existing_res.id[1] > new_res.id[1]:
                    insert_pos = i
                    break
            orig_chain.child_list.insert(insert_pos, new_res)
            orig_chain.child_dict[new_res.id] = new_res
            new_res.set_parent(orig_chain)

    io = PDBIO()
    io.set_structure(original)
    io.save(out_path)


def rfdiffusion_rebuild_missing_loops(pdb_path, step_num, work_dir,
                                       run_inference_script=None,
                                       timeout_sec=None, **kwargs):
    """
    6残基以上のループが構造から欠損している場合に、RFdiffusionの
    motif-scaffolding(inpainting)機能でループ領域を新規に構造予測させ、
    既存の実測座標と結合する。

    RFdiffusion自体は別途インストールが必要(GPU + 学習済み重み)。
    run_inference_scriptで場所を指定するか、環境変数
    RFDIFFUSION_RUN_INFERENCE で指定すること。
    """
    op_name = "rfdiffusion_rebuild_missing_loops"
    script = run_inference_script or RFDIFFUSION_DEFAULT_SCRIPT
    timeout = timeout_sec or RFDIFFUSION_DEFAULT_TIMEOUT_SEC

    total_missing, gaps = count_missing_residues(pdb_path)
    if total_missing < RFDIFFUSION_MISSING_RESIDUE_THRESHOLD:
        return {
            "op_name": op_name, "new_pdb_path": None, "extra_flags": None,
            "status": "repair_error",
            "error": (
                f"欠損残基数({total_missing})が閾値"
                f"({RFDIFFUSION_MISSING_RESIDUE_THRESHOLD})未満のためRFdiffusionは不要"
            ),
        }

    if not os.path.exists(script):
        return {
            "op_name": op_name, "new_pdb_path": None, "extra_flags": None,
            "status": "repair_error",
            "error": (
                f"RFdiffusion実行スクリプトが見つかりません: {script}\n"
                f"環境変数 RFDIFFUSION_RUN_INFERENCE か、"
                f"config.yamlのrfdiffusion.run_inference_scriptで"
                f"run_inference.pyのパスを指定してください。"
            ),
        }

    contig_str, chain_id = _build_rfdiffusion_contig(pdb_path, gaps)
    out_prefix = os.path.join(work_dir, f"step_{step_num}_{op_name}")

    cmd = [
        "python", script,
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={os.path.abspath(pdb_path)}",
        f"contigmap.contigs=[{contig_str}]",
        "inference.num_designs=1",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "")[-2000:]
        return {
            "op_name": op_name, "new_pdb_path": None, "extra_flags": None,
            "status": "repair_error", "error": f"RFdiffusion実行失敗: {tail}",
        }
    except subprocess.TimeoutExpired:
        return {
            "op_name": op_name, "new_pdb_path": None, "extra_flags": None,
            "status": "repair_timeout",
            "error": f"RFdiffusionが制限時間({timeout}秒)内に完了しませんでした",
        }

    designed_pdb = f"{out_prefix}_0.pdb"
    if not os.path.exists(designed_pdb):
        return {
            "op_name": op_name, "new_pdb_path": None, "extra_flags": None,
            "status": "repair_error", "error": "RFdiffusionの出力PDBが見つかりませんでした",
        }

    merged_path = os.path.join(work_dir, f"step_{step_num}_{op_name}_merged.pdb")
    _merge_rfdiffusion_output(pdb_path, designed_pdb, merged_path)

    return {
        "op_name": op_name,
        "new_pdb_path": merged_path,
        "extra_flags": None,
        "structure_altered": True,
    }


REPAIR_CANDIDATES = {
    "MISSING_ATOM": [pdbfixer_add_missing_atoms],
    "MISSING_RESIDUE_DB_ENTRY": [strip_unknown_residue, pdbfixer_replace_nonstandard_residues, strip_hetero_cofactors],
    "MISSING_HYDROGEN": [pdb2gmx_with_ignh_flag, pdbfixer_add_missing_atoms_and_hydrogens],
    "HETERO_CHAIN_TYPE_MISMATCH": [strip_hetero_cofactors],
    "CHAIN_SPLIT": [rename_duplicate_chain_ids],
    "TERMINUS_ISSUE": [pdb2gmx_with_explicit_ter_flag],
    "UNKNOWN": [],
}

def get_repair_candidates(category):
    return REPAIR_CANDIDATES.get(category, [])
