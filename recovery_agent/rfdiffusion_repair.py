# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
from pdbfixer import PDBFixer
from Bio.PDB import PDBParser, PDBIO


def _build_contig(fixer):
    """各鎖に欠損箇所が高々1つある前提でRFdiffusionのcontig文字列を作る"""
    chain_groups = []  # チェーンごとのトークンのリストのリスト
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        cid = chain.id
        gap = next(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items() if ci == chain.index),
            None,
        )
        start_num, end_num = int(residues[0].id), int(residues[-1].id)
        if gap is None:
            chain_groups.append([f"{cid}{start_num}-{end_num}"])
            continue
        pos, names = gap
        gap_len = len(names)
        if pos == 0:
            chain_groups.append([f"{gap_len}-{gap_len}", f"{cid}{start_num}-{end_num}"])
        elif pos >= len(residues):
            chain_groups.append([f"{cid}{start_num}-{end_num}", f"{gap_len}-{gap_len}"])
        else:
            mid_num, next_num = int(residues[pos - 1].id), int(residues[pos].id)
            chain_groups.append(
                [f"{cid}{start_num}-{mid_num}", f"{gap_len}-{gap_len}", f"{cid}{next_num}-{end_num}"]
            )
    # 同一チェーン内は "/" で連結、チェーン間(=新しい鎖の開始)は "," で区切る
    return ",".join("/".join(group) for group in chain_groups)


def _load_trb(trb_path):
    import torch
    return torch.load(trb_path, map_location="cpu", weights_only=False)


def _merge_designed_region(original_pdb_path, hal_pdb_path, trb_path, work_dir):
    """RFdiffusionの出力(hal)は骨格原子のみ・複合体全体を作り直したものであり、
    側鎖やHETATM(水・イオン・補因子等)を含まない。そのまま採用すると構造の大部分を
    失ってしまうため、.trbの対応表(con_hal_pdb_idx <-> con_ref_pdb_idx)を使って
    「実際に新規生成された(=対応表に無い)残基」だけを元の全原子構造に差し込む。
    """
    trb = _load_trb(trb_path)
    kept_hal_keys = {(chain, int(resnum)) for chain, resnum in trb["con_hal_pdb_idx"]}

    parser = PDBParser(QUIET=True)
    orig_structure = parser.get_structure("orig", original_pdb_path)
    hal_structure = parser.get_structure("hal", hal_pdb_path)
    orig_model, hal_model = orig_structure[0], hal_structure[0]

    for hal_chain in hal_model:
        cid = hal_chain.id
        if cid not in orig_model:
            continue  # 新規鎖の追加には未対応(既存鎖内の欠損補完のみサポート)
        orig_chain = orig_model[cid]
        for hal_res in hal_chain:
            if (cid, hal_res.id[1]) in kept_hal_keys:
                continue  # 既存領域: オリジナルの全原子座標をそのまま維持
            # 新規に設計された残基 -> 元の鎖に挿入(既存の同IDプレースホルダがあれば置換)
            if hal_res.id in orig_chain:
                orig_chain.detach_child(hal_res.id)
            orig_chain.add(hal_res.copy())
        orig_chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))  # resseq順に並べ直す

    out_path = os.path.join(work_dir, "rfdiffusion_merged.pdb")
    io = PDBIO()
    io.set_structure(orig_structure)
    io.save(out_path)
    return out_path


def run_rfdiffusion(pdb_path, work_dir, rf_config):
    """RFdiffusionを実行し、新規生成された欠損部分だけを元構造に統合したPDBのパスを返す"""
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    contig = _build_contig(fixer)

    out_prefix = os.path.join(work_dir, "rfdiffusion_out")
    cmd = [
        "python", rf_config["script_path"],
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={os.path.abspath(pdb_path)}",
        f"inference.model_directory_path={rf_config['model_directory_path']}",
        f"contigmap.contigs=[{contig}]",
        f"inference.num_designs={rf_config.get('num_designs', 1)}",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=work_dir, timeout=rf_config.get("timeout_sec", 1800),
    )
    if result.returncode != 0:
        raise RuntimeError(f"RFdiffusion failed: {result.stderr[-2000:]}")

    hal_pdb_path = f"{out_prefix}_0.pdb"
    trb_path = f"{out_prefix}_0.trb"
    if not os.path.exists(hal_pdb_path):
        raise RuntimeError(f"RFdiffusion output not found: {hal_pdb_path}")
    if not os.path.exists(trb_path):
        # .trb (対応表)が無いと、どの残基が新規生成分かを安全に判定できず、
        # 側鎖・HETATM・水を失った縮小構造をそのまま使ってしまう危険があるため停止する
        raise RuntimeError(f"RFdiffusion .trb file not found: {trb_path} (side-chain/HETATM-preserving merge requires it)")

    return _merge_designed_region(pdb_path, hal_pdb_path, trb_path, work_dir)

