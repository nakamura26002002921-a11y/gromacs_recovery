# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
import pickle
import re
import json
from Bio.PDB import PDBParser, PDBIO
from pdbfixer import PDBFixer

# sequence_recovery.py から配列復元に必要なモジュールをインポート
from recovery_agent.sequence_recovery import (
    fetch_rcsb_fasta, 
    parse_rcsb_fasta, 
    recover_complex_sequences, 
    _THREE_TO_ONE
)


def _parse_resnum(res_id):
    """'100A' や '-1' のようなPDB残基IDから整数部分のみを抽出する（挿入コード対策）"""
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


def _get_expected_missing_resnums(fixer):
    """
    PDBFixerの情報から、追加されるべき残基の「予想残基番号」と「既存残基辞書」を抽出する。
    これにより、RFdiffusion実行前にFASTAとの配列マッピングが可能になる。
    """
    pdb_complex_residues = {}
    generated_resnums_dict = {}
    missing_regions = {}  # chain_id -> [(start_resnum, end_resnum), ...] (provide_seq構築用)

    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        cid = chain.id
        pdb_complex_residues[cid] = {}
        for res in residues:
            resnum = _parse_resnum(res.id)
            pdb_complex_residues[cid][resnum] = res.name

        gaps = sorted(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items() if ci == chain.index),
            key=lambda x: x[0],
        )

        gen_resnums = set()
        regions = []
        for pos, names in gaps:
            gap_len = len(names)
            if pos == 0:
                # 先頭が欠損の場合、既存の最初の残基から逆算
                next_resnum = _parse_resnum(residues[0].id)
                start = next_resnum - gap_len
            else:
                # 途中の欠損の場合、直前の残基番号から連番で振る
                prev_resnum = _parse_resnum(residues[pos - 1].id)
                start = prev_resnum + 1

            end = start + gap_len - 1
            regions.append((start, end))
            for offset in range(gap_len):
                gen_resnums.add(start + offset)

        if gen_resnums:
            generated_resnums_dict[cid] = gen_resnums
            missing_regions[cid] = regions

    return pdb_complex_residues, generated_resnums_dict, missing_regions


def _build_contig(fixer):
    """RFdiffusion用のcontig文字列を作る。1鎖に欠損が複数箇所あっても正しく扱う。"""
    chain_groups = []
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        cid = chain.id
        start_num, end_num = _parse_resnum(residues[0].id), _parse_resnum(residues[-1].id)

        gaps = sorted(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items() if ci == chain.index),
            key=lambda x: x[0],
        )
        if not gaps:
            chain_groups.append([f"{cid}{start_num}-{end_num}"])
            continue

        tokens = []
        cursor = 0
        seg_start_num = start_num
        for pos, names in gaps:
            gap_len = len(names)
            if pos == 0:
                tokens.append(f"{gap_len}-{gap_len}")
                cursor = 0
                continue
            if cursor < pos:
                seg_end_num = _parse_resnum(residues[pos - 1].id)
                tokens.append(f"{cid}{seg_start_num}-{seg_end_num}")
            tokens.append(f"{gap_len}-{gap_len}")
            cursor = pos
            seg_start_num = _parse_resnum(residues[pos].id) if pos < len(residues) else None
        if cursor < len(residues) and seg_start_num is not None:
            tokens.append(f"{cid}{seg_start_num}-{end_num}")

        chain_groups.append(tokens)
    return ",".join("/".join(group) for group in chain_groups)


def _load_trb(trb_path):
    """RFdiffusionの.trbファイルを安全に読み込む"""
    try:
        with open(trb_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        import torch
        return torch.load(trb_path, map_location="cpu", weights_only=False)


def _merge_designed_region(original_pdb_path, hal_pdb_path, trb_path, work_dir):
    """RFdiffusionの出力(hal)から、新規生成された残基だけを元の全原子構造に差し込む。"""
    trb = _load_trb(trb_path)
    
    hal_idx = [(c, int(r)) for c, r in trb["con_hal_pdb_idx"]]
    ref_idx = [(c, int(r)) for c, r in trb["con_ref_pdb_idx"]]
    hal_to_ref = dict(zip(hal_idx, ref_idx))
    generated_positions = {}

    parser = PDBParser(QUIET=True)
    orig_structure = parser.get_structure("orig", original_pdb_path)
    hal_structure = parser.get_structure("hal", hal_pdb_path)
    orig_model, hal_model = orig_structure[0], hal_structure[0]

    for hal_chain in hal_model:
        hal_cid = hal_chain.id
        ref_cid = next((rc for (hc, _), (rc, _) in zip(hal_idx, ref_idx) if hc == hal_cid), hal_cid)
        if ref_cid not in orig_model:
            continue
        orig_chain = orig_model[ref_cid]

        prev_ref_resnum = None
        pending_new = []

        def _flush(next_ref_resnum):
            if not pending_new:
                return
            if prev_ref_resnum is not None:
                start = prev_ref_resnum + 1
            elif next_ref_resnum is not None:
                start = next_ref_resnum - len(pending_new)
            else:
                start = 1
                
            for offset, hal_res in enumerate(pending_new):
                new_resnum = start + offset
                new_id = (" ", new_resnum, " ")
                
                # 同一残基番号を持つ既存の残基（HETATM等）を安全に一掃
                to_detach = [res.id for res in orig_chain if res.id[1] == new_resnum]
                for rid in to_detach:
                    orig_chain.detach_child(rid)
                    
                hal_res.id = new_id
                orig_chain.add(hal_res)
                generated_positions.setdefault(ref_cid, set()).add(new_resnum)
            pending_new.clear()

        for hal_res in hal_chain:
            key = (hal_cid, hal_res.id[1])
            if key in hal_to_ref:
                _, ref_resnum = hal_to_ref[key]
                _flush(ref_resnum)
                prev_ref_resnum = ref_resnum
            else:
                pending_new.append(hal_res.copy())
        _flush(None)

        orig_chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))

    out_path = os.path.join(work_dir, "rfdiffusion_merged.pdb")
    io = PDBIO()
    io.set_structure(orig_structure)
    io.save(out_path)
    return out_path, generated_positions


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    """
    1. 欠損位置を特定
    2. FASTAから正しい配列を復元
    3. 配列指定付き(provide_seq)でRFdiffusionを実行し、元構造とマージする
    """
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    contig = _build_contig(fixer)

    # ステップ1: 欠損領域の位置と配列の特定
    pdb_complex_residues, generated_resnums_dict, missing_regions = _get_expected_missing_resnums(fixer)
    provide_seq_list = []

    # ステップ2: グローバルマッピングと RFdiffusion用フォーマットへの変換
    if pdb_id and rf_config.get("reassign_sequence_from_fasta") and generated_resnums_dict:
        fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=rf_config.get("fasta_cache_dir"))
        fasta_sequences = parse_rcsb_fasta(fasta_text)

        complex_corrections = recover_complex_sequences(
            pdb_complex_residues=pdb_complex_residues,
            generated_resnums_dict=generated_resnums_dict,
            fasta_sequences=fasta_sequences
        )

        for cid, regions in missing_regions.items():
            if cid not in complex_corrections:
                continue
            corrections = complex_corrections[cid]
            for start_res, end_res in regions:
                seq_chars = []
                for resnum in range(start_res, end_res + 1):
                    # 復元された3文字コードを1文字コードに変換 (不明な場合は安全のため 'G' にフォールバック)
                    resname = corrections.get(resnum, "GLY")
                    seq_chars.append(_THREE_TO_ONE.get(resname, "G"))
                
                seq_str = "".join(seq_chars)
                provide_seq_item = {
                    "res_idx": f"{cid}{start_res}-{end_res}",
                    "seq": seq_str
                }
                provide_seq_list.append(provide_seq_item)

    # ステップ3: コマンド構築と実行
    out_prefix = os.path.join(work_dir, "rfdiffusion_out")
    cmd = [
        "python", rf_config["script_path"],
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={os.path.abspath(pdb_path)}",
        f"inference.model_directory_path={rf_config['model_directory_path']}",
        f"contigmap.contigs=[{contig}]",
        f"inference.num_designs={rf_config.get('num_designs', 1)}",
    ]

    # 復元した配列情報があればJSON化して追加
    if provide_seq_list:
        seq_arg = f"contigmap.provide_seq=['{json.dumps(provide_seq_list)}']"
        cmd.append(seq_arg)

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
        raise RuntimeError(f"RFdiffusion .trb file not found: {trb_path}")

    # 構造のマージ (指定した配列情報に基づく正しいバックボーン・側鎖がPDBに統合される)
    merged_path, _ = _merge_designed_region(pdb_path, hal_pdb_path, trb_path, work_dir)

    return merged_path
