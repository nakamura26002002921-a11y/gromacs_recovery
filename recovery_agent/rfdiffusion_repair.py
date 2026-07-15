# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
import pickle
from Bio.PDB import PDBParser, PDBIO
from pdbfixer import PDBFixer


def _build_contig(fixer):
    """RFdiffusion用のcontig文字列を作る。1鎖に欠損が複数箇所あっても正しく扱う。"""
    chain_groups = []  # チェーンごとのトークンのリストのリスト
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        cid = chain.id
        start_num, end_num = int(residues[0].id), int(residues[-1].id)

        # このチェーンの欠損箇所を全て集め、位置(pos)の昇順に処理する
        gaps = sorted(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items() if ci == chain.index),
            key=lambda x: x[0],
        )
        if not gaps:
            chain_groups.append([f"{cid}{start_num}-{end_num}"])
            continue

        tokens = []
        cursor = 0            # 直前に処理し終えたresidues内の位置(exclusive)
        seg_start_num = start_num  # 次に出力する既存区間の開始残基番号
        for pos, names in gaps:
            gap_len = len(names)
            if pos == 0:
                # 鎖の先頭が欠損: 既存区間はまだ無いのでgapトークンのみ
                tokens.append(f"{gap_len}-{gap_len}")
                cursor = 0
                continue
            if cursor < pos:
                seg_end_num = int(residues[pos - 1].id)
                tokens.append(f"{cid}{seg_start_num}-{seg_end_num}")
            tokens.append(f"{gap_len}-{gap_len}")
            cursor = pos
            seg_start_num = int(residues[pos].id) if pos < len(residues) else None
        if cursor < len(residues) and seg_start_num is not None:
            tokens.append(f"{cid}{seg_start_num}-{end_num}")

        chain_groups.append(tokens)
    # 同一チェーン内は "/" で連結、チェーン間(=新しい鎖の開始)は "," で区切る
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
                new_id = (" ", start + offset, " ")
                if new_id in orig_chain:
                    orig_chain.detach_child(new_id)
                hal_res.id = new_id
                orig_chain.add(hal_res)
                generated_positions.setdefault(ref_cid, set()).add(new_id[1])
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


def _reassign_generated_sequence(merged_pdb_path, generated_positions, pdb_id, rf_config, work_dir):
    """
    【修正点】
    鎖ごとのループ処理をやめ、複合体全体を一度に `recover_complex_sequences` に渡し、
    ハンガリー法を用いたグローバルな最適マッピングを行います。
    """
    from Bio.PDB import PDBParser, PDBIO
    from recovery_agent.sequence_recovery import fetch_rcsb_fasta, parse_rcsb_fasta, recover_complex_sequences

    # FASTAの取得とパース
    fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=rf_config.get("fasta_cache_dir"))
    fasta_sequences = parse_rcsb_fasta(fasta_text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("merged", merged_pdb_path)
    model = structure[0]

    # 1. 複合体全体の残基辞書を構築
    pdb_complex_residues = {}
    for chain_id in generated_positions.keys():
        if chain_id in model:
            chain = model[chain_id]
            # HETATM等を除外したアミノ酸残基のみを取得
            pdb_complex_residues[chain_id] = {res.id[1]: res.resname for res in chain if res.id[0] == " "}

    # 2. MM-align的思想に基づくグローバル全体最適化を一括で実行
    complex_corrections = recover_complex_sequences(
        pdb_complex_residues=pdb_complex_residues,
        generated_resnums_dict=generated_positions,
        fasta_sequences=fasta_sequences
    )

    # 3. 最適化されたマッピング結果を各鎖に適用
    for chain_id, corrections in complex_corrections.items():
        if chain_id in model:
            chain = model[chain_id]
            for resnum, correct_resname in corrections.items():
                if resnum in chain:
                    chain[resnum].resname = correct_resname

    out_path = os.path.join(work_dir, "rfdiffusion_seq_recovered.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    return out_path


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    """RFdiffusionを実行し、新規生成された欠損部分だけを元構造に統合したPDBのパスを返す。"""
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
        raise RuntimeError(f"RFdiffusion .trb file not found: {trb_path} (side-chain/HETATM-preserving merge requires it)")

    merged_path, generated_positions = _merge_designed_region(pdb_path, hal_pdb_path, trb_path, work_dir)

    if pdb_id and rf_config.get("reassign_sequence_from_fasta") and generated_positions:
        merged_path = _reassign_generated_sequence(merged_path, generated_positions, pdb_id, rf_config, work_dir)

    return merged_path
