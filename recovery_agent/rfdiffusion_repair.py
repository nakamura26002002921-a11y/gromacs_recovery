# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
from pdbfixer import PDBFixer
from Bio.PDB import PDBParser, PDBIO


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
    # RFdiffusionの.trbファイルは torch.save() ではなく通常の pickle.dump() で保存されるため、
    # torch.load() (zip/tar形式を前提とするパーサ) では "Invalid magic number; corrupt file?" になる。
    # まず素のpickleとして読み、失敗した場合のみtorch.loadにフォールバックする。
    import pickle
    try:
        with open(trb_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        import torch
        return torch.load(trb_path, map_location="cpu", weights_only=False)


def _merge_designed_region(original_pdb_path, hal_pdb_path, trb_path, work_dir):
    """RFdiffusionの出力(hal)は骨格原子のみ・複合体全体を作り直したものであり、
    側鎖やHETATM(水・イオン・補因子等)を含まない。そのまま採用すると構造の大部分を
    失ってしまうため、.trbの対応表(con_hal_pdb_idx <-> con_ref_pdb_idx)を使って
    「実際に新規生成された(=対応表に無い)残基」だけを元の全原子構造に差し込む。

    注意: hal側の残基番号はRFdiffusionが全長(固定領域+新規生成領域)を通し番号で
    振り直したものであり、元のPDBの番号(欠番のあるオリジナル番号)とは無関係。
    新規残基をそのまま元の鎖に足すと番号がずれてしまうため、con_ref_pdb_idxで
    分かる前後の既存残基の「元の番号」から補間して正しい番号を割り当てる。

    戻り値: (統合後のPDBパス, generated_positions)
      generated_positions: {ref_chain_id: {resnum, ...}, ...}
        RFdiffusionが新規生成した(=本来のアミノ酸種が不明な。バックボーンのみでGLY)
        残基の位置。後段の配列復元(sequence_recovery)で使用する。
    """
    trb = _load_trb(trb_path)
    hal_idx = [(c, int(r)) for c, r in trb["con_hal_pdb_idx"]]
    ref_idx = [(c, int(r)) for c, r in trb["con_ref_pdb_idx"]]
    hal_to_ref = dict(zip(hal_idx, ref_idx))  # hal(出力側)の(鎖,番号) -> ref(元)の(鎖,番号)
    generated_positions = {}

    parser = PDBParser(QUIET=True)
    orig_structure = parser.get_structure("orig", original_pdb_path)
    hal_structure = parser.get_structure("hal", hal_pdb_path)
    orig_model, hal_model = orig_structure[0], hal_structure[0]

    for hal_chain in hal_model:
        hal_cid = hal_chain.id
        # このhal鎖が元のどの鎖に対応するかは、con_hal_pdb_idx/con_ref_pdb_idxのマッピングから判定する
        ref_cid = next((rc for (hc, _), (rc, _) in zip(hal_idx, ref_idx) if hc == hal_cid), hal_cid)
        if ref_cid not in orig_model:
            continue  # 新規鎖の追加には未対応(既存鎖内の欠損補完のみサポート)
        orig_chain = orig_model[ref_cid]

        prev_ref_resnum = None   # 直前に確定した既存残基の「元の番号」
        pending_new = []         # まだ番号が確定していない新規残基(hal順)

        def _flush(next_ref_resnum):
            if not pending_new:
                return
            if prev_ref_resnum is not None:
                start = prev_ref_resnum + 1
            elif next_ref_resnum is not None:
                start = next_ref_resnum - len(pending_new)
            else:
                start = 1  # 前後どちらの手がかりも無い場合の最終手段
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
                _flush(ref_resnum)         # ここまで溜まっていた新規残基に番号を確定させる
                prev_ref_resnum = ref_resnum
            else:
                pending_new.append(hal_res.copy())
        _flush(None)

        orig_chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))  # resseq順に並べ直す

    out_path = os.path.join(work_dir, "rfdiffusion_merged.pdb")
    io = PDBIO()
    io.set_structure(orig_structure)
    io.save(out_path)
    return out_path, generated_positions


def _reassign_generated_sequence(merged_pdb_path, generated_positions, pdb_id, rf_config, work_dir):
    """RFdiffusionが新規生成した残基(バックボーンのみ・GLY)について、RCSBの正式配列と
    アラインメントして本来のアミノ酸種を推定し、resnameを付け替える。
    側鎖原子は(GLYのものしか無い=事実上ないので)そのまま残し、後段のPDBFixerの
    addMissingAtoms()に正しい残基名に基づいた側鎖の再構築を任せる。
    """
    from Bio.PDB import PDBParser, PDBIO
    from recovery_agent.sequence_recovery import fetch_rcsb_fasta, parse_rcsb_fasta, map_generated_residues_to_sequence

    fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=rf_config.get("fasta_cache_dir"))
    fasta_sequences = parse_rcsb_fasta(fasta_text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("merged", merged_pdb_path)
    model = structure[0]

    for chain_id, gen_resnums in generated_positions.items():
        if chain_id not in model:
            continue
        chain = model[chain_id]
        orig_chain_residues = {res.id[1]: res.resname for res in chain if res.id[0] == " "}
        corrections = map_generated_residues_to_sequence(chain_id, orig_chain_residues, gen_resnums, fasta_sequences)
        for resnum, correct_resname in corrections.items():
            if resnum in chain:
                chain[resnum].resname = correct_resname

    out_path = os.path.join(work_dir, "rfdiffusion_seq_recovered.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    return out_path


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    """RFdiffusionを実行し、新規生成された欠損部分だけを元構造に統合したPDBのパスを返す。

    RFdiffusionは骨格(バックボーン)しか設計せず、新規生成領域は慣例的に全てGLYとして
    出力される(アミノ酸の種類自体は設計しない)。rf_config['reassign_sequence_from_fasta']
    が有効かつpdb_idが与えられている場合は、RCSBの正式配列とアラインメントして
    新規生成位置の本来のアミノ酸種を復元する。
    """
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

    merged_path, generated_positions = _merge_designed_region(pdb_path, hal_pdb_path, trb_path, work_dir)

    if pdb_id and rf_config.get("reassign_sequence_from_fasta") and generated_positions:
        merged_path = _reassign_generated_sequence(merged_path, generated_positions, pdb_id, rf_config, work_dir)

    return merged_path

