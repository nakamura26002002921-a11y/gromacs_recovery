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
    """
    trb = _load_trb(trb_path)
    hal_idx = [(c, int(r)) for c, r in trb["con_hal_pdb_idx"]]
    ref_idx = [(c, int(r)) for c, r in trb["con_ref_pdb_idx"]]
    hal_to_ref = dict(zip(hal_idx, ref_idx))  # hal(出力側)の(鎖,番号) -> ref(元)の(鎖,番号)

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

