# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
from pdbfixer import PDBFixer


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


def run_rfdiffusion(pdb_path, work_dir, rf_config):
    """RFdiffusionを実行し、欠損部分を補完したPDBのパスを返す"""
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

    new_pdb_path = f"{out_prefix}_0.pdb"
    if not os.path.exists(new_pdb_path):
        raise RuntimeError(f"RFdiffusion output not found: {new_pdb_path}")
    return new_pdb_path

