# recovery_agent/modeller_minimize.py
#
# RFdiffusion(6残基以上の欠損)またはPDBFixer(1〜5残基の欠損)による修復の直後、
# gmx pdb2gmx に渡す前の最終ステップとして、MODELLERで局所的なエネルギー極小化を行う。
#
# 背景:
#   - RFdiffusion経路: sequence_recovery.py でGLYを正しい残基名に置換した後、
#     PDBFixerが側鎖原子を機械的に追加するだけなので、側鎖同士の衝突や
#     不自然なねじれ角が残ったままになりうる。
#   - PDBFixer経路(1〜5残基): 欠損原子・欠損残基をPDBFixerのテンプレートベースで
#     機械的に埋めているだけで、同様にエネルギー的な妥当性は保証されない。
#
#   どちらの経路でも「新規に座標が追加された領域」を中心に、MODELLERの
#   conjugate gradients + MD annealing による極小化(optimize)をかけることで、
#   pdb2gmx がクラッシュ原子や異常な結合長/角度で失敗する確率を下げる。
#
# 注意:
#   MODELLERは全体構造を再モデリングする(automodel)のではなく、既存座標を
#   出発点とした極小化のみに使う。実験構造由来の座標まで大きく動かさないよう、
#   極小化対象は「新規生成/補完された残基とその近傍」に限定するのが望ましい。

import os
import re


def _parse_resnum(res_id):
    """'100A' や '-1' のようなPDB残基IDから整数部分のみを抽出する"""
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


def _get_repaired_resnums(original_pdb_path):
    """
    極小化前(欠損情報を保持している段階)の元PDBから、
    「どのチェーンのどの残基番号が今回の修復で新規に追加/補完されたか」を求める。
    MODELLERのselection対象をこの範囲(+近傍)に絞り込むために使う。

    :return: {chain_id: set(resnum, ...), ...}
    """
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=original_pdb_path)
    fixer.findMissingResidues()

    repaired_resnums = {}
    for chain in fixer.topology.chains():
        cid = chain.id
        residues = list(chain.residues())
        if not residues:
            continue

        gaps = sorted(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items() if ci == chain.index),
            key=lambda x: x[0],
        )

        gen_resnums = set()
        for pos, names in gaps:
            gap_len = len(names)
            if pos == 0:
                start = _parse_resnum(residues[0].id) - gap_len
            else:
                prev_resnum = _parse_resnum(residues[pos - 1].id)
                start = prev_resnum + 1
            end = start + gap_len - 1
            for resnum in range(start, end + 1):
                gen_resnums.add(resnum)

        if gen_resnums:
            repaired_resnums[cid] = gen_resnums

    return repaired_resnums


def minimize_with_modeller(original_pdb_path, repaired_pdb_path, work_dir, modeller_config,
                            out_name="modeller_minimized.pdb"):
    """
    修復済みPDB(repaired_pdb_path)に対し、新規生成/補完された残基とその近傍のみを
    対象としてMODELLERで局所エネルギー極小化を行う。

    :param original_pdb_path: 修復前(欠損情報が残っている)の元PDB。極小化対象の特定に使用。
    :param repaired_pdb_path: RFdiffusion+配列復元+側鎖補完後、またはPDBFixer補完後のPDB。
    :param work_dir: 作業ディレクトリ。
    :param modeller_config: config.yaml の `modeller` セクション。
        - enabled (bool): 極小化を実行するか
        - license_key (str): MODELLERのライセンスキー
        - md_level (str): "very_fast" | "fast" | "slow" | "very_slow" (refineレベル)
        - neighbor_window (int): 極小化対象に含める前後残基数のマージン
        - timeout_sec (int): 極小化処理のタイムアウト
    :param out_name: 出力ファイル名
    :return: 極小化後のPDBファイルパス。対象残基がない、またはenabled=falseの場合は
        repaired_pdb_path をそのまま返す。
    """
    if not modeller_config or not modeller_config.get("enabled", False):
        return repaired_pdb_path

    repaired_resnums = _get_repaired_resnums(original_pdb_path)
    if not repaired_resnums:
        return repaired_pdb_path

    # MODELLERは重いネイティブ拡張(_modeller)を持つため、実行時import。
    # 未インストール環境でも本モジュールのimport自体は失敗しないようにする。
    try:
        from modeller import Environ, Selection, log
        from modeller.optimizers import ConjugateGradients, MolecularDynamics, actions
        from modeller.scripts import complete_pdb
    except ImportError as e:
        raise RuntimeError(
            "MODELLERがインストールされていません。environment.ymlにMODELLERを追加し、"
            "ライセンスキーを設定してください。"
        ) from e

    log.none()
    env = Environ()
    license_key = modeller_config.get("license_key")
    if license_key:
        env.io.license_key = license_key
    env.io.atom_files_directory = [work_dir, "."]
    env.libs.topology.read(file="$(LIB)/top_heav.lib")
    env.libs.parameters.read(file="$(LIB)/par.lib")

    abs_repaired_path = os.path.abspath(repaired_pdb_path)
    mdl = complete_pdb(env, abs_repaired_path)

    # 極小化対象: 新規生成/補完された残基 ± neighbor_window
    window = modeller_config.get("neighbor_window", 3)
    selection_residues = []
    for chain in mdl.chains:
        cid = chain.name.strip()
        if cid not in repaired_resnums:
            continue
        target_nums = repaired_resnums[cid]
        expanded = set()
        for n in target_nums:
            for w in range(-window, window + 1):
                expanded.add(n + w)
        for residue in chain.residues:
            try:
                resnum = int(str(residue.num).strip())
            except ValueError:
                continue
            if resnum in expanded:
                selection_residues.append(residue)

    if not selection_residues:
        return repaired_pdb_path

    atmsel = Selection(*selection_residues)

    # 1. Conjugate Gradientsで大きな衝突・歪みを素早く解消
    cg_iterations = modeller_config.get("cg_iterations", 200)
    cg = ConjugateGradients(output="NO_REPORT")
    cg.optimize(atmsel, max_iterations=cg_iterations)

    # 2. 短時間のMD annealingで局所安定構造へ緩和
    md_iterations = modeller_config.get("md_iterations", 200)
    md = MolecularDynamics(output="NO_REPORT")
    md.optimize(
        atmsel,
        temperature=300,
        max_iterations=md_iterations,
        actions=[actions.trace(10)],
    )

    # 3. 仕上げにもう一度CGでエネルギーを下げ切る
    cg.optimize(atmsel, max_iterations=cg_iterations)

    out_path = os.path.join(work_dir, out_name)
    mdl.write(file=out_path)

    return out_path
