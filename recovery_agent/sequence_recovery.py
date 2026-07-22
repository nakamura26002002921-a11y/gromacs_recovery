# recovery_agent/sequence_recovery.py
"""
RFdiffusion出力(GLYまみれ)の複合体PDBに対して、RCSB FASTAとの配列アラインメントに
基づき正しいアミノ酸名を割り当てるモジュール。

RFdiffusionはバックボーン(N,CA,C,O)のみを生成し、新規残基は常にGLYとして出力する
(公式仕様)。本モジュールはRFdiffusion実行後の独立ステップとして、欠損領域に
入るべき正しいアミノ酸種をFASTAから推定し、GLYの残基名を書き換える。
"""
import os
import re
import requests
import numpy as np
from scipy.optimize import linear_sum_assignment
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser, PDBIO
from pdbfixer import PDBFixer

RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}"

_THREE_TO_ONE = {k.upper(): v for k, v in protein_letters_3to1.items()}
_ONE_TO_THREE = {v: k.upper() for k, v in protein_letters_3to1.items()}


def fetch_rcsb_fasta(pdb_id, cache_dir=None, timeout=30):
    """RCSBから該当PDB IDのFASTA配列を取得する (cache_dir指定時はキャッシュする)"""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{pdb_id}.fasta")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

    url = RCSB_FASTA_URL.format(pdb_id=pdb_id)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException as e:
        raise RuntimeError(
            f"RCSB FASTAのダウンロードに失敗しました (pdb_id={pdb_id}, url={url}): {e}"
        )

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def parse_rcsb_fasta(fasta_text):
    """RCSBのFASTAテキストから 'チェーンID -> アミノ酸配列(1文字)' の対応表を作る。"""
    sequences = {}
    header, seq_lines = None, []

    def _flush(hdr, seq_list):
        if hdr is None:
            return
        seq = "".join(seq_list).upper()
        parts = hdr.split("|")
        if len(parts) >= 2:
            chain_part = parts[1]
            m = re.search(r"Chains?\s+([A-Za-z0-9,\s]+)", chain_part, re.IGNORECASE)
            if m:
                raw_chains = m.group(1).split(",")
                chain_ids = [c.strip().split()[0] for c in raw_chains if c.strip()]
                for cid in chain_ids:
                    sequences[cid] = seq

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush(header, seq_lines)
            header, seq_lines = line, []
        else:
            seq_lines.append(line)
    _flush(header, seq_lines)
    return sequences


def _build_wildcard_aligner():
    """'X'(新規生成/欠損位置)をどのアミノ酸とも中立(スコア0.5)でマッチさせるアライナー。"""
    letters = "ACDEFGHIKLMNPQRSTVWYX"
    matrix = substitution_matrices.Array(alphabet=letters, dims=2)
    for a in letters:
        for b in letters:
            if a == "X" or b == "X":
                matrix[a, b] = 0.5
            elif a == b:
                matrix[a, b] = 2.0
            else:
                matrix[a, b] = -1.0

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def _parse_resnum(res_id):
    """'100A' や '-1' のようなPDB残基IDから整数部分のみを抽出する"""
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


def recover_complex_sequences(pdb_complex_residues, generated_resnums_dict, fasta_sequences):
    """
    【MM-alignの目的関数を適用した全体最適化】
    複合体に含まれる全PDB鎖と全FASTA鎖の間でアラインメントスコア行列を作成し、
    ハンガリー法を用いて「複合体全体のアラインメントスコアの総和」が最大となる
    1対1の鎖マッピング π を決定する。

    :param pdb_complex_residues: {chain_id: {resnum: "ALA", ...}, ...} (全鎖の残基データ)
    :param generated_resnums_dict: {chain_id: {resnum, ...}, ...} (全鎖の欠損resnum集合)
    :param fasta_sequences: {chain_id: "MKT..."} (FASTA配列辞書)
    :return: {chain_id: {resnum: "ALA", ...}, ...} (全鎖の推定結果)
    """
    pdb_chain_ids = list(pdb_complex_residues.keys())
    fasta_chain_ids = list(fasta_sequences.keys())

    if not pdb_chain_ids or not fasta_chain_ids:
        return {}

    aligner = _build_wildcard_aligner()

    # 1. 鎖間類似度行列（スコア行列）の作成
    cost_matrix = np.zeros((len(pdb_chain_ids), len(fasta_chain_ids)))

    template_seqs = {}
    sorted_resnums_dict = {}

    for i, p_cid in enumerate(pdb_chain_ids):
        orig_residues = pdb_complex_residues[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())

        # ================================================================
        # 【バグ修正】resnums_sorted に欠損resnumも含める。
        #
        # 旧コード:
        #   resnums_sorted = sorted(orig_residues.keys(), key=_parse_resnum)
        #
        # 問題点:
        #   orig_residues は元PDBのATOMレコード由来のため、欠損残基のresnumを
        #   キーとして持たない。そのため template_seq に 'X'(欠損マーカー)が
        #   一切挿入されず、アラインメントで「FASTAのどの位置が欠損に対応するか」
        #   を特定できず、結果として1つも置換が行われなかった。
        #
        # 修正:
        #   実在残基のresnumと欠損resnumの和集合をソートして使う。
        #   欠損位置には 'X' を、実在位置には実際のアミノ酸1文字コードを割り当てる。
        # ================================================================
        all_resnums = set(orig_residues.keys()) | gen_resnums
        resnums_sorted = sorted(all_resnums, key=_parse_resnum)
        sorted_resnums_dict[p_cid] = resnums_sorted

        if not resnums_sorted:
            template_seqs[p_cid] = ""
            continue

        template_seq = "".join(
            "X" if resnum in gen_resnums
            else _THREE_TO_ONE.get(str(orig_residues[resnum]).upper(), "X")
            for resnum in resnums_sorted
        )
        template_seqs[p_cid] = template_seq

        for j, f_cid in enumerate(fasta_chain_ids):
            target_seq = fasta_sequences[f_cid]
            if template_seq and target_seq:
                score = aligner.score(template_seq, target_seq)
            else:
                score = 0
            cost_matrix[i, j] = -score

    # 2. 鎖マッピング π の最適化（ハンガリー法）
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    optimal_mapping = {}
    for r, c in zip(row_ind, col_ind):
        p_cid = pdb_chain_ids[r]
        f_cid = fasta_chain_ids[c]
        if -cost_matrix[r, c] > 0:
            optimal_mapping[p_cid] = f_cid

    # 3. 確定した最適マッピングに基づく残基の復元
    complex_results = {}
    for p_cid, f_cid in optimal_mapping.items():
        template_seq = template_seqs[p_cid]
        target_seq = fasta_sequences[f_cid]
        resnums_sorted = sorted_resnums_dict[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())

        if not template_seq or not target_seq or not gen_resnums:
            continue

        alignments = aligner.align(template_seq, target_seq)
        if not alignments:
            continue

        best_aln = alignments[0]
        aligned_template = str(best_aln[0])
        aligned_target = str(best_aln[1])

        mapping = {}
        ti = 0

        for a_char, b_char in zip(aligned_template, aligned_target):
            if a_char != "-":
                current_resnum = resnums_sorted[ti]
                if current_resnum in gen_resnums:
                    if b_char not in ("-", "X") and b_char.isalpha():
                        mapping[current_resnum] = _ONE_TO_THREE.get(b_char.upper(), "GLY")
                ti += 1

        if mapping:
            complex_results[p_cid] = mapping

    return complex_results


def _get_generated_resnums_from_original(original_pdb_path):
    """
    RFdiffusion実行前の(まだ欠損がある)元PDBに対してPDBFixerの欠損検出を走らせ、
    「どのチェーンのどの残基番号がRFdiffusionによって新規生成される予定だったか」を
    復元する。rfdiffusion_repair.py の _get_expected_missing_resnums と同じロジック。

    PDBFixer.findMissingResidues() は SEQRES レコードと ATOM レコードの差分から
    欠損を検出する。したがって元PDBには正しいSEQRESレコードが存在する必要がある。

    :return: pdb_complex_residues, generated_resnums_dict
    """
    fixer = PDBFixer(filename=original_pdb_path)
    fixer.findMissingResidues()

    pdb_complex_residues = {}
    generated_resnums_dict = {}

    for chain in fixer.topology.chains():
        cid = chain.id
        residues = list(chain.residues())
        if not residues:
            continue
        pdb_complex_residues[cid] = {}
        for res in residues:
            pdb_complex_residues[cid][_parse_resnum(res.id)] = res.name

    for chain in fixer.topology.chains():
        cid = chain.id
        residues = list(chain.residues())
        gaps = sorted(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items()
             if ci == chain.index),
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
            generated_resnums_dict[cid] = gen_resnums

    return pdb_complex_residues, generated_resnums_dict


def apply_sequence_recovery(original_pdb_path, rfdiffusion_pdb_path, work_dir, pdb_id,
                            out_name="sequence_recovered.pdb", cache_dir=None):
    """
    RFdiffusion実行後(新規残基が全てGLYの状態)の複合体PDBに対して、
    RCSB FASTAとの配列アラインメントに基づき正しいアミノ酸名を割り当てる。

    :param original_pdb_path: RFdiffusion実行前の(欠損が残っている)元のPDB
    :param rfdiffusion_pdb_path: RFdiffusionの出力をマージ済みの複合体PDB(新規残基はGLY)
    :param work_dir: 出力先ディレクトリ
    :param pdb_id: RCSB FASTAを取得するためのPDB ID
    :param out_name: 出力ファイル名
    :param cache_dir: FASTAキャッシュディレクトリ(任意)
    :return: 配列復元後のPDBファイルパス。復元対象がない場合はrfdiffusion_pdb_pathをそのまま返す。
    """
    pdb_complex_residues, generated_resnums_dict = _get_generated_resnums_from_original(
        original_pdb_path
    )

    if not generated_resnums_dict or not pdb_id:
        return rfdiffusion_pdb_path

    fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=cache_dir)
    fasta_sequences = parse_rcsb_fasta(fasta_text)

    complex_corrections = recover_complex_sequences(
        pdb_complex_residues=pdb_complex_residues,
        generated_resnums_dict=generated_resnums_dict,
        fasta_sequences=fasta_sequences,
    )

    if not complex_corrections:
        return rfdiffusion_pdb_path

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("seqrec", rfdiffusion_pdb_path)
    model = structure[0]

    for cid, resnum_to_resname in complex_corrections.items():
        if cid not in model:
            continue
        chain = model[cid]
        for resnum, correct_resname in resnum_to_resname.items():
            for res in chain:
                if res.id[1] == resnum and res.id[0] == " ":
                    res.resname = correct_resname
                    break

    out_path = os.path.join(work_dir, out_name)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)

    return out_path


def map_generated_residues_to_sequence(chain_id, orig_chain_residues, generated_resnums,
                                       fasta_sequences):
    """
    (後方互換性用ラッパー関数)
    単一鎖の処理として呼ばれた場合でも、内部で全体最適化関数を利用する。
    """
    pdb_complex = {chain_id: orig_chain_residues}
    gen_resnums_dict = {chain_id: generated_resnums}
    results = recover_complex_sequences(pdb_complex, gen_resnums_dict, fasta_sequences)
    return results.get(chain_id, {})
