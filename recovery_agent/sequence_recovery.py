# recovery_agent/sequence_recovery.py
import os
import re
import requests
import numpy as np
from scipy.optimize import linear_sum_assignment
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Data.IUPACData import protein_letters_3to1

RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}"

_THREE_TO_ONE = {k.upper(): v for k, v in protein_letters_3to1.items()}
_ONE_TO_THREE = {v: k.upper() for k, v in protein_letters_3to1.items()}


def fetch_rcsb_fasta(pdb_id, cache_dir=None, timeout=30):
    """RCSBから該当PDB IDのFASTA配列を取得する(cache_dir指定時はキャッシュする)"""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{pdb_id}.fasta")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()

    url = RCSB_FASTA_URL.format(pdb_id=pdb_id)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException as e:
        raise RuntimeError(f"RCSB FASTAのダウンロードに失敗しました (pdb_id={pdb_id}, url={url}): {e}")

    if cache_path:
        with open(cache_path, "w") as f:
            f.write(text)
    return text


def parse_rcsb_fasta(fasta_text):
    """RCSBのFASTAテキストから 'チェーンID -> アミノ酸配列(1文字)' の対応表を作る。"""
    sequences = {}
    header, seq_lines = None, []

    def _flush(header, seq_lines):
        if header is None:
            return
        seq = "".join(seq_lines)
        chain_part = header.split("|")[1] if "|" in header else ""
        m = re.search(r"Chains?\s+(.+)", chain_part, re.IGNORECASE)
        if not m:
            return
        chain_ids = [c.strip().split()[0] for c in m.group(1).split(",")]
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
    """'X'(新規生成位置)をどの文字とも同スコア(0)にするPairwiseAligner"""
    letters = "ACDEFGHIKLMNPQRSTVWYX"
    matrix = substitution_matrices.Array(alphabet=letters, dims=2)
    for a in letters:
        for b in letters:
            if a == "X" or b == "X":
                matrix[a, b] = 0
            elif a == b:
                matrix[a, b] = 2
            else:
                matrix[a, b] = -1
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def find_optimal_chain_mapping(pdb_complex_residues, generated_resnums_dict, fasta_sequences, aligner):
    """
    MM-alignベースのチェーン割当最適化:
    PDB構造の全チェーンと、FASTAの全配列の間で総当りのアラインメントスコア行列を計算し、
    ハンガリー法を用いて全体としてのスコアが最大化する「1対1の最適な対応関係」を解く。

    pdb_complex_residues: {pdb_chain_id: {resnum: "GLY", ...}, ...}
    generated_resnums_dict: {pdb_chain_id: {resnum, ...}, ...}
    fasta_sequences: {fasta_chain_id: "SEQUENCE", ...}
    """
    pdb_chain_ids = list(pdb_complex_residues.keys())
    fasta_chain_ids = list(fasta_sequences.keys())

    if not pdb_chain_ids or not fasta_chain_ids:
        return {}

    # コスト行列の初期化 (行: PDBチェーン, 列: FASTA配列)
    # scipyの最適化は最小化問題を解くため、アラインメントスコアをマイナスにする
    cost_matrix = np.zeros((len(pdb_chain_ids), len(fasta_chain_ids)))

    for i, p_cid in enumerate(pdb_chain_ids):
        orig_residues = pdb_complex_residues[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())

        resnums_sorted = sorted(orig_residues.keys())
        if not resnums_sorted:
            cost_matrix[i, :] = 0  # 空のチェーンはスコア0
            continue

        # テンプレート配列（Xを含む）を作成
        template_seq = "".join(
            "X" if resnum in gen_resnums else _THREE_TO_ONE.get(orig_residues[resnum], "X")
            for resnum in resnums_sorted
        )

        for j, f_cid in enumerate(fasta_chain_ids):
            target_seq = fasta_sequences[f_cid]
            # aligner.scoreでアラインメント経路生成をバイパスし高速計算
            score = aligner.score(template_seq, target_seq)
            cost_matrix[i, j] = -score

    # ハンガリー法による線形割当の実行
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # マッピング結果の構築
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        # 必要に応じてここに類似度カットオフ（極端に低いスコアのペアを除外する）を入れることも可能
        mapping[pdb_chain_ids[r]] = fasta_chain_ids[c]

    return mapping


def recover_complex_sequences(pdb_complex_residues, generated_resnums_dict, fasta_sequences):
    """
    複合体全体に対してチェーンの対応を最適化し、すべての新規生成残基を同時にリカバリーする。

    pdb_complex_residues: {pdb_chain_id: {resnum: "GLY", ...}, ...}
    generated_resnums_dict: {pdb_chain_id: {resnum1, ...}, ...}
    fasta_sequences: {fasta_chain_id: "SEQUENCE", ...}
    
    戻り値: {pdb_chain_id: {resnum: "ALA", ...}, ...} (各チェーンの新規生成残基のみの推定結果)
    """
    aligner = _build_wildcard_aligner()

    # 1. MM-align的なアプローチでPDBチェーンとFASTA配列の最適な対応を決定
    mapping = find_optimal_chain_mapping(
        pdb_complex_residues, generated_resnums_dict, fasta_sequences, aligner
    )

    complex_results = {}

    # 2. 決定した最適なマッピングに基づいて各チェーンをアラインメント
    for p_cid, f_cid in mapping.items():
        orig_residues = pdb_complex_residues[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())
        target_seq = fasta_sequences[f_cid]

        resnums_sorted = sorted(orig_residues.keys())
        if not resnums_sorted or not gen_resnums:
            continue

        template_seq = "".join(
            "X" if resnum in gen_resnums else _THREE_TO_ONE.get(orig_residues[resnum], "X")
            for resnum in resnums_sorted
        )

        alignments = aligner.align(template_seq, target_seq)
        if not alignments:
            continue
        alignment = alignments[0]
        aligned_template, aligned_target = str(alignment[0]), str(alignment[1])

        chain_result = {}
        ti = 0  # template_seq(resnums_sorted)側のインデックス
        for a_char, b_char in zip(aligned_template, aligned_target):
            if a_char != "-":
                if a_char == "X" and b_char not in ("-", "X"):
                    resnum = resnums_sorted[ti]
                    if resnum in gen_resnums:
                        chain_result[resnum] = _ONE_TO_THREE.get(b_char.upper(), "GLY")
                ti += 1

        if chain_result:
            complex_results[p_cid] = chain_result

    return complex_results


def map_generated_residues_to_sequence(chain_id, orig_chain_residues, generated_resnums, fasta_sequences):
    """
    （旧APIとの後方互換性用）
    単一のチェーンIDに対する処理要求であっても、裏側では最適化アルゴリズムを回し、
    最もアラインメントスコアの高いFASTA配列を自動選択して残基推定を実行する。
    """
    # 単一チェーンを複合体用のインターフェースにラップして実行
    pdb_complex_residues = {chain_id: orig_chain_residues}
    generated_resnums_dict = {chain_id: generated_resnums}

    results = recover_complex_sequences(pdb_complex_residues, generated_resnums_dict, fasta_sequences)
    return results.get(chain_id, {})
