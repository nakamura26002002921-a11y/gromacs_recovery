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
        raise RuntimeError(f"RCSB FASTAのダウンロードに失敗しました (pdb_id={pdb_id}, url={url}): {e}")

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def parse_rcsb_fasta(fasta_text):
    """RCSBのFASTAテキストから 'チェーンID -> アミノ酸配列(1文字)' の対応表を作る。
    ヘッダの 'Chains A, B' や 'Chain A' などを柔軟にパースする。
    """
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
    # Biopythonのバージョン差異によるTypeErrorを確実に防ぐため、標準のArrayオブジェクトを使用
    matrix = substitution_matrices.Array(alphabet=letters, dims=2)
    for a in letters:
        for b in letters:
            if a == "X" or b == "X":
                matrix[a, b] = 0.5  # Xは何かとマッチする方が、ギャップになるよりマシというスコア
            elif a == b:
                matrix[a, b] = 2.0  # マッチ
            else:
                matrix[a, b] = -1.0 # ミスマッチ

    aligner = PairwiseAligner()
    # localアライメントを使用し、PDBがFASTAの一部(フラグメント)であっても最適マッチさせる
    aligner.mode = "local"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def recover_complex_sequences(pdb_complex_residues, generated_resnums_dict, fasta_sequences):
    """
    【MM-alignの目的関数を適用した全体最適化】
    複合体に含まれる全PDB鎖と全FASTA鎖の間でアラインメントスコア行列を作成し、
    ハンガリー法を用いて「複合体全体のアラインメントスコアの総和」が最大となる
    1対1の鎖マッピング π を決定します。

    :param pdb_complex_residues: {chain_id: {resnum: "GLY", ...}, ...} (全鎖の残基データ)
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
    # scipy.optimizeは最小化を行うため、スコアにマイナスを掛けて格納する
    cost_matrix = np.zeros((len(pdb_chain_ids), len(fasta_chain_ids)))
    
    template_seqs = {}
    sorted_resnums_dict = {}

    for i, p_cid in enumerate(pdb_chain_ids):
        orig_residues = pdb_complex_residues[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())
        
        # 残基番号を文字列としてソート
        resnums_sorted = sorted(orig_residues.keys(), key=lambda x: str(x))
        sorted_resnums_dict[p_cid] = resnums_sorted
        
        if not resnums_sorted:
            template_seqs[p_cid] = ""
            continue

        template_seq = "".join(
            "X" if resnum in gen_resnums else _THREE_TO_ONE.get(str(orig_residues[resnum]).upper(), "X")
            for resnum in resnums_sorted
        )
        template_seqs[p_cid] = template_seq

        # MM-alignの「全鎖ペア間のTM-score計算」に相当
        for j, f_cid in enumerate(fasta_chain_ids):
            target_seq = fasta_sequences[f_cid]
            if template_seq and target_seq:
                score = aligner.score(template_seq, target_seq)
            else:
                score = 0
            cost_matrix[i, j] = -score

    # 2. 鎖マッピング π の最適化（ハンガリー法）
    # 複合体全体のスコア和が最大になる割り当てを解く
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    optimal_mapping = {}
    for r, c in zip(row_ind, col_ind):
        p_cid = pdb_chain_ids[r]
        f_cid = fasta_chain_ids[c]
        if -cost_matrix[r, c] > 0:  # 無意味なゼロスコアペアを除外
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


def map_generated_residues_to_sequence(chain_id, orig_chain_residues, generated_resnums, fasta_sequences):
    """
    (後方互換性用ラッパー関数)
    単一鎖の処理として呼ばれた場合でも、内部で全体最適化関数を利用します。
    ※真のグローバル最適化を行うには、この関数を鎖ごとに呼ぶのではなく、
    直接 recover_complex_sequences() に複合体全体の辞書を渡すことを推奨します。
    """
    pdb_complex = {chain_id: orig_chain_residues}
    gen_resnums_dict = {chain_id: generated_resnums}
    
    results = recover_complex_sequences(pdb_complex, gen_resnums_dict, fasta_sequences)
    return results.get(chain_id, {})
