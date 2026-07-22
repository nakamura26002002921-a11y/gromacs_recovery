# recovery_agent/sequence_recovery.py
"""
RFdiffusion出力(GLYまみれ)の複合体PDBに対して、RCSB FASTAとの配列アラインメントに
基づき正しいアミノ酸名を割り当てるモジュール。

【設計原則】
- RFdiffusionはバックボーン(N,CA,C,O)のみ生成し、新規残基は常にGLY(公式仕様)
- 本モジュールはFASTAとのアラインメントで「GLYに入るべき正しいアミノ酸」を推定
- 鎖マッピングはMM-align的思想(ハンガリー法)で複合体全体を最適化
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


# ============================================================================
# FASTA取得・パース
# ============================================================================

def fetch_rcsb_fasta(pdb_id, cache_dir=None, timeout=30):
    """RCSBからFASTA取得 (キャッシュ対応)"""
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
            f"RCSB FASTAのダウンロードに失敗 (pdb_id={pdb_id}, url={url}): {e}"
        )

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def parse_rcsb_fasta(fasta_text):
    """RCSB FASTA → {chain_id: sequence} 辞書"""
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


def _parse_resnum(res_id):
    """'100A' や '-1' から整数部分を抽出"""
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


# ============================================================================
# アライナー
# ============================================================================

def _build_aligner():
    """
    Global alignment アライナー (end gap penalty = 0 で semi-global 的挙動)。
    """
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
    aligner.mode = "global"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    # 末端ギャップはペナルティなし
    # Biopython >= 1.85 で属性名が変更されたため両対応
    try:
        aligner.end_insertion_score = 0   # 新: target_end_gap_score の後継
        aligner.end_deletion_score = 0    # 新: query_end_gap_score の後継
    except AttributeError:
        aligner.target_end_gap_score = 0  # 旧
        aligner.query_end_gap_score = 0   # 旧

    return aligner


# ============================================================================
# コア: 配列復元
# ============================================================================

def recover_complex_sequences(pdb_complex_residues, generated_resnums_dict, fasta_sequences):
    """
    MM-align的思想: ハンガリー法で複合体全体の鎖マッピングを最適化し、
    各鎖の欠損位置に正しいアミノ酸を割り当てる。

    :param pdb_complex_residues: {chain_id: {resnum: resname}}
    :param generated_resnums_dict: {chain_id: {resnum, ...}} (欠損resnum集合)
    :param fasta_sequences: {chain_id: "MKT..."}
    :return: {chain_id: {resnum: "ALA", ...}}
    """
    pdb_chain_ids = list(pdb_complex_residues.keys())
    fasta_chain_ids = list(fasta_sequences.keys())

    if not pdb_chain_ids or not fasta_chain_ids:
        return {}

    aligner = _build_aligner()

    # --- テンプレート配列の構築 ---
    template_seqs = {}
    sorted_resnums_dict = {}

    for p_cid in pdb_chain_ids:
        orig_residues = pdb_complex_residues[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())

        # 実在残基 + 欠損残基の和集合をソート
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

    # --- 重複排除: 同一テンプレートは1回だけアラインメント ---
    # (1AONのように14鎖が同一配列の場合、アラインメントを14回繰り返すと
    #  Biopythonの内部タイブレークで異なる結果が返る可能性がある。
    #  同一テンプレート×同一ターゲットの結果をキャッシュして再利用する)
    alignment_cache = {}

    def _get_alignment(t_seq, f_seq):
        key = (t_seq, f_seq)
        if key not in alignment_cache:
            alns = aligner.align(t_seq, f_seq)
            alignment_cache[key] = alns[0] if alns else None
        return alignment_cache[key]

    def _get_score(t_seq, f_seq):
        key = ("score", t_seq, f_seq)
        if key not in alignment_cache:
            alignment_cache[key] = aligner.score(t_seq, f_seq)
        return alignment_cache[key]

    # --- 1. コスト行列の構築 ---
    cost_matrix = np.zeros((len(pdb_chain_ids), len(fasta_chain_ids)))
    for i, p_cid in enumerate(pdb_chain_ids):
        t_seq = template_seqs[p_cid]
        for j, f_cid in enumerate(fasta_chain_ids):
            f_seq = fasta_sequences[f_cid]
            if t_seq and f_seq:
                cost_matrix[i, j] = -_get_score(t_seq, f_seq)

    # --- 2. ハンガリー法で最適鎖マッピング ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal_mapping = {}
    for r, c in zip(row_ind, col_ind):
        if -cost_matrix[r, c] > 0:
            optimal_mapping[pdb_chain_ids[r]] = fasta_chain_ids[c]

    # --- 3. 各鎖の欠損位置にアミノ酸を割り当て ---
    complex_results = {}
    for p_cid, f_cid in optimal_mapping.items():
        t_seq = template_seqs[p_cid]
        f_seq = fasta_sequences[f_cid]
        resnums_sorted = sorted_resnums_dict[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())

        if not t_seq or not f_seq or not gen_resnums:
            continue

        best_aln = _get_alignment(t_seq, f_seq)
        if best_aln is None:
            continue

        aligned_template = str(best_aln[0])
        aligned_target = str(best_aln[1])

        # 検証: 非X位置の一致率
        match_count = 0
        non_x_count = 0
        for a_ch, b_ch in zip(aligned_template, aligned_target):
            if a_ch != "-" and a_ch != "X" and b_ch != "-":
                non_x_count += 1
                if a_ch == b_ch:
                    match_count += 1

        identity = match_count / max(non_x_count, 1)
        if identity < 0.50:
            print(f"  [Warning] Chain {p_cid}: alignment identity {identity:.1%} "
                  f"is too low. Using positional fallback.")
            mapping = _positional_fallback(t_seq, f_seq, resnums_sorted, gen_resnums)
        else:
            mapping = _extract_mapping(aligned_template, aligned_target,
                                       resnums_sorted, gen_resnums)

        if mapping:
            complex_results[p_cid] = mapping

    return complex_results


def _extract_mapping(aligned_template, aligned_target, resnums_sorted, gen_resnums):
    """アラインメント結果から欠損位置→アミノ酸のマッピングを抽出"""
    mapping = {}
    ti = 0
    for a_char, b_char in zip(aligned_template, aligned_target):
        if a_char != "-":
            current_resnum = resnums_sorted[ti]
            if current_resnum in gen_resnums:
                if b_char not in ("-", "X") and b_char.isalpha():
                    mapping[current_resnum] = _ONE_TO_THREE.get(b_char.upper(), "GLY")
            ti += 1
    return mapping


def _positional_fallback(template_seq, target_seq, resnums_sorted, gen_resnums):
    """
    アラインメントが破綻した場合のフォールバック。
    テンプレート先頭の非X残基でオフセットを推定し、位置ベースで直接マッピング。
    """
    len_t = len(template_seq)
    len_f = len(target_seq)
    if len_t == 0 or len_f == 0:
        return {}

    # 先頭20残基(非X)でオフセットを探索
    probe_len = min(20, len_t)
    probe = template_seq[:probe_len]
    best_offset = 0
    best_score = -1

    search_range = range(max(0, -5), min(len_f, len_t + 5))
    for offset in search_range:
        score = 0
        for k in range(probe_len):
            tp = offset + k
            if 0 <= tp < len_f and probe[k] != "X":
                if probe[k] == target_seq[tp]:
                    score += 1
        if score > best_score:
            best_score = score
            best_offset = offset

    mapping = {}
    for i, resnum in enumerate(resnums_sorted):
        if resnum in gen_resnums:
            target_pos = best_offset + i
            if 0 <= target_pos < len_f:
                aa = target_seq[target_pos]
                if aa.isalpha():
                    mapping[resnum] = _ONE_TO_THREE.get(aa, "GLY")
    return mapping


# ============================================================================
# 欠損検出
# ============================================================================

def _get_generated_resnums_from_original(original_pdb_path):
    """
    元PDBからPDBFixerで欠損を検出。
    rfdiffusion_repair.py の _get_expected_missing_resnums と同じロジック。
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


# ============================================================================
# メインエントリポイント
# ============================================================================

def apply_sequence_recovery(original_pdb_path, rfdiffusion_pdb_path, work_dir, pdb_id,
                            out_name="sequence_recovered.pdb", cache_dir=None):
    """
    RFdiffusion出力(GLYまみれ)に正しいアミノ酸名を割り当てる。

    :param original_pdb_path: RFdiffusion実行前の元PDB (欠損あり)
    :param rfdiffusion_pdb_path: rfdiffusion_final_merged.pdb (GLY埋め済み)
    :param work_dir: 出力先
    :param pdb_id: RCSB PDB ID
    :param out_name: 出力ファイル名
    :param cache_dir: FASTAキャッシュディレクトリ
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
    """(後方互換ラッパー)"""
    pdb_complex = {chain_id: orig_chain_residues}
    gen_dict = {chain_id: generated_resnums}
    results = recover_complex_sequences(pdb_complex, gen_dict, fasta_sequences)
    return results.get(chain_id, {})
