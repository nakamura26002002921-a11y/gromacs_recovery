# recovery_agent/sequence_recovery.py
"""
RFdiffusion出力(GLYまみれ)の複合体PDBに正しいアミノ酸名を割り当てる。

【設計方針】
アラインメント(Biopython PairwiseAligner)は使わない。
代わりに「アンカーベースの位置マッピング」を主アルゴリズムとする。

理由:
- BiopythonのPairwiseAlignerはglobal+カスタム行列+end_gap_score=0で
  破綻したアラインメントを返すことがある(1AONで実証)
- PDBとFASTAが同じタンパク質を表現している場合、非欠損残基をアンカーに
  すればオフセットは一意に決まり、アラインメントは不要
- 同一配列のホモマー(GroEL 14量体など)でも確実に全鎖同一結果になる
"""
import os
import re
import requests
import numpy as np
from scipy.optimize import linear_sum_assignment
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
        raise RuntimeError(f"RCSB FASTA download failed (pdb_id={pdb_id}): {e}")
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def parse_rcsb_fasta(fasta_text):
    sequences = {}
    header, seq_lines = None, []

    def _flush(hdr, seq_list):
        if hdr is None:
            return
        seq = "".join(seq_list).upper()
        parts = hdr.split("|")
        if len(parts) >= 2:
            m = re.search(r"Chains?\s+([A-Za-z0-9,\s]+)", parts[1], re.IGNORECASE)
            if m:
                for c in m.group(1).split(","):
                    cid = c.strip().split()[0] if c.strip() else ""
                    if cid:
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
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


# ============================================================================
# コア: アンカーベース位置マッピング (アラインメント不使用)
# ============================================================================

def _find_offset_by_anchor(template_seq, target_seq, anchor_len=15):
    """
    テンプレート先頭の非X残基(アンカー)をFASTA内で探索し、
    テンプレート位置 → FASTA位置 のオフセットを決定する。

    :param template_seq: "AAKDVK...XXXXX" (X=欠損)
    :param target_seq:   "AAKDVK..." (FASTA配列)
    :param anchor_len:   アンカーに使う残基数
    :return: offset (int) または None
    """
    # テンプレート先頭から連続する非X残基をアンカーとして抽出
    anchor = []
    for ch in template_seq:
        if ch == "X":
            break
        anchor.append(ch)
        if len(anchor) >= anchor_len:
            break

    if len(anchor) < 5:
        # 先頭がXで始まる場合、末尾から試す
        anchor = []
        for ch in reversed(template_seq):
            if ch == "X":
                break
            anchor.append(ch)
            if len(anchor) >= anchor_len:
                break
        anchor = anchor[::-1]
        if len(anchor) < 5:
            return None
        # 末尾アンカーのオフセットを計算
        anchor_str = "".join(anchor)
        # テンプレート内でのアンカー開始位置
        t_start = len(template_seq) - len(anchor)
        pos = target_seq.find(anchor_str)
        if pos >= 0:
            return pos - t_start
        # 部分マッチ (アンカーの先頭10残基で再試行)
        partial = anchor_str[:10]
        pos = target_seq.find(partial)
        if pos >= 0:
            return pos - t_start
        return None

    anchor_str = "".join(anchor)
    # 完全マッチ
    pos = target_seq.find(anchor_str)
    if pos >= 0:
        return pos  # テンプレート位置0 → FASTA位置pos

    # 部分マッチ (アンカーの先頭10残基)
    partial = anchor_str[:10]
    pos = target_seq.find(partial)
    if pos >= 0:
        return pos

    # 部分マッチ (アンカーの先頭7残基)
    partial = anchor_str[:7]
    pos = target_seq.find(partial)
    if pos >= 0:
        return pos

    return None


def _recover_chain_by_positional_mapping(template_seq, target_seq, resnums_sorted, gen_resnums):
    """
    アンカーベースの位置マッピングで1鎖の欠損残基を復元する。

    1. テンプレート先頭の非X残基でFASTA内のオフセットを決定
    2. オフセットを適用して全欠損位置のFASTA残基を読み取る
    3. 検証: 非欠損位置の一致率をチェック
    """
    if not template_seq or not target_seq or not gen_resnums:
        return {}

    offset = _find_offset_by_anchor(template_seq, target_seq)
    if offset is None:
        return {}

    # 検証: 非X位置の一致率
    match_count = 0
    check_count = 0
    for i, ch in enumerate(template_seq):
        if ch != "X":
            tp = offset + i
            if 0 <= tp < len(target_seq):
                check_count += 1
                if ch == target_seq[tp]:
                    match_count += 1

    identity = match_count / max(check_count, 1)
    if identity < 0.80:
        # オフセットが間違っている可能性 → 全オフセットをブルートフォース探索
        best_offset = offset
        best_identity = identity
        for trial_offset in range(-5, len(target_seq) - len(template_seq) + 6):
            mc = 0
            cc = 0
            for i, ch in enumerate(template_seq):
                if ch != "X":
                    tp = trial_offset + i
                    if 0 <= tp < len(target_seq):
                        cc += 1
                        if ch == target_seq[tp]:
                            mc += 1
            trial_id = mc / max(cc, 1)
            if trial_id > best_identity:
                best_identity = trial_id
                best_offset = trial_offset
        offset = best_offset
        identity = best_identity

    if identity < 0.50:
        return {}

    # マッピング実行
    mapping = {}
    for i, resnum in enumerate(resnums_sorted):
        if resnum in gen_resnums:
            tp = offset + i
            if 0 <= tp < len(target_seq):
                aa = target_seq[tp]
                if aa.isalpha():
                    mapping[resnum] = _ONE_TO_THREE.get(aa, "GLY")
    return mapping


# ============================================================================
# 複合体全体の最適化 (ハンガリー法で鎖マッピング)
# ============================================================================

def recover_complex_sequences(pdb_complex_residues, generated_resnums_dict, fasta_sequences):
    """
    MM-align的思想: ハンガリー法で複合体全体の鎖マッピングを最適化。
    各鎖の復元はアンカーベース位置マッピング(アラインメント不使用)。
    """
    pdb_chain_ids = list(pdb_complex_residues.keys())
    fasta_chain_ids = list(fasta_sequences.keys())

    if not pdb_chain_ids or not fasta_chain_ids:
        return {}

    # --- テンプレート構築 ---
    template_seqs = {}
    sorted_resnums_dict = {}

    for p_cid in pdb_chain_ids:
        orig_residues = pdb_complex_residues[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())
        all_resnums = set(orig_residues.keys()) | gen_resnums
        resnums_sorted = sorted(all_resnums, key=_parse_resnum)
        sorted_resnums_dict[p_cid] = resnums_sorted

        if not resnums_sorted:
            template_seqs[p_cid] = ""
            continue

        template_seq = "".join(
            "X" if rn in gen_resnums
            else _THREE_TO_ONE.get(str(orig_residues[rn]).upper(), "X")
            for rn in resnums_sorted
        )
        template_seqs[p_cid] = template_seq

    # --- コスト行列: アンカーマッチ数でスコアリング ---
    # (アラインメントスコアの代わりに、アンカーの完全マッチ長を使用)
    cost_matrix = np.zeros((len(pdb_chain_ids), len(fasta_chain_ids)))

    for i, p_cid in enumerate(pdb_chain_ids):
        t_seq = template_seqs[p_cid]
        if not t_seq:
            continue
        for j, f_cid in enumerate(fasta_chain_ids):
            f_seq = fasta_sequences[f_cid]
            if not f_seq:
                continue
            # スコア = 長さ類似度 + アンカーマッチ
            len_score = -abs(len(t_seq) - len(f_seq))
            offset = _find_offset_by_anchor(t_seq, f_seq, anchor_len=10)
            anchor_score = 0
            if offset is not None:
                # アンカー位置の一致数をカウント
                for k, ch in enumerate(t_seq[:20]):
                    if ch != "X":
                        tp = offset + k
                        if 0 <= tp < len(f_seq) and ch == f_seq[tp]:
                            anchor_score += 1
            cost_matrix[i, j] = -(len_score + anchor_score * 10)

    # --- ハンガリー法 ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal_mapping = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 0:
            optimal_mapping[pdb_chain_ids[r]] = fasta_chain_ids[c]

    # --- 各鎖の復元 (位置マッピング) ---
    complex_results = {}
    for p_cid, f_cid in optimal_mapping.items():
        t_seq = template_seqs[p_cid]
        f_seq = fasta_sequences[f_cid]
        resnums_sorted = sorted_resnums_dict[p_cid]
        gen_resnums = generated_resnums_dict.get(p_cid, set())

        mapping = _recover_chain_by_positional_mapping(
            t_seq, f_seq, resnums_sorted, gen_resnums
        )
        if mapping:
            complex_results[p_cid] = mapping

    return complex_results


# ============================================================================
# 欠損検出
# ============================================================================

def _get_generated_resnums_from_original(original_pdb_path):
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
    pdb_complex = {chain_id: orig_chain_residues}
    gen_dict = {chain_id: generated_resnums}
    results = recover_complex_sequences(pdb_complex, gen_dict, fasta_sequences)
    return results.get(chain_id, {})
