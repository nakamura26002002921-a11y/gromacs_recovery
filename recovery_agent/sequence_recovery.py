# recovery_agent/sequence_recovery.py
import os
import re
import requests
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
    """RCSBのFASTAテキストから 'チェーンID -> アミノ酸配列(1文字)' の対応表を作る。
    RCSBのFASTAヘッダは ">1AON_1|Chains A, B, C, ...|..." のように、同一配列を
    共有する複数チェーンをまとめて記載するため、ヘッダの 'Chains X, Y, Z' 部分を
    パースして該当する全チェーンIDに同じ配列を割り当てる。
    """
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


def map_generated_residues_to_sequence(chain_id, orig_chain_residues, generated_resnums, fasta_sequences):
    """既知残基(1文字)+新規生成残基('X')のテンプレート文字列を作り、FASTA配列と
    グローバルアラインメントすることで、新規生成位置に対応する本来のアミノ酸を推定する。

    orig_chain_residues: {resnum: "GLY"などの3文字コード, ...} (該当チェーンの現在の全残基)
    generated_resnums:   RFdiffusionが新規生成した(=本来の配列が不明な)resnumの集合
    戻り値: {resnum: "ALA"などの正しい3文字コード, ...} (generated_resnumsのみ)
    """
    if chain_id not in fasta_sequences:
        return {}
    target_seq = fasta_sequences[chain_id]

    resnums_sorted = sorted(orig_chain_residues.keys())
    if not resnums_sorted:
        return {}

    template_seq = "".join(
        "X" if resnum in generated_resnums else _THREE_TO_ONE.get(orig_chain_residues[resnum], "X")
        for resnum in resnums_sorted
    )

    aligner = _build_wildcard_aligner()
    alignment = aligner.align(template_seq, target_seq)[0]
    aligned_template, aligned_target = str(alignment[0]), str(alignment[1])

    result = {}
    ti = 0  # template_seq(resnums_sorted)側のインデックス
    for a_char, b_char in zip(aligned_template, aligned_target):
        if a_char != "-":
            if a_char == "X" and b_char not in ("-", "X"):
                resnum = resnums_sorted[ti]
                if resnum in generated_resnums:
                    result[resnum] = _ONE_TO_THREE.get(b_char.upper(), "GLY")
            ti += 1
    return result
