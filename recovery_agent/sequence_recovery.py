# recovery_agent/sequence_recovery.py
"""
rfdiffusion_final_merged.pdb (GLYまみれ) に正しいアミノ酸名を割り当てる。

アルゴリズム:
  1. 元PDBからPDBFixerで欠損resnumを検出 (rfdiffusion_repair.pyと同じロジック)
  2. 元PDBの実在残基(先頭〜15残基)をアンカーにFASTA内でstr.find()
  3. オフセットを確定し、欠損resnum → FASTA[offset + i] を直接参照
  4. 同一パターンの鎖(ホモマー)は1回だけ計算して全鎖に適用

アラインメントライブラリ・ハンガリー法・Biopython PairwiseAligner は一切不使用。
"""
import os
import re
import requests
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
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
    return resp.text


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
# コア: オフセット決定 + 直接参照
# ============================================================================

def _determine_offset(present_residues_sorted, fasta_seq):
    """
    実在残基の先頭アンカーをFASTA内でstr.find()し、
    PDB resnum → FASTA position のオフセットを決定する。

    :param present_residues_sorted: [(resnum, one_letter_aa), ...] resnum昇順
    :param fasta_seq: FASTA配列 (1文字)
    :return: offset (int) or None
             offset の意味: FASTA[resnum + offset] がその残基の正しいアミノ酸
    """
    if not present_residues_sorted or not fasta_seq:
        return None

    # アンカー: 先頭15残基の1文字配列
    anchor_len = min(15, len(present_residues_sorted))
    anchor = "".join(aa for _, aa in present_residues_sorted[:anchor_len])

    # 完全マッチ
    pos = fasta_seq.find(anchor)
    if pos >= 0:
        offset = pos - present_residues_sorted[0][0]
        return offset

    # 部分マッチ (10残基)
    if anchor_len > 10:
        pos = fasta_seq.find(anchor[:10])
        if pos >= 0:
            offset = pos - present_residues_sorted[0][0]
            return offset

    # 部分マッチ (7残基)
    if anchor_len > 7:
        pos = fasta_seq.find(anchor[:7])
        if pos >= 0:
            offset = pos - present_residues_sorted[0][0]
            return offset

    return None


def _verify_offset(present_residues_sorted, fasta_seq, offset):
    """オフセットの正しさを全実在残基で検証。一致率を返す。"""
    match = 0
    total = 0
    for resnum, aa in present_residues_sorted:
        fp = resnum + offset
        if 0 <= fp < len(fasta_seq):
            total += 1
            if aa == fasta_seq[fp]:
                match += 1
    return match / max(total, 1)


def _find_fasta_for_chain(present_residues_sorted, fasta_sequences):
    """
    全FASTA鎖の中から、実在残基と最も一致する鎖とオフセットを見つける。
    """
    best_cid = None
    best_offset = None
    best_identity = 0.0

    for f_cid, f_seq in fasta_sequences.items():
        offset = _determine_offset(present_residues_sorted, f_seq)
        if offset is None:
            continue
        identity = _verify_offset(present_residues_sorted, f_seq, offset)
        if identity > best_identity:
            best_identity = identity
            best_cid = f_cid
            best_offset = offset

    if best_identity < 0.80:
        return None, None
    return best_cid, best_offset


# ============================================================================
# メイン
# ============================================================================

def apply_sequence_recovery(original_pdb_path, rfdiffusion_pdb_path, work_dir, pdb_id,
                            out_name="sequence_recovered.pdb", cache_dir=None):
    """
    rfdiffusion_final_merged.pdb のGLY残基を正しいアミノ酸名に置換する。

    :param original_pdb_path: RFdiffusion実行前の元PDB (欠損あり)
    :param rfdiffusion_pdb_path: rfdiffusion_final_merged.pdb (GLY埋め済み)
    :param work_dir: 出力先ディレクトリ
    :param pdb_id: RCSB PDB ID
    :param out_name: 出力ファイル名
    :param cache_dir: FASTAキャッシュディレクトリ
    """
    # --- 1. 元PDBから欠損情報を取得 (rfdiffusion_repair.pyと同じロジック) ---
    fixer = PDBFixer(filename=original_pdb_path)
    fixer.findMissingResidues()

    # 鎖ごとに「実在残基」と「欠損resnum」を収集
    chain_data = {}  # {cid: {"present": [(resnum, aa), ...], "missing": {resnum, ...}}}

    for chain in fixer.topology.chains():
        cid = chain.id
        residues = list(chain.residues())
        if not residues:
            continue

        present = []
        for res in residues:
            rn = _parse_resnum(res.id)
            aa = _THREE_TO_ONE.get(res.name.upper(), "X")
            present.append((rn, aa))
        present.sort(key=lambda x: x[0])

        # 欠損resnumの計算
        gaps = sorted(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items()
             if ci == chain.index),
            key=lambda x: x[0],
        )
        missing = set()
        for pos, names in gaps:
            gap_len = len(names)
            if pos == 0:
                start = _parse_resnum(residues[0].id) - gap_len
            else:
                prev_rn = _parse_resnum(residues[pos - 1].id)
                start = prev_rn + 1
            for rn in range(start, start + gap_len):
                missing.add(rn)

        if missing:
            chain_data[cid] = {"present": present, "missing": missing}

    if not chain_data or not pdb_id:
        return rfdiffusion_pdb_path

    # --- 2. FASTA取得 ---
    fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=cache_dir)
    fasta_sequences = parse_rcsb_fasta(fasta_text)
    if not fasta_sequences:
        return rfdiffusion_pdb_path

    # --- 3. ホモマー重複排除: 同一パターンの鎖は1回だけ計算 ---
    # キー: (実在残基の1文字配列, 欠損resnumのfrozenset)
    pattern_cache = {}  # pattern_key → {resnum: resname}
    corrections = {}    # {cid: {resnum: resname}}

    for cid, data in chain_data.items():
        present = data["present"]
        missing = data["missing"]

        # パターンキー: 実在配列 + 欠損位置
        present_seq = "".join(aa for _, aa in present)
        pattern_key = (present_seq, frozenset(missing))

        if pattern_key in pattern_cache:
            # 同一パターン → キャッシュから再利用
            corrections[cid] = dict(pattern_cache[pattern_key])
            continue

        # FASTA鎖とオフセットを決定
        fasta_cid, offset = _find_fasta_for_chain(present, fasta_sequences)
        if fasta_cid is None:
            print(f"  [Warning] Chain {cid}: no suitable FASTA match found, skipping.")
            continue

        f_seq = fasta_sequences[fasta_cid]

        # 欠損resnum → FASTA参照で正しいアミノ酸を決定
        mapping = {}
        for resnum in sorted(missing):
            fp = resnum + offset
            if 0 <= fp < len(f_seq):
                aa = f_seq[fp]
                if aa.isalpha():
                    mapping[resnum] = _ONE_TO_THREE.get(aa, "GLY")

        corrections[cid] = mapping
        pattern_cache[pattern_key] = mapping

        print(f"  [Info] Chain {cid}: matched FASTA chain {fasta_cid} "
              f"(offset={offset}, recovered {len(mapping)}/{len(missing)} residues)")

    if not corrections:
        return rfdiffusion_pdb_path

    # --- 4. rfdiffusion_final_merged.pdb のGLYを置換 ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("seqrec", rfdiffusion_pdb_path)
    model = structure[0]

    for cid, resnum_to_resname in corrections.items():
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


# 後方互換ラッパー
def map_generated_residues_to_sequence(chain_id, orig_chain_residues, generated_resnums,
                                       fasta_sequences):
    present = sorted(
        [(rn, _THREE_TO_ONE.get(str(name).upper(), "X"))
         for rn, name in orig_chain_residues.items()],
        key=lambda x: x[0]
    )
    fasta_cid, offset = _find_fasta_for_chain(present, fasta_sequences)
    if fasta_cid is None:
        return {}
    f_seq = fasta_sequences[fasta_cid]
    mapping = {}
    for resnum in generated_resnums:
        fp = resnum + offset
        if 0 <= fp < len(f_seq):
            mapping[resnum] = _ONE_TO_THREE.get(f_seq[fp], "GLY")
    return mapping

def _get_generated_resnums_from_original(original_pdb_path):
    """
    元PDBからPDBFixerで欠損を検出する。
    テストスクリプト(test_real_data.py)の診断表示用。
    apply_sequence_recovery内部でも同じロジックを使用。
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
                prev_rn = _parse_resnum(residues[pos - 1].id)
                start = prev_rn + 1
            for rn in range(start, start + gap_len):
                gen_resnums.add(rn)
        if gen_resnums:
            generated_resnums_dict[cid] = gen_resnums

    return pdb_complex_residues, generated_resnums_dict
