# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
import re
import pickle
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model
from pdbfixer import PDBFixer

from recovery_agent.sequence_recovery import (
    fetch_rcsb_fasta, 
    parse_rcsb_fasta, 
    recover_complex_sequences
)

# GPUメモリの断片化を防ぐためのPyTorch環境変数設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _parse_resnum(res_id):
    """'100A' や '-1' のようなPDB残基IDから整数部分のみを抽出する"""
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


def _get_expected_missing_resnums(fixer):
    """PDBFixerから全チェーンの欠損領域を解析する"""
    pdb_complex_residues = {}
    generated_resnums_dict = {}
    missing_regions = {}

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
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items() if ci == chain.index),
            key=lambda x: x[0],
        )
        
        gen_resnums = set()
        regions = []

        for pos, names in gaps:
            gap_len = len(names)
            if pos == 0:
                start = _parse_resnum(residues[0].id) - gap_len
            else:
                prev_resnum = _parse_resnum(residues[pos - 1].id)
                start = prev_resnum + 1
                
            end = start + gap_len - 1
            regions.append((start, end))
            for resnum in range(start, end + 1):
                gen_resnums.add(resnum)
                
        if gen_resnums:
            generated_resnums_dict[cid] = gen_resnums
            missing_regions[cid] = regions

    return pdb_complex_residues, generated_resnums_dict, missing_regions


def _clean_and_extract_chain_pdb(original_pdb_path, target_chain_id, work_dir):
    """
    指定された鎖を抽出し、座標異常(NaN/inf)やHETATMを除去してクリーンなPDBを出力する。
    【追加】SVDエラー(特異行列)を確実に防ぐため、座標に微小なノイズ(Jittering)を加える。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("orig", original_pdb_path)
    model = structure[0]
    
    if target_chain_id not in model:
        raise ValueError(f"Chain {target_chain_id} not found in PDB.")

    new_struct = Structure.Structure("cleaned")
    new_model = Model.Model(0)
    new_chain = model[target_chain_id].copy()
    
    valid_resnums = []
    
    for res in list(new_chain):
        # 標準アミノ酸以外(水やHETATM)は除外
        if res.id[0] != " ":
            new_chain.detach_child(res.id)
            continue
            
        if "CA" not in res:
            new_chain.detach_child(res.id)
            continue
            
        # 異常座標(NaN, Inf)のチェック
        ca_coord = res["CA"].get_coord()
        if np.any(np.isnan(ca_coord)) or np.any(np.isinf(ca_coord)):
            new_chain.detach_child(res.id)
            continue
            
        # 【ここが追加ポイント】SVDの収束を安定させるための微小ノイズ (0.0001 Å)
        for atom in res:
            coord = atom.get_coord()
            jitter = np.random.normal(0, 1e-4, 3)
            atom.set_coord(coord + jitter)
            
        valid_resnums.append(res.id[1])
        
    if not valid_resnums:
        raise ValueError(f"No valid residues found in chain {target_chain_id}.")

    new_model.add(new_chain)
    new_struct.add(new_model)
    
    out_path = os.path.join(work_dir, f"cleaned_chain_{target_chain_id}.pdb")
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(out_path)
    
    return out_path, sorted(valid_resnums)


def _build_optimized_contig_and_seq(valid_resnums, missing_regions, chain_id):
    """
    実在する座標ブロックと欠損ブロックを順番に並べ、
    RFdiffusionに渡す最適化された contig と provide_seq を生成する。
    """
    tokens = []
    provide_seq_ranges = []
    
    # 既存の残基を連続したセグメントにまとめる
    segments = []
    if valid_resnums:
        start = prev = valid_resnums[0]
        for r in valid_resnums[1:]:
            if r == prev + 1:
                prev = r
            else:
                segments.append((start, prev))
                start = prev = r
        segments.append((start, prev))

    # existing と missing を混ぜて開始インデックスでソート
    all_blocks = []
    for s, e in segments:
        all_blocks.append(('existing', s, e))
    for s, e in sorted(missing_regions, key=lambda x: x[0]):
        all_blocks.append(('missing', s, e))
        
    all_blocks.sort(key=lambda x: x[1])
    
    current_out_idx = 0
    for btype, s, e in all_blocks:
        if btype == 'existing':
            tokens.append(f"{chain_id}{s}-{e}")
            seg_len = e - s + 1
            provide_seq_ranges.append(f"{current_out_idx}-{current_out_idx + seg_len - 1}")
            current_out_idx += seg_len
        elif btype == 'missing':
            gap_len = e - s + 1
            tokens.append(f"{gap_len}-{gap_len}")
            current_out_idx += gap_len
            
    contig = ",".join(tokens)
    provide_seq = ",".join(provide_seq_ranges)
    
    return contig, provide_seq


def _merge_single_chain_to_complex(current_complex_pdb, hal_pdb_path, trb_path, target_cid, regions, corrections, out_path):
    """
    RFdiffusionで生成された単一鎖の新規残基座標を、全体の複合体PDBにマージ（移植）する。
    同時にFASTA由来の正しいアミノ酸名を割り当てる。
    """
    with open(trb_path, "rb") as f:
        trb = pickle.load(f)

    kept_hal_ids = set(trb.get("con_hal_pdb_idx", []))

    parser = PDBParser(QUIET=True)
    complex_struct = parser.get_structure("cpx", current_complex_pdb)
    hal_struct = parser.get_structure("hal", hal_pdb_path)

    # hal出力から「新規生成された残基」のみを抽出
    newly_generated_hal_residues = []
    for chain in hal_struct[0]:
        for res in chain:
            if res.id[0] != " ":
                continue
            is_kept = False
            for kept_id in kept_hal_ids:
                if kept_id[0] == chain.id and kept_id[1] == res.id[1]:
                    is_kept = True
                    break
            if not is_kept:
                newly_generated_hal_residues.append(res)

    # 複合体PDBの挿入先スロットを算出
    expected_missing_slots = []
    for start_res, end_res in sorted(regions, key=lambda x: x[0]):
        for resnum in range(start_res, end_res + 1):
            expected_missing_slots.append(resnum)

    if len(newly_generated_hal_residues) != len(expected_missing_slots):
        raise RuntimeError(
            f"Mismatch in chain {target_cid}: expected {len(expected_missing_slots)} residues, "
            f"but AI generated {len(newly_generated_hal_residues)}."
        )

    # 複合体PDBの対象チェーンへ外科的に移植し、正しい残基名を与える
    target_chain = complex_struct[0][target_cid]
    for resnum, hal_res in zip(expected_missing_slots, newly_generated_hal_residues):
        to_detach = [r.id for r in target_chain if r.id[1] == resnum and r.id[0] == " "]
        for rid in to_detach:
            target_chain.detach_child(rid)
            
        new_res = hal_res.copy()
        new_res.id = (" ", resnum, " ")
        
        # FASTAから取得した正しいアミノ酸名を設定
        correct_resname = corrections.get(target_cid, {}).get(resnum, "GLY")
        new_res.resname = correct_resname
        
        target_chain.add(new_res)
        
    target_chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))

    io = PDBIO()
    io.set_structure(complex_struct)
    io.save(out_path)


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    """
    メイン・エントリポイント。
    """
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()

    pdb_complex_residues, generated_resnums_dict, missing_regions = _get_expected_missing_resnums(fixer)

    if not missing_regions:
        return pdb_path

    complex_corrections = {}
    if pdb_id and rf_config.get("reassign_sequence_from_fasta") and generated_resnums_dict:
        fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=rf_config.get("fasta_cache_dir"))
        fasta_sequences = parse_rcsb_fasta(fasta_text)
        complex_corrections = recover_complex_sequences(
            pdb_complex_residues=pdb_complex_residues,
            generated_resnums_dict=generated_resnums_dict,
            fasta_sequences=fasta_sequences
        )

    current_complex_pdb = pdb_path
    
    for cid, regions in missing_regions.items():
        print(f"[Info] Processing chain {cid} for missing regions: {regions}")
        
        # 1. 複合体から対象鎖のみを抽出し、座標異常を除去 + Jittering適用
        clean_pdb_path, valid_resnums = _clean_and_extract_chain_pdb(current_complex_pdb, cid, work_dir)
        
        # 2. 最適化されたcontigとprovide_seqの作成
        contig, provide_seq = _build_optimized_contig_and_seq(valid_resnums, regions, cid)
        print(f"[Info] Optimized contig for chain {cid}: {contig}")
        
        # 3. 引数の構築
        out_prefix = os.path.join(work_dir, f"rf_out_chain_{cid}")
        cmd = [
            "python", rf_config["script_path"],
            f"inference.output_prefix={out_prefix}",
            f"inference.input_pdb={os.path.abspath(clean_pdb_path)}",
            f"inference.model_directory_path={rf_config['model_directory_path']}",
            f"contigmap.contigs=[{contig}]",
            f"inference.num_designs={rf_config.get('num_designs', 1)}",
        ]
        if provide_seq:
            cmd.append(f"contigmap.provide_seq=[{provide_seq}]")
            
        print(f"[Info] Running RFdiffusion for chain {cid}...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=work_dir, timeout=rf_config.get("timeout_sec", 1800)
        )
        if result.returncode != 0:
            print(f"[Error] RFdiffusion stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"RFdiffusion failed on chain {cid}. Check logs.")
            
        hal_pdb_path = f"{out_prefix}_0.pdb"
        trb_path = f"{out_prefix}_0.trb"
        
        if not os.path.exists(hal_pdb_path) or not os.path.exists(trb_path):
            raise RuntimeError(f"RFdiffusion output files missing for chain {cid}")
        
        # 4. 修復された鎖の一部（新規残基）を、複合体PDBにマージ (正解配列もここで反映)
        next_complex_pdb = os.path.join(work_dir, f"merged_step_chain_{cid}.pdb")
        _merge_single_chain_to_complex(
            current_complex_pdb, hal_pdb_path, trb_path, cid, regions, complex_corrections, next_complex_pdb
        )
        
        current_complex_pdb = next_complex_pdb

    final_out_path = os.path.join(work_dir, "rfdiffusion_final_merged.pdb")
    os.rename(current_complex_pdb, final_out_path)
    
    return final_out_path
