# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
import re
import pickle
from Bio.PDB import PDBParser, PDBIO, Residue
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


def _extract_single_chain(complex_pdb_path, target_cid, out_path):
    """複合体PDBから対象の鎖のみを抽出して保存する"""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("cpx", complex_pdb_path)
    
    for model in struct:
        chains_to_remove = [chain.id for chain in model if chain.id != target_cid]
        for cid in chains_to_remove:
            model.detach_child(cid)
            
    io = PDBIO()
    io.set_structure(struct)
    io.save(out_path)


def _prepare_single_chain_input(chain_pdb_path, cid, corrections, regions, out_path):
    """単一鎖の欠損部分に、正しいアミノ酸名を持つダミー残基を挿入する"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("chain", chain_pdb_path)
    model = structure[0]
    
    if cid in model:
        chain = model[cid]
        residues = sorted([res for res in chain if res.id[0] == " "], key=lambda r: r.id[1])
        
        for start_res, end_res in regions:
            for resnum in range(start_res, end_res + 1):
                resname = corrections.get(cid, {}).get(resnum, "GLY")
                new_res = Residue.Residue((" ", resnum, " "), resname, " ")
                
                # 直前の残基からCA座標をコピー（ダミー用）
                if residues:
                    prev_res = residues[-1]
                    if "CA" in prev_res:
                        new_res.add(prev_res["CA"].copy())
                    else:
                        for atom in prev_res:
                            new_res.add(atom.copy())
                chain.add(new_res)
                residues.append(new_res)
                
        chain.child_list.sort(key=lambda r: r.id[1])
        
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def _build_args_for_single_chain(prepared_pdb_path, cid, regions, corrections):
    """単一鎖に対するRFdiffusion引数（contig, inpaint_str, provide_seq）を生成"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prep", prepared_pdb_path)
    chain = structure[0][cid]
    
    residues = [res for res in chain if res.id[0] == " "]
    if not residues:
        return "", "", ""
        
    start_num = residues[0].id[1]
    end_num = residues[-1].id[1]
    
    # 単一鎖なので結合タグ (/0 ) は不要でシンプル
    contig = f"{cid}{start_num}-{end_num}"
    
    inpaint_str_list = [f"{cid}{start}-{end}" for start, end in regions]
    inpaint_str = ",".join(inpaint_str_list)
    
    provide_seq_indices = []
    current_idx = 0
    for res in residues:
        resnum = res.id[1]
        is_missing = any(start <= resnum <= end for start, end in regions)
        if is_missing and cid in corrections and resnum in corrections[cid]:
            provide_seq_indices.append(current_idx)
        current_idx += 1
        
    provide_seq_ranges = []
    if provide_seq_indices:
        start = prev = provide_seq_indices[0]
        for idx in provide_seq_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                provide_seq_ranges.append(f"{start}-{prev}")
                start = prev = idx
        provide_seq_ranges.append(f"{start}-{prev}")
        
    provide_seq = ",".join(provide_seq_ranges)
    
    return contig, inpaint_str, provide_seq


def _merge_single_chain_to_complex(current_complex_pdb, hal_pdb_path, trb_path, target_cid, regions, out_path):
    """
    RFdiffusionで生成された単一鎖の新規残基座標を、全体の複合体PDBにマージ（移植）する。
    """
    with open(trb_path, "rb") as f:
        trb = pickle.load(f)

    # .trb から保持された残基を取得
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
            if (chain.id, res.id[1]) not in kept_hal_ids:
                newly_generated_hal_residues.append(res)

    # 複合体PDBの挿入先スロットを算出
    expected_missing_slots = []
    for start_res, end_res in regions:
        for resnum in range(start_res, end_res + 1):
            expected_missing_slots.append(resnum)

    if len(newly_generated_hal_residues) != len(expected_missing_slots):
        raise RuntimeError(
            f"Mismatch in chain {target_cid}: expected {len(expected_missing_slots)} residues, "
            f"but AI generated {len(newly_generated_hal_residues)}."
        )

    # 複合体PDBの対象チェーンへ外科的に移植
    target_chain = complex_struct[0][target_cid]
    for resnum, hal_res in zip(expected_missing_slots, newly_generated_hal_residues):
        to_detach = [r.id for r in target_chain if r.id[1] == resnum and r.id[0] == " "]
        for rid in to_detach:
            target_chain.detach_child(rid)
            
        new_res = hal_res.copy()
        new_res.id = (" ", resnum, " ")
        target_chain.add(new_res)
        
    target_chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))

    io = PDBIO()
    io.set_structure(complex_struct)
    io.save(out_path)


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    """
    メイン・エントリポイント。
    複合体全体から欠損を検知し、欠損のある「鎖ごと」に分割してRFdiffusionを実行・マージする。
    """
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()

    pdb_complex_residues, generated_resnums_dict, missing_regions = _get_expected_missing_resnums(fixer)

    # 欠損がない場合はそのまま返す
    if not missing_regions:
        return pdb_path

    # 全体の正解配列マッピングを取得
    complex_corrections = {}
    if pdb_id and rf_config.get("reassign_sequence_from_fasta") and generated_resnums_dict:
        fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=rf_config.get("fasta_cache_dir"))
        fasta_sequences = parse_rcsb_fasta(fasta_text)
        complex_corrections = recover_complex_sequences(
            pdb_complex_residues=pdb_complex_residues,
            generated_resnums_dict=generated_resnums_dict,
            fasta_sequences=fasta_sequences
        )

    # 逐次マージ用のトラッキング変数
    current_complex_pdb = pdb_path
    
    # 欠損のある鎖ごとにループ処理
    for cid, regions in missing_regions.items():
        print(f"[Info] Processing chain {cid} for missing regions: {regions}")
        
        # 1. 複合体から対象鎖のみを抽出
        chain_pdb_path = os.path.join(work_dir, f"temp_chain_{cid}.pdb")
        _extract_single_chain(current_complex_pdb, cid, chain_pdb_path)
        
        # 2. 欠損部に正しい配列のダミー残基を挿入
        prep_chain_pdb = os.path.join(work_dir, f"prep_chain_{cid}.pdb")
        _prepare_single_chain_input(chain_pdb_path, cid, complex_corrections, regions, prep_chain_pdb)
        
        # 3. 引数の構築
        contig, inpaint_str, provide_seq = _build_args_for_single_chain(
            prep_chain_pdb, cid, regions, complex_corrections
        )
        
        out_prefix = os.path.join(work_dir, f"rf_out_chain_{cid}")
        cmd = [
            "python", rf_config["script_path"],
            f"inference.output_prefix={out_prefix}",
            f"inference.input_pdb={os.path.abspath(prep_chain_pdb)}",
            f"inference.model_directory_path={rf_config['model_directory_path']}",
            f"contigmap.contigs=[{contig}]",
            f"inference.num_designs={rf_config.get('num_designs', 1)}",
        ]
        if inpaint_str:
            cmd.append(f"contigmap.inpaint_str=[{inpaint_str}]")
        if provide_seq:
            cmd.append(f"contigmap.provide_seq=[{provide_seq}]")
            
        print(f"[Info] Running RFdiffusion for chain {cid}...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=work_dir, timeout=rf_config.get("timeout_sec", 1800)
        )
        if result.returncode != 0:
            raise RuntimeError(f"RFdiffusion failed on chain {cid}: {result.stderr[-2000:]}")
            
        hal_pdb_path = f"{out_prefix}_0.pdb"
        trb_path = f"{out_prefix}_0.trb"
        
        # 4. 修復された鎖の一部（新規残基）を、複合体PDBにマージ
        next_complex_pdb = os.path.join(work_dir, f"merged_step_chain_{cid}.pdb")
        _merge_single_chain_to_complex(
            current_complex_pdb, hal_pdb_path, trb_path, cid, regions, next_complex_pdb
        )
        
        # 次のループへマージ済みPDBを引き継ぐ
        current_complex_pdb = next_complex_pdb

    final_out_path = os.path.join(work_dir, "rfdiffusion_final_merged.pdb")
    os.rename(current_complex_pdb, final_out_path)
    
    return final_out_path
