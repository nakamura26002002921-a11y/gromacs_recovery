# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
import re
from Bio.PDB import PDBParser, PDBIO, Residue
from pdbfixer import PDBFixer

from recovery_agent.sequence_recovery import (
    fetch_rcsb_fasta, 
    parse_rcsb_fasta, 
    recover_complex_sequences, 
    _THREE_TO_ONE
)


def _parse_resnum(res_id):
    """'100A' や '-1' のようなPDB残基IDから整数部分のみを抽出する"""
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


def _get_expected_missing_resnums(fixer):
    """PDBFixerから正確な欠損数と位置を取得する"""
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


def _prepare_input_pdb_with_correct_sequence(pdb_path, corrections, missing_regions, work_dir):
    """欠損部分に正しいアミノ酸名を持つダミー残基（CAのみ）を挿入した完全なPDBを生成する"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("orig", pdb_path)
    model = structure[0]
    
    for cid, regions in missing_regions.items():
        if cid not in model or cid not in corrections:
            continue
        chain = model[cid]
        residues = sorted([res for res in chain if res.id[0] == " "], key=lambda r: r.id[1])
        
        for start_res, end_res in regions:
            for resnum in range(start_res, end_res + 1):
                resname = corrections[cid].get(resnum, "GLY")
                new_res = Residue.Residue((" ", resnum, " "), resname, " ")
                
                # ダミー座標として既存の残基から座標をコピー（後でRFdiffusionが再生成するためダミーでよい）
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
        
    out_path = os.path.join(work_dir, "rfdiffusion_input_seq_corrected.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    return out_path


def _build_contig_from_prepared_pdb(pdb_path):
    """ダミー補完済みPDBから全チェーンをマッピングするcontigを生成。

    RFdiffusionのcontig記法では、
      - "/" は同一出力チェーン内でセグメントを連結する区切り（例: A2-100/10-10/A111-200 のような1本のchain内のギャップ表現）
      - "," は独立した別々の出力チェーンを区切る記法
    である。マルチチェーンの複合体（A, B, C, ... の各チェーンをそれぞれ
    別チェーンとしてRFdiffusionに渡したい場合）は "," で連結しなければならない。
    誤って "/0/" で連結すると、全チェーンが1本のchainとして扱われてしまい、
    "Multiple chain IDs in chain" のようなアサーションエラーになる。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prep", pdb_path)
    chain_groups = []
    
    for chain in structure[0]:
        residues = [res for res in chain if res.id[0] == " "]
        if not residues:
            continue
        cid = chain.id
        start_num = residues[0].id[1]
        end_num = residues[-1].id[1]
        chain_groups.append(f"{cid}{start_num}-{end_num}")
        
    # 【重要】RFdiffusionで複数チェーンを独立したchainとして扱わせる正しい記法はカンマ区切り
    return ",".join(chain_groups)


def _get_provide_seq_ranges(pdb_path, missing_regions):
    """provide_seqに渡すための「生成構造全体の0始まりインデックス」の範囲を算出する"""
    provide_seq_indices = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prep", pdb_path)
    
    current_idx = 0
    for chain in structure[0]:
        cid = chain.id
        for res in chain:
            if res.id[0] != " ":
                continue
            resnum = res.id[1]
            
            # この残基が欠損（再生成対象）かチェック
            is_missing = False
            if cid in missing_regions:
                for start_res, end_res in missing_regions[cid]:
                    if start_res <= resnum <= end_res:
                        is_missing = True
                        break
                        
            if is_missing:
                provide_seq_indices.append(current_idx)
            current_idx += 1
            
    # 連続するインデックスを "9-13" のような範囲文字列に圧縮
    ranges = []
    if provide_seq_indices:
        start = provide_seq_indices[0]
        prev = provide_seq_indices[0]
        for idx in provide_seq_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                ranges.append(f"{start}-{prev}")
                start = idx
                prev = idx
        ranges.append(f"{start}-{prev}")
        
    return ranges


def _merge_designed_region(original_pdb_path, hal_pdb_path, missing_regions, work_dir):
    """
    全マッピングで出力されたRFdiffusionの結果(hal)から、
    missing_regionsに該当する残基の座標だけを元の構造に差し替える。
    """
    parser = PDBParser(QUIET=True)
    orig_structure = parser.get_structure("orig", original_pdb_path)
    hal_structure = parser.get_structure("hal", hal_pdb_path)
    orig_model = orig_structure[0]
    hal_model = hal_structure[0]

    for cid, regions in missing_regions.items():
        if cid not in orig_model or cid not in hal_model:
            continue
        orig_chain = orig_model[cid]
        hal_chain = hal_model[cid]

        for start_res, end_res in regions:
            for resnum in range(start_res, end_res + 1):
                hal_res_list = [r for r in hal_chain if r.id[1] == resnum and r.id[0] == " "]
                if not hal_res_list:
                    continue
                hal_res = hal_res_list[0].copy()

                to_detach = [r.id for r in orig_chain if r.id[1] == resnum]
                for rid in to_detach:
                    orig_chain.detach_child(rid)

                orig_chain.add(hal_res)

        orig_chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))

    out_path = os.path.join(work_dir, "rfdiffusion_merged.pdb")
    io = PDBIO()
    io.set_structure(orig_structure)
    io.save(out_path)
    return out_path


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()

    pdb_complex_residues, generated_resnums_dict, missing_regions = _get_expected_missing_resnums(fixer)

    if pdb_id and rf_config.get("reassign_sequence_from_fasta") and generated_resnums_dict:
        fasta_text = fetch_rcsb_fasta(pdb_id, cache_dir=rf_config.get("fasta_cache_dir"))
        fasta_sequences = parse_rcsb_fasta(fasta_text)

        complex_corrections = recover_complex_sequences(
            pdb_complex_residues=pdb_complex_residues,
            generated_resnums_dict=generated_resnums_dict,
            fasta_sequences=fasta_sequences
        )
    else:
        complex_corrections = {}

    # 1. 欠損部に正しい配列を持つダミー残基を挿入（入力配列の固定）
    input_pdb_for_rfdiffusion = _prepare_input_pdb_with_correct_sequence(
        pdb_path, complex_corrections, missing_regions, work_dir
    )

    # 2. ダミー残基ごと全チェーンをマッピングする contig を構築（例: A1-100,B1-50）
    contig = _build_contig_from_prepared_pdb(input_pdb_for_rfdiffusion)
    
    # 3. ダミー残基部分の「座標」を再生成させるための inpaint_str の構築
    inpaint_str_list = []
    for cid, regions in missing_regions.items():
        if cid not in complex_corrections:
            continue
        for start_res, end_res in regions:
            inpaint_str_list.append(f"{cid}{start_res}-{end_res}")

    # 4. ダミー残基部分の「配列（正しいアミノ酸）」を維持するための provide_seq (0始まりインデックス)
    provide_seq_ranges = _get_provide_seq_ranges(input_pdb_for_rfdiffusion, missing_regions)

    out_prefix = os.path.join(work_dir, "rfdiffusion_out")
    cmd = [
        "python", rf_config["script_path"],
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={os.path.abspath(input_pdb_for_rfdiffusion)}",
        f"inference.model_directory_path={rf_config['model_directory_path']}",
        f"contigmap.contigs=[{contig}]",
        f"inference.num_designs={rf_config.get('num_designs', 1)}",
    ]

    if inpaint_str_list:
        cmd.append(f"contigmap.inpaint_str=[{','.join(inpaint_str_list)}]")
    if provide_seq_ranges:
        cmd.append(f"contigmap.provide_seq=[{','.join(provide_seq_ranges)}]")

    print(f"[Info] Running RFdiffusion with command:\n{' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=work_dir, timeout=rf_config.get("timeout_sec", 1800),
    )
    if result.returncode != 0:
        raise RuntimeError(f"RFdiffusion failed: {result.stderr[-2000:]}")

    hal_pdb_path = f"{out_prefix}_0.pdb"
    if not os.path.exists(hal_pdb_path):
        raise RuntimeError(f"RFdiffusion output not found: {hal_pdb_path}")

    # .trb ファイルを使わず、missing_regions に基づいて直接座標をマージ
    merged_path = _merge_designed_region(pdb_path, hal_pdb_path, missing_regions, work_dir)

    return merged_path
