# recovery_agent/rfdiffusion_repair.py
import os
import subprocess
import re
from Bio.PDB import PDBParser, PDBIO, Residue
from Bio.PDB.NeighborSearch import NeighborSearch
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


def _get_anchor_atoms(model, missing_regions, padding=10):
    """
    各欠損チェーンについて、欠損区間の直前・直後にある「実在する」残基
    （アンカー残基）の原子を集める。空間的近傍探索の基準点として使う。
    padding: アンカーとして採用する、欠損区間の前後何残基分を見るか。
    """
    anchor_atoms = []
    for chain in model:
        cid = chain.id
        if cid not in missing_regions:
            continue
        residues_sorted = sorted(
            (res for res in chain if res.id[0] == " "), key=lambda r: r.id[1]
        )
        resnum_to_res = {res.id[1]: res for res in residues_sorted}
        resnums_sorted = [res.id[1] for res in residues_sorted]

        for start_res, end_res in missing_regions[cid]:
            # 欠損区間の直前 padding 残基
            before = [r for r in resnums_sorted if r < start_res][-padding:]
            # 欠損区間の直後 padding 残基
            after = [r for r in resnums_sorted if r > end_res][:padding]
            for resnum in before + after:
                res = resnum_to_res.get(resnum)
                if res is not None:
                    anchor_atoms.extend(res.get_atoms())
    return anchor_atoms


def _select_context_resnums(model, missing_regions, anchor_atoms, radius=15.0):
    """
    アンカー原子から radius [Å] 以内に原子を持つ残基を「保持すべき近傍残基」として
    チェーンごとに resnum の集合で返す。欠損チェーン自身の全残基は無条件で含める
    （欠損チェーンはそのままRFdiffusionのmotif/生成対象として渡すため）。
    """
    ns = NeighborSearch(anchor_atoms)
    keep_resnums = {}  # cid -> set(resnum)

    for chain in model:
        cid = chain.id
        keep_resnums.setdefault(cid, set())
        if cid in missing_regions:
            # 欠損チェーンは丸ごと保持（ダミー残基込みで後段のcontigに使うため）
            for res in chain:
                if res.id[0] == " ":
                    keep_resnums[cid].add(res.id[1])
            continue

        for res in chain:
            if res.id[0] != " ":
                continue
            for atom in res:
                nearby = ns.search(atom.coord, radius)
                if nearby:
                    keep_resnums[cid].add(res.id[1])
                    break

    return keep_resnums


def _resnums_to_contig_ranges(resnums_sorted):
    """昇順resnumのリストを、連続区間の [(start, end), ...] に圧縮する。"""
    if not resnums_sorted:
        return []
    ranges = []
    start = prev = resnums_sorted[0]
    for n in resnums_sorted[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append((start, prev))
            start = prev = n
    ranges.append((start, prev))
    return ranges


def _build_truncated_pdb(pdb_path, missing_regions, work_dir, padding=10, radius=15.0):
    """
    欠損チェーンはそのまま保持しつつ、欠損のないチェーンについては
    欠損部アンカー残基から radius [Å] 以内にある残基だけを残した
    「トランケーション済みPDB」を作る。RFdiffusionへの入力サイズ
    （＝GPUメモリ使用量、O(N^2)でスケールする）を大幅に削減するための処理。

    戻り値:
        truncated_pdb_path: トランケーション後のPDBファイルパス
        context_ranges: {chain_id: [(start, end), ...]} 各チェーンで
            保持された連続区間のリスト（contig構築・fixed領域指定に使う）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("full", pdb_path)
    model = structure[0]

    anchor_atoms = _get_anchor_atoms(model, missing_regions, padding=padding)
    keep_resnums = _select_context_resnums(model, missing_regions, anchor_atoms, radius=radius)

    context_ranges = {}
    for chain in list(model):
        cid = chain.id
        keep = keep_resnums.get(cid, set())
        # 保持対象外の残基を削除
        to_remove = [res.id for res in chain if res.id[0] == " " and res.id[1] not in keep]
        for rid in to_remove:
            chain.detach_child(rid)

        remaining_resnums = sorted(res.id[1] for res in chain if res.id[0] == " ")
        if cid not in missing_regions:
            # コンテキスト鎖の場合のみ、後でcontigに使う連続区間を記録
            context_ranges[cid] = _resnums_to_contig_ranges(remaining_resnums)

        if not remaining_resnums:
            model.detach_child(chain.id)

    out_path = os.path.join(work_dir, "rfdiffusion_input_truncated.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    return out_path, context_ranges


def _build_contig(truncated_pdb_path, missing_regions, context_ranges):
    """トランケーション済みPDBから全チェーンをマッピングするcontigを生成。

    RFdiffusionのcontig記法（公式ドキュメント/実例に基づく）:
      - "/" はセグメントの連結。同一チェーン内で複数の断片（歯抜け区間）を
        つなぐ場合や、直後に "0" を置いて "chain break" を表す場合に使う。
      - 複数の入力チェーンをそれぞれ独立した出力チェーンとして扱わせたい場合は、
        各チェーンのブロックの間に "/0 "（スラッシュ0 + 半角スペース）を挟む。
        例: "A2-525/0 B2-525/0 C2-525" のように、チェーンの区切りは
        カンマではなく "/0 "(chain-break + スペース) である。
      - カンマ(,)はcontigsのチェーン区切りには使わない
        （provide_seq / inpaint_str など、範囲のリストを列挙する引数の区切りとして使う）。

    トランケーション（空間的近傍抽出）によりコンテキスト鎖が歯抜けになる
    ことがあるため、各チェーンを単一の start-end ではなく、
    context_ranges（コンテキスト鎖）/ missing_regions（欠損チェーン、
    ダミー残基を挟んだ連続範囲）に基づく複数区間として組み立てる。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prep", truncated_pdb_path)
    model = structure[0]

    chain_groups = []
    for chain in model:
        cid = chain.id
        residues = [res for res in chain if res.id[0] == " "]
        if not residues:
            continue

        if cid in missing_regions:
            # 欠損チェーン: ダミー残基込みで実在する残基は連続しているはずなので
            # 単純に最初-最後の範囲でよい（トランケーションで削っていない）
            start_num = residues[0].id[1]
            end_num = residues[-1].id[1]
            chain_groups.append(f"{cid}{start_num}-{end_num}")
        else:
            # コンテキスト鎖: トランケーションで歯抜けになった区間を
            # "/" でつないだ1つのcontigブロックとして表現する
            ranges = context_ranges.get(cid) or []
            if not ranges:
                continue
            segment = "/".join(f"{cid}{s}-{e}" for s, e in ranges)
            chain_groups.append(segment)

    # 【重要】RFdiffusionでチェーンを独立させる正しい記法は "/0 "(スラッシュ0+スペース)
    return "/0 ".join(chain_groups)


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


def _merge_designed_region(input_pdb_for_rfdiffusion, hal_pdb_path, trb_path, missing_regions, work_dir):
    """
    RFdiffusionの出力(hal)から、新規生成された残基の座標だけを
    ダミー補完済み入力PDB(input_pdb_for_rfdiffusion)へマージする。

    【重要】RFdiffusionは、入力PDBに複数チェーンがあっても出力PDBで
    チェーンを勝手に統合・再編成することがある(既知の挙動。例:
    https://github.com/RosettaCommons/RFdiffusion/issues/315 )。
    そのため出力側の "chain id" は当てにできず、chain idでの突き合わせは
    静かに失敗して(該当チェーンがhal_model側に無いとみなされ)マージが
    スキップされ、欠損残基がそのまま残ってしまう不具合の原因になっていた。

    代わりに .trb ファイルの con_ref_pdb_idx / con_hal_pdb_idx
    (= 「元のPDBのどの(chain,resnum)が、出力PDBのどの(chain,resnum)に
    対応するか」という、motif=保持された残基についての対応表)を使う。
    出力PDBの残基のうち、この対応表に載っていないものが
    「新規生成された残基」であり、これを入力側の欠損位置の並び順と
    1対1で対応させてマージする。
    """
    import pickle

    with open(trb_path, "rb") as f:
        trb = pickle.load(f)

    # con_hal_pdb_idx: 出力PDB上で「保持された(=新規生成でない)」残基の (chain, resnum) 集合
    kept_hal_ids = set(trb.get("con_hal_pdb_idx", []))

    parser = PDBParser(QUIET=True)
    input_structure = parser.get_structure("input", input_pdb_for_rfdiffusion)
    hal_structure = parser.get_structure("hal", hal_pdb_path)
    input_model = input_structure[0]
    hal_model = hal_structure[0]

    # 出力PDBの全残基を、PDB内の出現順（chainの並び順→resnumの並び順）でリスト化
    hal_residues_in_order = []
    for chain in hal_model:
        for res in chain:
            if res.id[0] != " ":
                continue
            hal_residues_in_order.append((chain.id, res.id[1], res))

    # 「新規生成された」出力残基だけを出現順に抽出
    newly_generated_hal_residues = [
        res for (cid, resnum, res) in hal_residues_in_order
        if (cid, resnum) not in kept_hal_ids
    ]

    # 入力側（ダミー補完済みPDB）の欠損位置も、chainの並び順→resnumの昇順で同じ順序に並べる
    # ※ input_pdb_for_rfdiffusion の chain 順序が RFdiffusion に渡した contig の順序と
    #   一致している前提（_prepare_input_pdb_with_correct_sequenceで生成した構造そのまま）
    expected_missing_slots = []
    for chain in input_model:
        cid = chain.id
        if cid not in missing_regions:
            continue
        for start_res, end_res in missing_regions[cid]:
            for resnum in range(start_res, end_res + 1):
                expected_missing_slots.append((cid, resnum))

    if len(newly_generated_hal_residues) != len(expected_missing_slots):
        raise RuntimeError(
            "RFdiffusion output/trb residue count mismatch: "
            f"expected {len(expected_missing_slots)} newly generated residues "
            f"(from missing_regions), but found {len(newly_generated_hal_residues)} "
            "in the output PDB that are not marked as 'kept' in the .trb file. "
            "This usually means the contig string did not match the input PDB "
            "structure, or RFdiffusion merged/reordered chains unexpectedly."
        )

    # 1対1で対応付けて、入力構造の欠損残基をhalの新規生成残基座標で置き換える
    merged_structure = input_structure
    merged_model = merged_structure[0]
    for (cid, resnum), hal_res in zip(expected_missing_slots, newly_generated_hal_residues):
        chain = merged_model[cid]
        to_detach = [r.id for r in chain if r.id[1] == resnum and r.id[0] == " "]
        for rid in to_detach:
            chain.detach_child(rid)
        new_res = hal_res.copy()
        new_res.id = (" ", resnum, " ")
        chain.add(new_res)
        chain.child_list.sort(key=lambda r: (r.id[1], r.id[2]))

    out_path = os.path.join(work_dir, "rfdiffusion_merged.pdb")
    io = PDBIO()
    io.set_structure(merged_structure)
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

    # 1. 欠損部に正しい配列を持つダミー残基を挿入（入力配列の固定）。
    #    このフルサイズPDBは、後段の座標マージの「差し戻し先」として使う。
    full_pdb_with_dummy = _prepare_input_pdb_with_correct_sequence(
        pdb_path, complex_corrections, missing_regions, work_dir
    )

    # 2. 空間的トランケーション: 欠損チェーンはそのまま保持し、欠損のない
    #    コンテキスト鎖は欠損部アンカー残基の近傍だけを残す。
    #    RFdiffusionはO(N^2)でGPUメモリを消費するため、数千残基級の複合体を
    #    まるごと渡すとCUDA OOMになる。近傍だけに絞ることで計算量を削減しつつ、
    #    構造予測に必要な周辺の相互作用情報は維持する。
    truncation_cfg = rf_config.get("truncation", {})
    anchor_padding = truncation_cfg.get("anchor_padding_residues", 10)
    context_radius = truncation_cfg.get("context_radius_angstrom", 15.0)

    truncated_pdb, context_ranges = _build_truncated_pdb(
        full_pdb_with_dummy, missing_regions, work_dir,
        padding=anchor_padding, radius=context_radius,
    )

    # 3. トランケーション後のPDBに基づいて contig を構築
    contig = _build_contig(truncated_pdb, missing_regions, context_ranges)

    # 4. ダミー残基部分の「座標」を再生成させるための inpaint_str の構築
    inpaint_str_list = []
    for cid, regions in missing_regions.items():
        if cid not in complex_corrections:
            continue
        for start_res, end_res in regions:
            inpaint_str_list.append(f"{cid}{start_res}-{end_res}")

    # 5. ダミー残基部分の「配列（正しいアミノ酸）」を維持するための provide_seq
    #    (トランケーション後PDB全体を通した0始まりインデックス)
    provide_seq_ranges = _get_provide_seq_ranges(truncated_pdb, missing_regions)

    out_prefix = os.path.join(work_dir, "rfdiffusion_out")
    cmd = [
        "python", rf_config["script_path"],
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={os.path.abspath(truncated_pdb)}",
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
    trb_path = f"{out_prefix}_0.trb"
    if not os.path.exists(hal_pdb_path):
        raise RuntimeError(f"RFdiffusion output not found: {hal_pdb_path}")
    if not os.path.exists(trb_path):
        raise RuntimeError(f"RFdiffusion .trb metadata not found: {trb_path}")

    # .trb ファイルの con_hal_pdb_idx を使い、「新規生成された残基」を
    # チェーンIDに依存せず特定してマージする（RFdiffusionは出力側で
    # チェーンを統合・再編成することがあるため、chain idでの突き合わせは不可）。
    # マージ先はトランケーション前のフルサイズPDB
    # （full_pdb_with_dummy）にすることで、削っていたコンテキスト鎖の
    # 残りの部分（近傍外の残基）を最終出力に残す。
    merged_path = _merge_designed_region(
        full_pdb_with_dummy, hal_pdb_path, trb_path, missing_regions, work_dir
    )

    return merged_path
