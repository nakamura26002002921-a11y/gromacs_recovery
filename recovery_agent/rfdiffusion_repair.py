# recovery_agent/rfdiffusion_repair.py
#
# 【重要】RFdiffusionはバックボーン(N, CA, C, O)のみを生成するモデルであり、
# アミノ酸配列(側鎖)は設計しない。公式ドキュメントの通り、設計された残基は
# 常にポリグリシン(GLY)として出力される。これはバグではなく仕様であり、
# 側鎖の座標に対して損失が適用されていないため、そのまま信頼できるアミノ酸種
# として扱ってはならない。
#
# したがって本モジュールの責務はRFdiffusionによるバックボーン生成と、
# 生成された新規残基(常にGLY)を複合体PDBへマージすることのみに限定する。
# 実際のアミノ酸配列の推定・割り当ては別ステップとして sequence_recovery.py
# 側で行う(オーケストレーションは graph.py を参照)。
import os
import subprocess
import re
import pickle
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Superimposer
from pdbfixer import PDBFixer

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


def _build_optimized_contig(valid_resnums, missing_regions, chain_id):
    """
    実在する座標ブロックと欠損ブロックを順番に並べ、RFdiffusionに渡す
    最適化された contig を生成する。

    【注意】contigmap.provide_seq は partial diffusion (diffuser.partial_T
    を設定するモード) 専用のオプションであり、通常のinpainting/design
    モードで渡すとRFdiffusion側の初期化で
        AssertionError: The provide_seq input is specifically for partial diffusion
    となり実行が失敗する。本エージェントは partial diffusion を使わず、
    contig の範囲指定（例: "A2-525"）だけで既存座標をそのまま保持させる
    通常のinpaintingモードを用いるため、provide_seq は一切生成・使用しない。
    """
    tokens = []

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

    for btype, s, e in all_blocks:
        if btype == 'existing':
            tokens.append(f"{chain_id}{s}-{e}")
        elif btype == 'missing':
            gap_len = e - s + 1
            tokens.append(f"{gap_len}-{gap_len}")

    # RFdiffusionのcontig記法では、同一鎖内で既存領域と新規生成領域を
    # 連結する際の区切りは "," ではなく "/" を使う仕様
    # (rfdiffusion/contigs.py: get_sampled_mask() 内で
    #  `contig_list = self.contigs[0].strip().split()` によりスペースで分割された
    #  各要素は、さらに `subcons = con.split("/")` でスラッシュ区切りに分解される)
    contig = "/".join(tokens)

    return contig


def _merge_single_chain_to_complex(current_complex_pdb, hal_pdb_path, trb_path, target_cid, regions, out_path):
    """
    RFdiffusionで生成された単一鎖の新規残基座標を、全体の複合体PDBにマージ（移植）する。

    RFdiffusionはバックボーンのみを設計するモデルであり、側鎖・配列の推定は
    行わない(公式仕様により常にGLYとして出力される)。そのため、ここでは
    生成された残基をそのままGLYとして複合体PDBに移植するだけにとどめ、
    アミノ酸名の再割り当ては行わない。正しい配列の推定・上書きは、この関数の
    呼び出し元とは別の後続ステップ(sequence_recovery.py)の責務とする。
    """
    with open(trb_path, "rb") as f:
        trb = pickle.load(f)

    # 【重要】RFdiffusionのcontigs.py (get_mappings) の仕様により、
    # con_hal_pdb_idx に格納されるhal側チェーンIDは、入力PDBの実際のチェーン文字
    # (例: 'B') とは無関係に、常に chain_order[0]='A' から採番される
    # (expand_sampled_mask() の inpaint_hal.extend([(chain_order[inpaint_chain_idx], i)...]) を参照)。
    # 単一鎖のinpaintingでは元のチェーンが 'A' か 'B' かによらず常にこの採番になるため、
    # チェーン文字同士を突き合わせる判定は信頼できない(元チェーンが 'A' の場合のみ
    # たまたま一致して見えてしまう)。
    #
    # 【バグ修正】以前の実装は「hal側で"既存(kept)"に含まれない残基を、
    # ファイル中の出現順のまま expected_missing_slots (実座標のresnum昇順リスト)
    # とzip()で1対1対応させる」という位置ベースの対応付けを行っていた。
    # これは、hal側の「既存として保持された残基の個数」が実座標側の
    # 「欠損領域直前までの既存残基の実resnum」と必ず一致する、という誤った前提に
    # 依存しており、鎖内に他の欠損・除去済み残基(_clean_and_extract_chain_pdbで
    # CA欠損などにより間引かれた残基等)が1つでもあると、生成された残基全体が
    # 1つズレて隣の実resnumに移植されてしまう(個数は一致するため例外は発生しない)。
    #
    # 正しい対応付けは、trbが直接提供する con_ref_pdb_idx <-> con_hal_pdb_idx の
    # ペア(「hal側のこの残基は実座標側のこの残基からコピーされた」という一次情報)
    # を「既存(kept)」判定に使い、そこに含まれないhal残基だけを新規生成として扱う。
    # さらに、新規生成領域をhal側resnumの昇順で並べ、real側のexpected_missing_slots
    # と対応付けることで、real側とhal側どちらの数値配列も「昇順・連続」という
    # 保証された性質のみに依拠するようにする(ズレの起きようがない対応付け)。
    con_ref_pdb_idx = trb.get("con_ref_pdb_idx", [])
    con_hal_pdb_idx = trb.get("con_hal_pdb_idx", [])
    if len(con_ref_pdb_idx) != len(con_hal_pdb_idx):
        raise RuntimeError(
            f"Chain {target_cid}: con_ref_pdb_idx (len={len(con_ref_pdb_idx)}) and "
            f"con_hal_pdb_idx (len={len(con_hal_pdb_idx)}) length mismatch in trb file; "
            f"cannot reliably map kept residues."
        )

    # kept (既存として保持された) hal側 (chain_id, resnum) の集合。
    # con_hal_pdb_idx の値は (chain_id, resnum) のタプルなので、resnumだけでなく
    # chain_idも含めてキーにする(hal側は複数チェーンに分かれることがあるため、
    # resnumだけで判定すると別チェーンの同一resnumと衝突しうる)。
    kept_hal_keys = {(v[0], v[1]) for v in con_hal_pdb_idx}

    # kept残基について real側resnum -> hal側(chain_id, resnum) の対応も保持しておく
    hal_by_real = {}
    for (ref_chain, ref_resnum), (hal_chain, hal_resnum) in zip(con_ref_pdb_idx, con_hal_pdb_idx):
        hal_by_real[int(ref_resnum)] = (hal_chain, int(hal_resnum))

    parser = PDBParser(QUIET=True)
    complex_struct = parser.get_structure("cpx", current_complex_pdb)
    hal_struct = parser.get_structure("hal", hal_pdb_path)
    target_chain = complex_struct[0][target_cid]

    # hal構造の全残基を (chain_id, resnum) -> Residue の辞書にしておく
    # (Superimposerの対応点探しと、新規生成残基の抽出の両方で使う)
    hal_res_by_key = {}
    for chain in hal_struct[0]:
        for res in chain:
            if res.id[0] == " ":
                hal_res_by_key[(chain.id, res.id[1])] = res

    # =====================================================================
    # 【座標系のアライメント】
    # RFdiffusionの出力(hal)は、入力複合体全体に対して並進・回転している
    # ことがある(measure_shift.pyで実測: 回転はほぼ無視できるが並進が
    # 数十Å規模で乗ることを確認済み)。kept(維持された)残基のCA原子同士を
    # 基準に hal_struct 全体を complex_struct の座標系へ重ね合わせてから
    # 新規残基を移植しないと、ペプチド結合が破綻した状態でマージされる。
    # =====================================================================
    ref_atoms = []
    hal_atoms = []
    for ref_resnum, hal_key in hal_by_real.items():
        ref_id = (" ", ref_resnum, " ")
        hal_res = hal_res_by_key.get(hal_key)
        if hal_res is None:
            continue
        if ref_id in target_chain and "CA" in target_chain[ref_id] and "CA" in hal_res:
            ref_atoms.append(target_chain[ref_id]["CA"])
            hal_atoms.append(hal_res["CA"])

    if len(ref_atoms) >= 3:
        sup = Superimposer()
        sup.set_atoms(ref_atoms, hal_atoms)
        sup.apply(list(hal_struct[0].get_atoms()))
        print(f"[Info] Chain {target_cid}: superimposed hal onto complex "
              f"(kept CA pairs={len(ref_atoms)}, RMSD={sup.rms:.3f} Å)")
    else:
        print(f"[Warning] Chain {target_cid}: not enough kept CA atoms to "
              f"superimpose (found {len(ref_atoms)}); merging without alignment.")

    # hal出力から「新規生成された残基」のみを抽出し、hal側resnumの昇順に並べる
    # (ファイル中の出現順に依存しない)
    newly_generated_hal_residues = []
    for chain in hal_struct[0]:
        for res in chain:
            if res.id[0] != " ":
                continue
            if (chain.id, res.id[1]) not in kept_hal_keys:
                newly_generated_hal_residues.append(res)
    newly_generated_hal_residues.sort(key=lambda r: r.id[1])

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

    # 【整合性チェック】各欠損領域の直前の実在残基(あれば)がkept mappingに
    # 存在し、そのhal側resnumが「新規生成ブロックの開始位置の直前」に
    # 連続しているかを検証する。ズレている場合は黙って移植せず、
    # ここで検知してRuntimeErrorとする(以前のバグはここが無かったため
    # 検出されずに1残基ズレたまま移植されていた)。
    hal_idx_cursor = 0
    for start_res, end_res in sorted(regions, key=lambda x: x[0]):
        gap_len = end_res - start_res + 1
        block = newly_generated_hal_residues[hal_idx_cursor:hal_idx_cursor + gap_len]
        hal_idx_cursor += gap_len

        prev_real_resnum = start_res - 1
        if prev_real_resnum in hal_by_real:
            expected_prev_hal_chain, expected_prev_hal_resnum = hal_by_real[prev_real_resnum]
            actual_first_hal = block[0].id[1]
            if actual_first_hal != expected_prev_hal_resnum + 1:
                raise RuntimeError(
                    f"Chain {target_cid}: hal numbering discontinuity detected before gap "
                    f"{start_res}-{end_res}. Residue {prev_real_resnum} (real) maps to hal "
                    f"resnum {expected_prev_hal_resnum} (chain {expected_prev_hal_chain}), but the "
                    f"newly-generated block starts at hal resnum {actual_first_hal} "
                    f"(expected {expected_prev_hal_resnum + 1}). "
                    f"Refusing to merge to avoid silently misassigning generated residues."
                )

    # 複合体PDBの対象チェーンへ外科的に移植し、正しい残基名を与える
    # (target_chainは上のSuperimposer準備時に既に取得済み)
    for resnum, hal_res in zip(expected_missing_slots, newly_generated_hal_residues):
        to_detach = [r.id for r in target_chain if r.id[1] == resnum and r.id[0] == " "]
        for rid in to_detach:
            target_chain.detach_child(rid)
            
        new_res = hal_res.copy()
        new_res.id = (" ", resnum, " ")

        # RFdiffusionはバックボーンのみ設計するため、新規残基は常にGLYのまま
        # 複合体PDBへ移植する(側鎖・配列の推定は後続の sequence_recovery ステップで行う)
        new_res.resname = "GLY"

        target_chain.add(new_res)

    # target_chain.child_list.sort(...) のようにBiopythonの内部リストを直接
    # ソートするのは非推奨(child_dictとの不整合を招く恐れがある)。
    # 全残基を一度detachしてからソート順にaddし直す、正規のAPIのみに
    # 依拠した安全な方法で並び順を揃える。
    all_res = list(target_chain)
    for res in all_res:
        target_chain.detach_child(res.id)
    all_res.sort(key=lambda r: (r.id[1], r.id[2]))
    for res in all_res:
        target_chain.add(res)

    io = PDBIO()
    io.set_structure(complex_struct)
    io.save(out_path)


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    """
    メイン・エントリポイント。

    欠損領域のバックボーンをRFdiffusionで生成し、複合体PDBへマージして返す。
    生成された新規残基は常にGLY(公式仕様どおりバックボーンのみ・配列は未設計)
    のままとなる。pdb_id / FASTAを用いた配列の復元はここでは行わない
    (呼び出し側で sequence_recovery.py を後段に実行すること)。

    :param pdb_id: 後方互換のため引数として残しているが、本関数内では未使用。
        配列復元に使うpdb_idは graph.py 側で sequence_recovery ステップに渡す。
    """
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()

    pdb_complex_residues, generated_resnums_dict, missing_regions = _get_expected_missing_resnums(fixer)

    if not missing_regions:
        return pdb_path

    current_complex_pdb = pdb_path
    
    for cid, regions in missing_regions.items():
        print(f"[Info] Processing chain {cid} for missing regions: {regions}")
        
        # 1. 複合体から対象鎖のみを抽出し、座標異常を除去 + Jittering適用
        clean_pdb_path, valid_resnums = _clean_and_extract_chain_pdb(current_complex_pdb, cid, work_dir)
        
        # 2. 最適化されたcontigの作成 (provide_seqは使用しない。理由は
        #    _build_optimized_contig() のdocstring参照)
        contig = _build_optimized_contig(valid_resnums, regions, cid)
        print(f"[Info] Optimized contig for chain {cid}: {contig}")
        
        # 3. 引数の構築
        out_prefix = os.path.join(work_dir, f"rf_out_chain_{cid}")
        cmd = [
            "python", rf_config["script_path"],
            f"inference.output_prefix={out_prefix}",
            f"inference.input_pdb={os.path.abspath(clean_pdb_path)}",
            f"inference.model_directory_path={rf_config['model_directory_path']}",
            f'contigmap.contigs=["{contig}"]',
            f"inference.num_designs={rf_config.get('num_designs', 1)}",
        ]
            
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
        
        # 4. 修復された鎖の一部（新規残基, GLYのまま）を、複合体PDBにマージ
        next_complex_pdb = os.path.join(work_dir, f"merged_step_chain_{cid}.pdb")
        _merge_single_chain_to_complex(
            current_complex_pdb, hal_pdb_path, trb_path, cid, regions, next_complex_pdb
        )
        
        current_complex_pdb = next_complex_pdb

    final_out_path = os.path.join(work_dir, "rfdiffusion_final_merged.pdb")
    os.rename(current_complex_pdb, final_out_path)
    
    return final_out_path
