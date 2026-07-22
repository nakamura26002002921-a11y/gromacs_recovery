# recovery_agent/rfdiffusion_repair.py
#
# RFdiffusionはバックボーン(N,CA,C,O)のみ生成し、新規残基は常にGLY。
# 配列の復元は sequence_recovery.py の責務。
import os
import subprocess
import re
import pickle
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Superimposer
from pdbfixer import PDBFixer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _parse_resnum(res_id):
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


def _get_expected_missing_resnums(fixer):
    pdb_complex_residues = {}
    generated_resnums_dict = {}
    missing_regions = {}

    for chain in fixer.topology.chains():
        cid = chain.id
        residues = list(chain.residues())
        if not residues:
            continue
        pdb_complex_residues[cid] = {_parse_resnum(r.id): r.name for r in residues}

    for chain in fixer.topology.chains():
        cid = chain.id
        residues = list(chain.residues())
        gaps = sorted(
            ((pos, names) for (ci, pos), names in fixer.missingResidues.items()
             if ci == chain.index),
            key=lambda x: x[0],
        )
        gen_resnums = set()
        regions = []
        for pos, names in gaps:
            gap_len = len(names)
            start = (_parse_resnum(residues[0].id) - gap_len if pos == 0
                     else _parse_resnum(residues[pos - 1].id) + 1)
            end = start + gap_len - 1
            regions.append((start, end))
            gen_resnums.update(range(start, end + 1))
        if gen_resnums:
            generated_resnums_dict[cid] = gen_resnums
            missing_regions[cid] = regions

    return pdb_complex_residues, generated_resnums_dict, missing_regions


def _clean_and_extract_chain_pdb(original_pdb_path, target_chain_id, work_dir):
    parser = PDBParser(QUIET=True)
    model = parser.get_structure("orig", original_pdb_path)[0]
    new_chain = model[target_chain_id].copy()
    valid_resnums = []

    for res in list(new_chain):
        if res.id[0] != " " or "CA" not in res:
            new_chain.detach_child(res.id)
            continue
        ca = res["CA"].get_coord()
        if np.any(np.isnan(ca)) or np.any(np.isinf(ca)):
            new_chain.detach_child(res.id)
            continue
        for atom in res:
            atom.set_coord(atom.get_coord() + np.random.normal(0, 1e-4, 3))
        valid_resnums.append(res.id[1])

    new_struct = Structure.Structure("cleaned")
    new_model = Model.Model(0)
    new_model.add(new_chain)
    new_struct.add(new_model)

    out_path = os.path.join(work_dir, f"cleaned_chain_{target_chain_id}.pdb")
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(out_path)
    return out_path, sorted(valid_resnums)


def _build_optimized_contig(valid_resnums, missing_regions, chain_id):
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

    all_blocks = [("existing", s, e) for s, e in segments]
    all_blocks += [("missing", s, e) for s, e in sorted(missing_regions, key=lambda x: x[0])]
    all_blocks.sort(key=lambda x: x[1])

    tokens = []
    for btype, s, e in all_blocks:
        if btype == "existing":
            tokens.append(f"{chain_id}{s}-{e}")
        else:
            tokens.append(f"{e - s + 1}-{e - s + 1}")

    return "/".join(tokens)


def _merge_single_chain_to_complex(current_complex_pdb, hal_pdb_path, trb_path,
                                   target_cid, regions, out_path):
    with open(trb_path, "rb") as f:
        trb = pickle.load(f)

    con_ref = trb.get("con_ref_pdb_idx", [])
    con_hal = trb.get("con_hal_pdb_idx", [])

    kept_hal_resnums = {int(v[1]) for v in con_hal}
    hal_by_real = {int(r[1]): int(h[1]) for r, h in zip(con_ref, con_hal)}

    parser = PDBParser(QUIET=True)
    complex_struct = parser.get_structure("cpx", current_complex_pdb)
    hal_struct = parser.get_structure("hal", hal_pdb_path)
    target_chain = complex_struct[0][target_cid]
    hal_chain = list(hal_struct[0])[0]

    hal_res_by_resnum = {int(r.id[1]): r for r in hal_chain if r.id[0] == " "}

    # 座標系アライメント (kept CA で重ね合わせ)
    ref_atoms, hal_atoms = [], []
    for real_rn, hal_rn in hal_by_real.items():
        ref_id = (" ", real_rn, " ")
        hal_res = hal_res_by_resnum.get(hal_rn)
        if hal_res and ref_id in target_chain and "CA" in target_chain[ref_id] and "CA" in hal_res:
            ref_atoms.append(target_chain[ref_id]["CA"])
            hal_atoms.append(hal_res["CA"])

    if len(ref_atoms) >= 3:
        sup = Superimposer()
        sup.set_atoms(ref_atoms, hal_atoms)
        sup.apply(list(hal_struct[0].get_atoms()))

    # 新規生成残基を抽出
    new_residues = sorted(
        [r for r in hal_chain if r.id[0] == " " and int(r.id[1]) not in kept_hal_resnums],
        key=lambda r: r.id[1])

    expected_slots = [rn for s, e in sorted(regions) for rn in range(s, e + 1)]

    # 移植
    for resnum, hal_res in zip(expected_slots, new_residues):
        for rid in [r.id for r in target_chain if r.id[1] == resnum and r.id[0] == " "]:
            target_chain.detach_child(rid)
        new_res = hal_res.copy()
        new_res.id = (" ", resnum, " ")
        new_res.resname = "GLY"
        target_chain.add(new_res)

    # 残基順ソート
    all_res = list(target_chain)
    for r in all_res:
        target_chain.detach_child(r.id)
    for r in sorted(all_res, key=lambda r: (r.id[1], r.id[2])):
        target_chain.add(r)

    io = PDBIO()
    io.set_structure(complex_struct)
    io.save(out_path)


def run_rfdiffusion(pdb_path, work_dir, rf_config, pdb_id=None):
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    _, _, missing_regions = _get_expected_missing_resnums(fixer)

    if not missing_regions:
        return pdb_path

    current_pdb = pdb_path
    for cid, regions in missing_regions.items():
        clean_path, valid_resnums = _clean_and_extract_chain_pdb(current_pdb, cid, work_dir)
        contig = _build_optimized_contig(valid_resnums, regions, cid)

        out_prefix = os.path.join(work_dir, f"rf_out_chain_{cid}")
        cmd = [
            "python", rf_config["script_path"],
            f"inference.output_prefix={out_prefix}",
            f"inference.input_pdb={os.path.abspath(clean_path)}",
            f"inference.model_directory_path={rf_config['model_directory_path']}",
            f'contigmap.contigs=["{contig}"]',
            f"inference.num_designs={rf_config.get('num_designs', 1)}",
        ]
        subprocess.run(cmd, capture_output=True, text=True,
                       cwd=work_dir, timeout=rf_config.get("timeout_sec", 1800),
                       check=True)

        hal_path = f"{out_prefix}_0.pdb"
        trb_path = f"{out_prefix}_0.trb"
        next_pdb = os.path.join(work_dir, f"merged_step_chain_{cid}.pdb")
        _merge_single_chain_to_complex(current_pdb, hal_path, trb_path, cid, regions, next_pdb)
        current_pdb = next_pdb

    final_path = os.path.join(work_dir, "rfdiffusion_final_merged.pdb")
    os.rename(current_pdb, final_path)
    return final_path
