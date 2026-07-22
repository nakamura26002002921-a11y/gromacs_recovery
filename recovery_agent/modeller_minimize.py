# recovery_agent/modeller_minimize.py
import os
import re


def _parse_resnum(res_id):
    m = re.search(r'-?\d+', str(res_id))
    return int(m.group()) if m else 0


def _get_repaired_resnums(original_pdb_path):
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=original_pdb_path)
    fixer.findMissingResidues()

    repaired = {}
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        gen = set()
        for (ci, pos), names in fixer.missingResidues.items():
            if ci != chain.index:
                continue
            gap_len = len(names)
            start = (_parse_resnum(residues[0].id) - gap_len if pos == 0
                     else _parse_resnum(residues[pos - 1].id) + 1)
            gen.update(range(start, start + gap_len))
        if gen:
            repaired[chain.id] = gen
    return repaired


def minimize_with_modeller(original_pdb_path, repaired_pdb_path, work_dir, modeller_config,
                           out_name="modeller_minimized.pdb"):
    from modeller import Environ, Selection, log
    from modeller.optimizers import ConjugateGradients, MolecularDynamics, actions
    from modeller.scripts import complete_pdb
    import modeller as _mod

    repaired_resnums = _get_repaired_resnums(original_pdb_path)
    if not repaired_resnums:
        return repaired_pdb_path

    _mod.license = modeller_config["license_key"]
    log.none()

    env = Environ()
    env.io.hetatm = False
    env.io.water = False
    env.io.atom_files_directory = [work_dir, os.path.dirname(os.path.abspath(repaired_pdb_path)) or ".", "."]
    env.libs.topology.read(file="$(LIB)/top_heav.lib")
    env.libs.parameters.read(file="$(LIB)/par.lib")
    env.edat.dynamic_sphere = True
    env.edat.contact_shell = 4.0

    mdl = complete_pdb(env, os.path.abspath(repaired_pdb_path))

    window = modeller_config.get("neighbor_window", 3)
    sel_residues = []
    for chain in mdl.chains:
        cid = chain.name.strip()
        if cid not in repaired_resnums:
            continue
        expanded = {n + w for n in repaired_resnums[cid] for w in range(-window, window + 1)}
        for res in chain.residues:
            try:
                if int(str(res.num).strip()) in expanded:
                    sel_residues.append(res)
            except ValueError:
                continue

    if not sel_residues:
        return repaired_pdb_path

    atmsel = Selection(sel_residues)
    cg_iter = modeller_config.get("cg_iterations", 200)
    md_iter = modeller_config.get("md_iterations", 200)
    temp_high = modeller_config.get("md_temperature_high", 1000)
    temp_low = modeller_config.get("md_temperature_low", 300)

    cg = ConjugateGradients(output="NO_REPORT")
    cg.optimize(atmsel, max_iterations=cg_iter)
    md = MolecularDynamics(output="NO_REPORT")
    md.optimize(atmsel, temperature=temp_high, max_iterations=md_iter,
                actions=[actions.trace(10, os.path.join(work_dir, "modeller_md_trace.log"))])
    md.optimize(atmsel, temperature=temp_low, max_iterations=md_iter)
    cg.optimize(atmsel, max_iterations=cg_iter)

    out_path = os.path.join(work_dir, out_name)
    mdl.write(file=out_path)
    return out_path
