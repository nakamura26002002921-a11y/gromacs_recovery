# recovery_agent/sequence_recovery.py
import os
import re
import requests
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser, PDBIO
from pdbfixer import PDBFixer

_THREE_TO_ONE = {k.upper(): v for k, v in protein_letters_3to1.items()}
_ONE_TO_THREE = {v: k.upper() for k, v in protein_letters_3to1.items()}


def fetch_rcsb_fasta(pdb_id, cache_dir=None, timeout=30):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{pdb_id}.fasta")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return f.read()
    resp = requests.get(f"https://www.rcsb.org/fasta/entry/{pdb_id}", timeout=timeout)
    resp.raise_for_status()
    if cache_dir:
        with open(cache_path, "w") as f:
            f.write(resp.text)
    return resp.text


def parse_rcsb_fasta(fasta_text):
    sequences = {}
    header, seq_lines = None, []

    def _flush(hdr, lines):
        if hdr is None:
            return
        seq = "".join(lines).upper()
        parts = hdr.split("|")
        if len(parts) >= 2:
            m = re.search(r"Chains?\s+([A-Za-z0-9,\s]+)", parts[1], re.IGNORECASE)
            if m:
                for c in m.group(1).split(","):
                    cid = c.strip().split()[0]
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


def _determine_offset(present, fasta_seq):
    anchor = "".join(aa for _, aa in present[:15])
    for length in (len(anchor), 10, 7):
        pos = fasta_seq.find(anchor[:length])
        if pos >= 0:
            return pos - present[0][0]
    return None


def _find_fasta_for_chain(present, fasta_sequences):
    best_cid, best_offset, best_id = None, None, 0.0
    for f_cid, f_seq in fasta_sequences.items():
        offset = _determine_offset(present, f_seq)
        if offset is None:
            continue
        match = sum(1 for rn, aa in present
                    if 0 <= rn + offset < len(f_seq) and aa == f_seq[rn + offset])
        identity = match / max(len(present), 1)
        if identity > best_id:
            best_id, best_cid, best_offset = identity, f_cid, offset
    if best_id < 0.80:
        return None, None
    return best_cid, best_offset


def apply_sequence_recovery(original_pdb_path, rfdiffusion_pdb_path, work_dir, pdb_id,
                            out_name="sequence_recovered.pdb", cache_dir=None):
    fixer = PDBFixer(filename=original_pdb_path)
    fixer.findMissingResidues()

    chain_data = {}
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        present = sorted(
            [(_parse_resnum(r.id), _THREE_TO_ONE.get(r.name.upper(), "X")) for r in residues],
            key=lambda x: x[0])
        missing = set()
        for (ci, pos), names in fixer.missingResidues.items():
            if ci != chain.index:
                continue
            gap_len = len(names)
            start = (_parse_resnum(residues[0].id) - gap_len if pos == 0
                     else _parse_resnum(residues[pos - 1].id) + 1)
            missing.update(range(start, start + gap_len))
        if missing:
            chain_data[chain.id] = (present, missing)

    fasta_sequences = parse_rcsb_fasta(fetch_rcsb_fasta(pdb_id, cache_dir=cache_dir))

    pattern_cache = {}
    corrections = {}
    for cid, (present, missing) in chain_data.items():
        key = ("".join(aa for _, aa in present), frozenset(missing))
        if key in pattern_cache:
            corrections[cid] = dict(pattern_cache[key])
            continue
        fasta_cid, offset = _find_fasta_for_chain(present, fasta_sequences)
        if fasta_cid is None:
            continue
        f_seq = fasta_sequences[fasta_cid]
        mapping = {rn: _ONE_TO_THREE.get(f_seq[rn + offset], "GLY")
                   for rn in sorted(missing) if 0 <= rn + offset < len(f_seq)}
        corrections[cid] = mapping
        pattern_cache[key] = mapping

    structure = PDBParser(QUIET=True).get_structure("seqrec", rfdiffusion_pdb_path)
    model = structure[0]
    for cid, resnum_to_resname in corrections.items():
        if cid not in model:
            continue
        for res in model[cid]:
            if res.id[0] == " " and res.id[1] in resnum_to_resname:
                res.resname = resnum_to_resname[res.id[1]]

    out_path = os.path.join(work_dir, out_name)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    return out_path


def _get_generated_resnums_from_original(original_pdb_path):
    fixer = PDBFixer(filename=original_pdb_path)
    fixer.findMissingResidues()
    pdb_complex_residues = {}
    generated_resnums_dict = {}
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        pdb_complex_residues[chain.id] = {_parse_resnum(r.id): r.name for r in residues}
        gen = set()
        for (ci, pos), names in fixer.missingResidues.items():
            if ci != chain.index:
                continue
            gap_len = len(names)
            start = (_parse_resnum(residues[0].id) - gap_len if pos == 0
                     else _parse_resnum(residues[pos - 1].id) + 1)
            gen.update(range(start, start + gap_len))
        if gen:
            generated_resnums_dict[chain.id] = gen
    return pdb_complex_residues, generated_resnums_dict
