import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def pdbfixer_add_missing_atoms(pdb_path, step_num):
    new_pdb_path = f"step_{step_num}_add_missing_atoms.pdb"
    
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    with open(new_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
    return new_pdb_path, "pdbfixer_add_missing_atoms"

def get_repair_candidates(category):
    REPAIR_CANDIDATES = {
        "MISSING_ATOM": ["pdbfixer_add_missing_atoms"],
        "UNKNOWN": []
    }
    return REPAIR_CANDIDATES.get(category, [])
