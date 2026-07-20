# recovery_agent/missing_residues.py
from pdbfixer import PDBFixer

def count_missing_residues(pdb_path):
    """PDBFixerで検出できる欠損残基の総数を返す"""
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    return sum(len(names) for names in fixer.missingResidues.values())
