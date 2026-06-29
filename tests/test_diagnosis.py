# tests/test_diagnosis.py
import pytest
from recovery_agent.diagnosis import diagnose_error

def test_missing_heavy_atom_arg():
    """1AONで観測された、重原子(CG)の欠損エラー"""
    stderr = """Fatal error:
Residue 196 named ARG of a molecule in the input file was mapped
to an entry in the topology database, but the atom CG used in
that entry is not found in the input file. Perhaps your atom
and/or residue naming needs to be fixed.
"""
    assert diagnose_error(stderr) == "MISSING_ATOM"

def test_missing_heavy_atom_leu():
    """1CLLで観測された、重原子(CB)の欠損エラー"""
    stderr = """Fatal error:
Residue 1 named LEU of a molecule in the input file was mapped
to an entry in the topology database, but the atom CB used in
that entry is not found in the input file. Perhaps your atom
and/or residue naming needs to be fixed.
"""
    assert diagnose_error(stderr) == "MISSING_ATOM"

def test_missing_hydrogen_lys():
    """PDBFixer修復後に観測された、水素原子(HB3)の命名不一致エラー"""
    stderr = """Fatal error:
Atom HB3 in residue LYS 3 was not found in rtp entry LYS with 22 atoms
while sorting atoms.
"""
    assert diagnose_error(stderr) == "MISSING_HYDROGEN"

def test_missing_hydrogen_asp():
    """1CFDで観測された、水素原子(HB3)の命名不一致エラー"""
    stderr = """Fatal error:
Atom HB3 in residue ASP 2 was not found in rtp entry ASP with 12 atoms
while sorting atoms.
"""
    assert diagnose_error(stderr) == "MISSING_HYDROGEN"

def test_hetero_chain_type_mismatch():
    """1AON修復後に観測された、補因子とタンパク質鎖のタイプ不一致エラー"""
    stderr = """Fatal error:
The residues in the chain MG1--ADP2 do not have a consistent type. The first
residue has type 'Ion', while residue ADP2 is of type 'Other'...
"""
    assert diagnose_error(stderr) == "HETERO_CHAIN_TYPE_MISMATCH"

def test_unknown_error():
    """どのパターンにもマッチしない未知のエラー"""
    stderr = """Fatal error:
Something completely unexpected happened that we have never seen before.
"""
    assert diagnose_error(stderr) == "UNKNOWN"
