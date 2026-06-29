# recovery_agent/diagnosis.py
import re

def _is_missing_heavy_atom(text):
    # 例: "the atom CG used in that entry is not found in the input file"
    match = re.search(r"\batom\s+(\w+)\s+used in that entry is not found", text, re.IGNORECASE)
    if match:
        return not match.group(1).upper().startswith("H")
    return False

def _is_missing_hydrogen(text):
    # 例: "Atom HB3 in residue LYS 3 was not found in rtp entry LYS"
    match = re.search(r"\batom\s+(\w+)\s+in residue\s+\w+\s+\d+\s+was not found in rtp entry", text, re.IGNORECASE)
    if match:
        return match.group(1).upper().startswith("H")
    return False

def _is_missing_residue_db_entry(text):
    return bool(re.search(r"Residue\s+\S+\s+not found in residue topology database", text, re.IGNORECASE))

def _is_hetero_chain_type_mismatch(text):
    return bool(re.search(r"do not have a consistent type", text, re.IGNORECASE))

def _is_chain_split_fatal(text):
    return bool(re.search(r"\bmoleculetype\b", text, re.IGNORECASE))

def _is_terminus_issue(text):
    return bool(re.search(r"-ter\b", text, re.IGNORECASE))

DIAGNOSIS_RULES = [
    ("MISSING_HYDROGEN", _is_missing_hydrogen),
    ("MISSING_ATOM", _is_missing_heavy_atom),
    ("MISSING_RESIDUE_DB_ENTRY", _is_missing_residue_db_entry),
    ("HETERO_CHAIN_TYPE_MISMATCH", _is_hetero_chain_type_mismatch),
    ("CHAIN_SPLIT", _is_chain_split_fatal),
    ("TERMINUS_ISSUE", _is_terminus_issue),
]

def extract_fatal_error(stderr_text):
    match = re.search(
        r"Fatal error:\s*(.*?)(?:\nFor more information|\n-{5,}|\Z)",
        stderr_text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None

def diagnose_error(stderr_text):
    fatal_section = extract_fatal_error(stderr_text)
    search_target = (fatal_section or stderr_text).replace('\n', ' ')

    matched = [name for name, fn in DIAGNOSIS_RULES if fn(search_target)]

    if len(matched) > 1:
        return f"AMBIGUOUS({'/'.join(matched)})"
    if len(matched) == 1:
        return matched[0]
    return "UNKNOWN"
