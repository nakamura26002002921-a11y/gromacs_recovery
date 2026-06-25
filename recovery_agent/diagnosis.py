import re

ERROR_CATEGORIES = {
    "MISSING_ATOM": r"(atom .* is missing|Long Bond)",
    "MISSING_RESIDUE_DB_ENTRY": r"Residue .* not found in residue topology database",
    "MISSING_HYDROGEN": r"Atom .*H.* not found",
    "CHAIN_SPLIT": r"(moleculetype|Chain identifier)",
    "TERMINUS_ISSUE": r"-ter",
}

def diagnose_error(error_log):
    for category, pattern in ERROR_CATEGORIES.items():
        if re.search(pattern, error_log, re.IGNORECASE):
            return category
    return "UNKNOWN"
