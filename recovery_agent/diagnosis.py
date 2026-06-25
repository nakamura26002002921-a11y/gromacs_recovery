# recovery_agent/diagnosis.py
import re

ERROR_CATEGORIES = {
    "MISSING_HYDROGEN": r"Atom .* not found in rtp entry|Atom .*H.* not found",
    "MISSING_ATOM": r"atom .* is missing|Long Bond|atom .* is not found in the input file",
    "MISSING_RESIDUE_DB_ENTRY": r"Residue .* not found in residue topology database",
    "CHAIN_SPLIT": r"moleculetype|chain identifier",
    "TERMINUS_ISSUE": r"-ter",
}

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
    if fatal_section:
        search_target = fatal_section.replace('\n', ' ')
    else:
        search_target = stderr_text.replace('\n', ' ')
        
    for category, pattern in ERROR_CATEGORIES.items():
        if re.search(pattern, search_target, re.IGNORECASE):
            return category
            
    return "UNKNOWN"
