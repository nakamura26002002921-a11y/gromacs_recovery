# recovery_agent/diagnosis.py
import re

ERROR_CATEGORIES = {
    "MISSING_ATOM": r"(atom .* is missing|Long Bond)",
    "MISSING_RESIDUE_DB_ENTRY": r"Residue .* not found in residue topology database",
    "MISSING_HYDROGEN": r"Atom .*H.* not found",
    "CHAIN_SPLIT": r"(moleculetype|chain identifier)",
    "TERMINUS_ISSUE": r"-ter",
}

def extract_fatal_error(stderr_text):
    """GROMACSの出力から'Fatal error:'セクションのみを抜き出す"""
    # GROMACSの典型的なFatal errorブロック:
    # Fatal error:
    # <error message>
    # 
    # For more information and tips for troubleshooting...
    match = re.search(
        r"Fatal error:\s*(.*?)(?:\nFor more information|\n-{5,}|\Z)", 
        stderr_text, 
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None  # Fatal errorセクションが見つからない場合

def diagnose_error(stderr_text):
    """
    優先順位:
    1. 'Fatal error:'セクションが見つかれば、そこだけを対象に分類する(警告文を誤検知しないため)
    2. 見つからない場合のみ、全文を対象にフォールバック分類する
    """
    fatal_section = extract_fatal_error(stderr_text)
    search_target = fatal_section if fatal_section else stderr_text

    for category, pattern in ERROR_CATEGORIES.items():
        if re.search(pattern, search_target, re.IGNORECASE):
            return category

    return "UNKNOWN"
