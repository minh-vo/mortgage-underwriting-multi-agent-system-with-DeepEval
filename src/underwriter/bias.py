"""Fair-lending screen: flag mentions of protected characteristics. It never blocks."""

import re

_TERMS = ["race", "racial", "ethnic", "ethnicity", "religion", "religious", "nationality",
          "national origin", "gender", "sex", "sexual orientation", "marital status",
          "married", "divorced", "pregnant", "pregnancy", "disability", "disabled",
          "age", "elderly", "immigrant", "citizenship"]
_PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t in _TERMS) + r")\b", re.IGNORECASE)
# "age" is fine when it describes an account or a building, not a person.
_BENIGN_AGE = re.compile(
    r"\b(average |the )?age of (the |an? )?(oldest |newest )?"
    r"(account|accounts|credit|tradeline|tradelines|property|roof|home|house|building|furnace|appraisal|file)",
    re.IGNORECASE)


def screen(text: str) -> list[str]:
    """Return one flag per protected-characteristic term found in the text."""
    flags = []
    for match in _PATTERN.finditer(text):
        if match.group(1).lower() == "age" and _BENIGN_AGE.match(text, _line_start(text, match)):
            continue
        start, end = max(0, match.start() - 40), min(len(text), match.end() + 40)
        flags.append(f"'{match.group(1).lower()}' in: ...{text[start:end].strip()}...")
    return flags


def _line_start(text: str, match: re.Match) -> int:
    """Where a benign 'age of ...' phrase containing this match would begin."""
    prefix = text[max(0, match.start() - 8):match.start()].lower()
    for lead in ("average ", "the "):
        if prefix.endswith(lead):
            return match.start() - len(lead)
    return match.start()
