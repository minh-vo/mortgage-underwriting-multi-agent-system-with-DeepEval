"""Policy lookup: split data/policies.md into numbered sections and search them."""

import re
from functools import lru_cache

from .config import POLICY_PATH

_HEADING = re.compile(r"^## (\d\.\d) (.+)$", re.MULTILINE)
_STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "is", "are", "on",
              "what", "how", "with", "be", "by", "at", "it", "that", "this", "policy", "rule"}


@lru_cache
def sections() -> dict[str, dict]:
    """Return {"2.5": {"title": ..., "text": ...}} for every numbered section."""
    text = POLICY_PATH.read_text(encoding="utf-8")
    matches = list(_HEADING.finditer(text))
    result = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end]
        body = re.split(r"^# ", body, maxsplit=1, flags=re.MULTILINE)[0].strip()
        result[match.group(1)] = {"title": match.group(2).strip(), "text": body}
    return result


def _format(section_id: str) -> str:
    section = sections()[section_id]
    return f"[Policy {section_id} {section['title']}]\n{section['text']}"


def get_section(section: str) -> str:
    """Return one section ("2.5") or a whole chapter ("2"), with section numbers."""
    section = section.strip().rstrip(".")
    if section in sections():
        return _format(section)
    chapter = [sid for sid in sections() if sid.split(".")[0] == section]
    if chapter:
        return "\n\n".join(_format(sid) for sid in chapter)
    return f"No policy section '{section}'. Available sections: {', '.join(sections())}"


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def search(query: str, limit: int = 3) -> str:
    """Return the sections that best match the query words, with section numbers."""
    terms = set(_words(query))
    scored = []
    for sid, section in sections().items():
        title_words, body_words = _words(section["title"]), _words(section["text"])
        score = sum(3 * title_words.count(t) + body_words.count(t) for t in terms)
        if score:
            scored.append((score, sid))
    if not scored:
        return f"No policy section matches '{query}'. Available sections: {', '.join(sections())}"
    scored.sort(key=lambda item: (-item[0], item[1]))
    return "\n\n".join(_format(sid) for _, sid in scored[:limit])
