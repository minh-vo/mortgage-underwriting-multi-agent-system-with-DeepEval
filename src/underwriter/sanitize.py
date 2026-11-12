"""PII masking. Runs before the graph, so nothing downstream ever sees raw PII."""

import copy
import re

_SSN = re.compile(r"\b\d{3}-\d{2}-(\d{4})\b")
_PHONE = re.compile(r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?(\d{4})\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


def _mask_text(text: str, name: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _SSN.sub(lambda m: f"***-**-{m.group(1)}", text)
    text = _PHONE.sub(lambda m: f"***-***-{m.group(1)}", text)
    for part in [name, *name.split()]:
        if len(part) > 2:
            text = re.sub(rf"\b{re.escape(part)}\b", "[APPLICANT]", text, flags=re.IGNORECASE)
    return text


def _walk(value, name: str):
    if isinstance(value, str):
        return _mask_text(value, name)
    if isinstance(value, dict):
        return {key: _walk(item, name) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, name) for item in value]
    return value


def sanitize(application: dict) -> dict:
    """Return a masked copy of the application. The input is not changed."""
    app = copy.deepcopy(application)
    person = app.pop("applicant")
    name = person["name"]
    app = _walk(app, name)
    app["applicant"] = {
        "name": "[APPLICANT]",
        "ssn": f"***-**-{person['ssn'][-4:]}",
        "phone": f"***-***-{person['phone'][-4:]}",
        "email": "[EMAIL]",
        "address": "[ADDRESS]",
    }
    return app


def pii_table(application: dict, sanitized: dict) -> list[dict]:
    """Before/after rows for the UI."""
    return [{"field": field, "before": application["applicant"][field],
             "after": sanitized["applicant"][field]}
            for field in ("name", "ssn", "phone", "email", "address")]
