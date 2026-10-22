"""Validators: callables that raise ValidationError when a payload looks wrong.

They protect the cache from storing "Access Denied" pages, empty result sets or schema changes.
"""
from __future__ import annotations


class ValidationError(ValueError):
    pass


def nba_resultsets(min_rows: int = 1, required_headers: tuple = (), set_name: str | None = None):
    """Validator for stats.nba.com payloads: {'resultSets': [{'headers': [...], 'rowSet': [...]}]}."""
    def _check(payload):
        if not isinstance(payload, dict):
            raise ValidationError("payload is not a JSON object")
        sets = payload.get("resultSets", payload.get("resultSet"))
        if isinstance(sets, dict):
            sets = [sets]
        if not sets:
            raise ValidationError("payload has no resultSets")
        chosen = [s for s in sets if set_name is None or s.get("name") == set_name]
        if not chosen:
            raise ValidationError(f"result set {set_name!r} not found")
        first = chosen[0]
        headers = first.get("headers") or []
        missing = [h for h in required_headers if h not in headers]
        if missing:
            raise ValidationError(f"missing headers {missing} (schema change?)")
        if len(first.get("rowSet") or []) < min_rows:
            raise ValidationError(f"fewer than {min_rows} rows returned")
    return _check


def json_keys(*keys: str):
    def _check(payload):
        if not isinstance(payload, (dict, list)):
            raise ValidationError("payload is not JSON")
        if isinstance(payload, dict):
            missing = [k for k in keys if k not in payload]
            if missing:
                raise ValidationError(f"missing keys {missing}")
    return _check


def json_nonempty(payload):
    if not payload:
        raise ValidationError("empty JSON payload")


def html_contains(*needles: str, min_len: int = 2000):
    def _check(text):
        if not isinstance(text, str) or len(text) < min_len:
            raise ValidationError("page too short (blocked or error page?)")
        low = text.lower()
        for n in needles:
            if n.lower() not in low:
                raise ValidationError(f"expected marker {n!r} not found")
    return _check
