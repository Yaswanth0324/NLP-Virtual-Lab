"""
Lightweight shim for Python 3.13+ providing cgi.parse_header used by third-party
libraries (e.g., googletrans). The stdlib 'cgi' module was removed in Python 3.13.
This module implements only parse_header with a simple, practical parser.

Note: This is not a full replacement for the original cgi module.
"""
from typing import Dict, Tuple

__all__ = ["parse_header"]

def parse_header(line: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a Content-Type like header.
    Returns a tuple of (main_value, params_dict).

    Example:
    'text/html; charset=UTF-8' -> ('text/html', {'charset': 'UTF-8'})
    """
    if line is None:
        return "", {}
    if not isinstance(line, str):
        try:
            line = line.decode("utf-8", "ignore")
        except Exception:
            line = str(line)

    # Split on ';' to separate main value and parameters
    parts = [p.strip() for p in line.split(";") if p is not None and p.strip() != ""]
    if not parts:
        return "", {}

    main = parts[0]
    params: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip().strip('"')
            params[k] = v
        else:
            # Parameter without '='; record presence with empty value
            params[p.strip().lower()] = ""
    return main, params
