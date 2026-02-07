from __future__ import annotations
import inflect

_ENGINE = inflect.engine()

def get_engine() -> inflect.engine:
    return _ENGINE
