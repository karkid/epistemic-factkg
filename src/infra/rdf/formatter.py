from __future__ import annotations

from urllib.parse import unquote


def short_uri(x: str) -> str:
    """
    http://.../entities/Fork%7C%2B00... -> Fork|+00...
    http://.../relations/onTopOf -> onTopOf
    """
    x = unquote(str(x))
    if "#" in x:
        x = x.split("#")[-1]
    if "/" in x:
        x = x.rsplit("/", 1)[-1]
    return x


def ai2thor_object_type_from_entity_id(entity_id: str) -> str:
    """
    Fork|+00.62|+01.31|-02.48 -> Fork
    Fork_3 -> Fork
    """
    t = short_uri(entity_id)
    if "|" in t:
        return t.split("|", 1)[0]
    if "_" in t:
        return t.split("_", 1)[0]
    return t
