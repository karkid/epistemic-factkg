"""
Thin wrapper — real logic lives in src/adapters/ai2thor/converter.py.
Kept for backward compatibility with src/cli/convert_to_unified.py.
"""
from src.adapters.ai2thor.converter import AI2ThorConverter

_converter = AI2ThorConverter()


def convert_ai2thor_file(infile: str, outfile: str, split: str = None) -> int:
    return _converter.convert_file(infile, outfile, split)
