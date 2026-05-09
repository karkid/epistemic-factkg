"""
Thin wrapper — real logic lives in src/adapters/averitec/converter.py.
Kept for backward compatibility with any external callers.
"""

from src.adapters.averitec.converter import AveritecConverter

_converter = AveritecConverter()


def convert_averitec_file(infile: str, outfile: str, split_name: str) -> int:
    return _converter.convert_file(infile, outfile, split=split_name)
