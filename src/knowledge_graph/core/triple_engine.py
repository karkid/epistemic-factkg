from typing import Tuple


Triple = Tuple[str, str, str]


class TripleEngine:
    """A simple engine to manipulate RDF triples."""

    @staticmethod
    def invert_triple(triple: Triple) -> Triple:
        """Invert the subject and object of a triple."""
        s, p, o = triple
        return (o, p, s)

    @staticmethod
    def triple_to_str(triple: Triple) -> str:
        """Convert a triple to a string representation."""
        s, p, o = triple
        return f"({s}, {p}, {o})"

    @staticmethod
    def is_valid_triple(triple: Triple) -> bool:
        """Check if a triple is valid (non-empty subject, predicate, object)."""
        s, p, o = triple
        return all([s.strip(), p.strip(), o.strip()])
