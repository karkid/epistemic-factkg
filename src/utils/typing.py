from typing import Dict, List, Set, Tuple, Any, NamedTuple


class Triple(NamedTuple):
    s: str
    p: str
    o: str
TripleSet = Set[Triple]
TripleList = List[Triple]
