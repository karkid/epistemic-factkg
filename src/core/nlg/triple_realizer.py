from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.ports.nlg.template import BaseTemplate

from src.core.graph.types import Triple

from src.core.semantics.lexicon.predicates import PredicateLexicon
from src.core.semantics.lexicon.entities import EntityLexicon

@dataclass(frozen=True)
class TripleRealizer():
    template: BaseTemplate
    pred_lexicon: PredicateLexicon
    ent_lexicon: EntityLexicon
    normallizer: Optional[callable[[str], str]] = None  # e.g. for normalizing object labels

    def realize(self, triple: Triple) -> str:
        s, p, o = triple.s, triple.p, triple.o

        if self.normallizer:
            s = self.normallizer(s)
            o = self.normallizer(o)
            p = self.normallizer(p)

        lex = self.pred_lexicon.get(p)
        if lex is None:
            return ""  # unknown predicate => skip


        s_lex = self.ent_lexicon.get(s)
        subj = s_lex.label if s_lex else s.lower()

        # object label
        o_lex = self.ent_lexicon.get(o)
        obj = o_lex.label if o_lex else o.lower()

        return self.template.render(
            triple=Triple(s=subj, p=lex.label, o=obj),
            kind=lex.kind
        )
    
    def realize_conjunction(self, triple1: Triple, triple2: Triple, conj: str = "and") -> str:
        s1, p1, o1 = triple1.s, triple1.p, triple1.o
        s2, p2, o2 = triple2.s, triple2.p, triple2.o

        if self.normallizer:
            s1 = self.normallizer(s1)
            o1 = self.normallizer(o1)
            p1 = self.normallizer(p1)

            s2 = self.normallizer(s2)
            o2 = self.normallizer(o2)
            p2 = self.normallizer(p2)

        lex1 = self.pred_lexicon.get(p1)
        lex2 = self.pred_lexicon.get(p2)

        if lex1 is None or lex2 is None:
            return ""  # unknown predicate => skip

        s1_lex = self.ent_lexicon.get(s1)
        subj1 = s1_lex.label if s1_lex else s1.lower()

        o1_lex = self.ent_lexicon.get(o1)
        obj1 = o1_lex.label if o1_lex else o1.lower()

        s2_lex = self.ent_lexicon.get(s2)
        subj2 = s2_lex.label if s2_lex else s2.lower()

        o2_lex = self.ent_lexicon.get(o2)
        obj2 = o2_lex.label if o2_lex else o2.lower()

        return self.template.render_conjunction(
            t1=Triple(s=subj1, p=lex1.label, o=obj1),
            k1=lex1.kind,
            t2=Triple(s=subj2, p=lex2.label, o=obj2),
            k2=lex2.kind,
            conj=conj,
        )
    
    def realize_negation(self, triple: Triple) -> str:
        s, p, o = triple.s, triple.p, triple.o

        if self.normallizer:
            s = self.normallizer(s)
            o = self.normallizer(o)
            p = self.normallizer(p)

        lex = self.pred_lexicon.get(p)
        if lex is None:
            return ""  # unknown predicate => skip

        s_lex = self.ent_lexicon.get(s)
        subj = s_lex.label if s_lex else s.lower()

        o_lex = self.ent_lexicon.get(o)
        obj = o_lex.label if o_lex else o.lower()

        return self.template.render_negation(
            triple=Triple(s=subj, p=lex.label, o=obj),
            kind=lex.kind
        )
