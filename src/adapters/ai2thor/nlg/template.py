import inflect

from src.core.graph.types import Triple
from src.core.ports.nlg.template import BaseTemplate
from src.core.semantics.lexicon.predicates import PredicateForm
from src.core.nlg.sentence_template import SentenceTemplate


class Ai2ThorTemplate(BaseTemplate):

    def __init__(self):
        super().__init__()

        self._inflect = inflect.engine()

        self._templates = {}

        # Adjective: The apple is dirty.
        self._templates[PredicateForm.ADJ] = SentenceTemplate(
            template="The {s} is {p}.",
            fields={"s", "p"},
            normalize=True,
        )

        # Negated adjective: The apple is not dirty.
        self._templates[PredicateForm.ADJ + "_False"] = SentenceTemplate(
            template="The {s} is not {p}.",
            fields={"s", "p"},
            normalize=True,
        )

        # Verb: The apple touches the table.
        self._templates[PredicateForm.VERB] = SentenceTemplate(
            template="The {s} {p} the {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # Preposition: An apple is on the table.
        self._templates[PredicateForm.PREP] = SentenceTemplate(
            template="{s} is {p} {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # Attribute: The apple's color is red.
        self._templates[PredicateForm.ATTR] = SentenceTemplate(
            template="The {s}'s {p} is {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # Property: The apple is a receptacle.
        self._templates[PredicateForm.PROP] = SentenceTemplate(
            template="The {s} is {o}.",
            fields={"s", "o"},
            normalize=True,
        )

    # ------------------------
    # Grammar Helpers
    # ------------------------

    def _with_article(self, noun: str) -> str:
        """
        Add correct a/an article.
        Example: apple -> an apple
        """
        return self._inflect.a(noun)

    def _normalize_object(self, obj: str, kind: PredicateForm) -> str:

        if not isinstance(obj, str):
            return str(obj)

        obj = obj.strip()

        if not obj:
            return obj

        # Attribute / adjective values → no article
        if kind in {PredicateForm.ATTR, PredicateForm.ADJ, PredicateForm.PROP}:
            return obj.lower()

        # Proper noun
        if obj[0].isupper():
            return obj

        return f"the {obj}"

    
    def _normalize_sentence(self, sentence: str) -> str:
        """
        Basic normalization: spacing + capitalization.
        """

        # remove extra spaces
        sentence = " ".join(sentence.strip().split())

        # fix duplicate articles
        sentence = sentence.replace("the the ", "the ")
        sentence = sentence.replace("The the ", "The ")

        # capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        # Sentenc case 
        return sentence

    # ------------------------
    # Main Render
    # ------------------------

    def render(self, triple: Triple, kind: PredicateForm) -> str:

        # Unpack triple
        s, p, o = triple

        # ------------------------
        # Subject Handling
        # ------------------------

        # Only PREP needs a/an (An apple is on the table)
        if kind == PredicateForm.PREP:
            s_norm = self._with_article(s)
        else:
            s_norm = s

        # ------------------------
        # Object Handling
        # ------------------------

        o_norm = self._normalize_object(o, kind)

        # ------------------------
        # Template Selection
        # ------------------------

        if kind not in self._templates:

            # Handle negation fallback
            if kind.endswith("_False") and kind[:-6] in self._templates:
                template = self._templates[kind + "_False"]
            else:
                return ""

        else:
            template = self._templates[kind]

        # ------------------------
        # Render Sentence
        # ------------------------

        return self._normalize_sentence(template.verbalize(
            s=s_norm,
            p=p,
            o=o_norm,
        ))
    # ------------------------
    # Conjunction Render
    # ------------------------
    def render_conjunction(
        self,
        t1: Triple,
        k1: PredicateForm,
        t2: Triple,
        k2: PredicateForm,
        conj: str = "and",
    ) -> str:

        s1, _, _ = t1
        s2, _, _ = t2

        sent1 = self.render(t1, k1)
        sent2 = self.render(t2, k2)

        if not sent1 or not sent2:
            return ""

        sent1 = sent1.rstrip(".")
        sent2 = sent2.rstrip(".").lower()

        # Same subject → remove repetition
        if s1 == s2:

            # Remove "The apple " from second sentence
            prefix = f"The {s2} "

            if sent2.startswith(prefix):
                sent2 = sent2[len(prefix):]

            return f"{sent1} {conj} {sent2}."

        # Different subjects
        return self._normalize_sentence(f"{sent1} {conj} {sent2}.")
    
    def render_negation(self, triple: Triple, kind: PredicateForm) -> str:
        
        if kind in PredicateForm.ADJ:
            neg_kind = kind + "_False"
            return self.render(triple, neg_kind)
        return ""