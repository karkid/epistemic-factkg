import inflect
import re

from src.core.graph.types import Triple
from src.core.ports.nlg.template import BaseTemplate
from src.core.semantics.lexicon.predicates import PredicateForm, PredicateLexicon
from src.core.nlg.sentence_template import SentenceTemplate
from typing import Optional


class Ai2ThorTemplate(BaseTemplate):

    def __init__(self, predicate_lexicon: Optional[PredicateLexicon] = None):
        super().__init__()

        self._inflect = inflect.engine()
        self._predicate_lexicon = predicate_lexicon

        self._templates = {}

        # Adjective: The apple is dirty.
        self._templates[PredicateForm.ADJ] = SentenceTemplate(
            template="The {s} is {p}.",
            fields={"s", "p"},
            normalize=True,
        )

        # Negated adjective: The apple is not dirty.
        self._templates[PredicateForm.ADJ + "_Negation"] = SentenceTemplate(
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

        # Negated verb: The apple does not touch the table.
        self._templates[PredicateForm.VERB + "_Negation"] = SentenceTemplate(
            template="The {s} does not {p} the {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # Preposition: An apple is on the table.
        self._templates[PredicateForm.PREP] = SentenceTemplate(
            template="{s} is {p} {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # Negated preposition: An apple is not on the table.
        self._templates[PredicateForm.PREP + "_Negation"] = SentenceTemplate(
            template="{s} is not {p} {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # Attribute: The apple's color is red.
        self._templates[PredicateForm.ATTR] = SentenceTemplate(
            template="The {s}'s {p} is {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # Negated attribute: The apple's color is not red.
        self._templates[PredicateForm.ATTR + "_Negation"] = SentenceTemplate(
            template="The {s}'s {p} is not {o}.",
            fields={"s", "p", "o"},
            normalize=True,
        )

        # State Property: The apple is openable. The cabinet is hot.
        self._templates[PredicateForm.PROP_STATE] = SentenceTemplate(
            template="The {s} is {p}.",
            fields={"s", "p"},
            normalize=True,
        )

        # Negated state property: The apple is not openable. The fridge is not hot.
        self._templates[PredicateForm.PROP_STATE + "_Negation"] = SentenceTemplate(
            template="The {s} is not {p}.",
            fields={"s", "p"},
            normalize=True,
        )

        # Value Property: The apple is hot.
        self._templates[PredicateForm.PROP_VALUE] = SentenceTemplate(
            template="The {s} is {o}.",
            fields={"s", "o"},
            normalize=True,
        )

        # Negated value property: The apple is not hot.
        self._templates[PredicateForm.PROP_VALUE + "_Negation"] = SentenceTemplate(
            template="The {s} is not {o}.",
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
    
    def format_with_and(self, text: str) -> str:
        parts = [p.strip() for p in text.split(",") if p.strip()]

        if len(parts) == 0:
            return ""

        if len(parts) == 1:
            return parts[0]

        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"

        # More than 2
        return ", ".join(parts[:-1]) + f" and {parts[-1]}"

    
    def _normalize_object(self, predicate: str, obj: str) -> str:

        if not isinstance(obj, str):
            return str(obj)

        obj = obj.strip()

        if not obj:
            return obj
        
        if predicate == "material":
            obj_norm = self.format_with_and(obj)
            return f"made up of {obj_norm}".lower()
        
        if predicate == "temperature":
            temperature_map = {
                "RoomTemp": "at room temperature",
                "Hot": "hot",
                "Cold": "cold"
            }
            return temperature_map.get(obj, obj.lower())
        
        if predicate == "mass":
            return f"weighs {obj} kg".lower()

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
    
    def strip_repeated_subject(self, sent1, sent2, subject):
        """
        Removes 'The <subject> is/are/was' from sent2 if present.
        """

        patterns = [
            rf"^the {re.escape(subject.lower())} is\s+",
            rf"^the {re.escape(subject.lower())} are\s+",
            rf"^the {re.escape(subject.lower())} was\s+",
        ]

        s2 = sent2.lower()

        for p in patterns:
            s2 = re.sub(p, "", s2)

        return s2
    
    def smart_and_join(self, sent1: str, sent2: str, conj="and") -> str:
        sent1 = sent1.rstrip(".")
        sent2 = sent2.rstrip(".")

        # Check if first sentence already has "and"
        has_and = " and " in sent1.lower()

        if conj == "and" and has_and:
            connector = ", and"
        else:
            connector = f" {conj}"

        return f"{sent1}{connector} {sent2.lower()}."


    # ------------------------
    # Main Render
    # ------------------------

    def render(self, triple: Triple, kind: PredicateForm, negation: bool = False) -> str:

        # Unpack triple
        # An apple is inside the box.
        # An apple is dirty.
        # An apple is hot.
        # An apple is made up of food and glass.
        # A door is openable.
        s, p, o = triple

        # ------------------------
        # Subject Handling
        # ------------------------
        if o == "false" or o == "False" or negation:
            kind = kind + "_Negation"

        # Only PREP needs a/an (An apple is on the table)
        if kind == PredicateForm.PREP:
            s_norm = self._with_article(s)
        else:
            s_norm = s

        # ------------------------
        # Object Handling
        # ------------------------

        o_norm = self._normalize_object(p, o)

        # ------------------------
        # Template Selection
        # ------------------------

        template = self._templates[kind]

        # ------------------------
        # Render Sentence
        # ------------------------
        

        return self._normalize_sentence(template.verbalize(s=s_norm, p=p, o=o_norm))
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
        # handel this type of sentence:
        # s1 : The apple is made up of food and glass.
        # s2 : The apple is dirty.
        # conj: and


        # Same subject → remove repetition
        if s1 == s2:

            # Remove "The apple " from second sentence
            reduced = self.strip_repeated_subject(sent1, sent2, s1)

            return self._normalize_sentence(self.smart_and_join(sent1=sent1, sent2=reduced, conj=conj))


        # Different subjects
        return self._normalize_sentence(self.smart_and_join(sent1=sent1, sent2=sent2, conj=conj))
    
    def render_negation(self, triple: Triple, kind: PredicateForm) -> str:
        return self.render(triple, kind, negation=True)