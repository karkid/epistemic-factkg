from typing import List, Set, Optional
import random

from utils.typing import Triple, TripleList, TripleSet
from knowledge_graph.semantics.source.base import ClaimInstance, ClaimCorpus
from knowledge_graph.semantics.claims.base import BaseClaimGenerator, BaseClaimCorruptor

class ClaimGenerationPipeline:
    """
    Pipeline for generating supported/refuted claims
    from knowledge graph triples.
    """

    def __init__(
        self,
        *,
        generator: BaseClaimGenerator,
        corruptor: BaseClaimCorruptor,
        ontology,
        evidence_source_type: str,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        generator : BaseClaimGenerator
            Claim verbalizer
        corruptor : BaseClaimCorruptor
            Triple corruption logic
        ontology : object
            Ontology helper (AI2THOROntology, etc.)
        evidence_source_type : str
            e.g. "perception", "simulation"
        seed : int, optional
            Random seed
        """

        self.generator = generator
        self.corruptor = corruptor
        self.ontology = ontology
        self.evidence_source_type = evidence_source_type

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Core Helpers
    # ------------------------------------------------------------------

    def _make_claim(
        self,
        *,
        rec_id: str,
        claim: str,
        label: str,
        claim_triples: TripleList,
        evidence_triples: TripleList,
        scene_id: str,
        reasoning: str,
        notes: Optional[str] = None,
    ) -> ClaimInstance:
        """Factory for ClaimInstance."""

        return ClaimInstance.make_instance(
            rec_id=rec_id,
            claim=claim,
            label=label,
            claim_triples=claim_triples,
            structural_reasoning=reasoning,
            evidence_triples=evidence_triples,
            evidence_source="AI2-THOR-RDF",
            evidence_source_type=self.evidence_source_type,
            evidence_urls=[],
            scene_id=scene_id,
            generator="ai2thor",
            split=None,
            notes=notes,
        )

    def _random_triple(self, triples: TripleList) -> Triple:
        return random.choice(triples)

    def _is_valid_claim(self, text: str) -> bool:
        return bool(text and text.strip())

    def _infer_reasoning(
        self,
        claim: str,
        evidence: List[Triple],
    ) -> str:

        c = claim.lower()

        if " and " in c:
            return "conjunction"

        if " not " in f" {c} ":
            return "negation"

        return "one-hop" if len(evidence) == 1 else "conjunction"

    def _make_conjunction_text(self, c1: str, c2: str) -> str:

        c1 = c1.strip().rstrip(".")
        c2 = c2.strip()

        if c2:
            c2 = c2[0].lower() + c2[1:]

        return f"{c1} and {c2}"

    # ------------------------------------------------------------------
    # One-hop Generation
    # ------------------------------------------------------------------

    def generate_one_hop(
        self,
        *,
        scene_id: str,
        triples: TripleList,
        triple_set: TripleSet,
        n_supported: int,
        n_refuted: int,
        seed_prefix: str,
        max_sup_attempts: int = 10,
        max_ref_attempts: int = 15,
        debug: bool = False,
    ) -> ClaimCorpus:
        """
        Generate supported and refuted one-hop claims.
        """

        corpus = ClaimCorpus()

        # ---------------- SUPPORTED ---------------- #

        sup_target = n_supported
        sup_attempts = sup_target * max_sup_attempts

        sup_count = 0
        attempts = 0

        while sup_count < sup_target and attempts < sup_attempts:

            triple = self._random_triple(triples)

            claim = self.generator.verbalize(triple)

            if not self._is_valid_claim(claim):
                attempts += 1
                continue

            rec_id = f"{seed_prefix}-{scene_id}-onehop-sup-{sup_count:06d}"

            instance = self._make_claim(
                rec_id=rec_id,
                claim=claim,
                label="SUPPORTED",
                claim_triples=[triple],
                evidence_triples=[triple],
                scene_id=scene_id,
                reasoning="one-hop",
            )

            corpus.add_claim(instance)

            sup_count += 1
            attempts += 1

        # ---------------- REFUTED ---------------- #

        ref_target = n_refuted
        ref_attempts = ref_target * max_ref_attempts

        ref_count = 0
        attempts = 0

        while ref_count < ref_target and attempts < ref_attempts:

            evidence = self._random_triple(triples)

            corrupted = self.corruptor.corrupt_triple(
                evidence,
                triples,
                triple_set,
            )

            if corrupted in triple_set:

                if debug:
                    print("Corruption hit true triple:", corrupted)

                attempts += 1
                continue

            claim = self.generator.verbalize(corrupted)

            if not self._is_valid_claim(claim):
                attempts += 1
                continue

            rec_id = f"{seed_prefix}-{scene_id}-onehop-ref-{ref_count:06d}"

            instance = self._make_claim(
                rec_id=rec_id,
                claim=claim,
                label="REFUTED",
                claim_triples=[corrupted],
                evidence_triples=[evidence],
                scene_id=scene_id,
                reasoning="one-hop",
                notes="corrupted",
            )

            corpus.add_claim(instance)

            ref_count += 1
            attempts += 1

        # ---------------- WARNINGS ---------------- #

        if debug:

            if sup_count < sup_target:
                print(f"[WARN] Supported: {sup_count}/{sup_target}")

            if ref_count < ref_target:
                print(f"[WARN] Refuted: {ref_count}/{ref_target}")

        return corpus

    # ------------------------------------------------------------------
    # Negation Pairs
    # ------------------------------------------------------------------

    def generate_negation_pairs(
        self,
        *,
        scene_id: str,
        triples: TripleList,
        triple_set: TripleSet,
        max_pairs: int,
        seed_prefix: str,
    ) -> ClaimCorpus:

        corpus = ClaimCorpus()

        bool_triples = [
            t for t in triples
            if self.ontology.is_state_predicate(t.p)
        ]

        if not bool_triples:
            return corpus

        sample = random.sample(
            bool_triples,
            k=min(max_pairs, len(bool_triples)),
        )

        for idx, triple in enumerate(sample):

            claim_true = self.generator.verbalize(triple)

            if not self._is_valid_claim(claim_true):
                continue

            flipped_o = (
                "True"
                if str(triple.o).lower() == "false"
                else "False"
            )

            t_flip = (triple.s, triple.p, flipped_o)

            claim_flip = self.generator.verbalize(t_flip)

            if not self._is_valid_claim(claim_flip):
                continue

            true_supported = triple in triple_set
            flip_supported = t_flip in triple_set

            lab_true = "SUPPORTED" if true_supported else "REFUTED"
            lab_flip = "SUPPORTED" if flip_supported else "REFUTED"

            rid1 = f"{seed_prefix}-{scene_id}-neg-a-{idx:06d}"
            rid2 = f"{seed_prefix}-{scene_id}-neg-b-{idx:06d}"

            corpus.add_claim(
                self._make_claim(
                    rec_id=rid1,
                    claim=claim_true,
                    label=lab_true,
                    claim_triples=[triple],
                    evidence_triples=[triple],
                    scene_id=scene_id,
                    reasoning="negation",
                    notes="negation_pair",
                )
            )

            corpus.add_claim(
                self._make_claim(
                    rec_id=rid2,
                    claim=claim_flip,
                    label=lab_flip,
                    claim_triples=[t_flip],
                    evidence_triples=[t_flip],
                    scene_id=scene_id,
                    reasoning="negation",
                    notes="negation_pair_flipped",
                )
            )

        return corpus

    # ------------------------------------------------------------------
    # Conjunction Generation
    # ------------------------------------------------------------------

    def generate_conjunction(
        self,
        *,
        scene_id: str,
        supported_onehop: ClaimCorpus,
        triples: List[Triple],
        triple_set: Set[Triple],
        n_supported: int,
        n_refuted: int,
        seed_prefix: str,
    ) -> ClaimCorpus:

        corpus = ClaimCorpus()

        candidates = [
            r for r in supported_onehop
            if r.label == "SUPPORTED"
        ]

        if len(candidates) < 2:
            return corpus

        random.shuffle(candidates)

        # ---------- Supported ---------- #

        for idx in range(min(n_supported, len(candidates) // 2)):

            a = candidates[2 * idx]
            b = candidates[2 * idx + 1]

            claim = self._make_conjunction_text(
                a.claim.text,
                b.claim.text,
            )

            ev = [
                a.evidence.evidence_triples[0],
                b.evidence.evidence_triples[0],
            ]

            rid = f"{seed_prefix}-{scene_id}-conj-sup-{idx:06d}"

            corpus.add_claim(
                self._make_claim(
                    rec_id=rid,
                    claim=claim,
                    label="SUPPORTED",
                    claim_triples=ev,
                    evidence_triples=ev,
                    scene_id=scene_id,
                    reasoning="conjunction",
                )
            )

        # ---------- Refuted ---------- #

        for idx in range(n_refuted):

            if len(candidates) < 2:
                break

            a, b = random.sample(candidates, 2)

            a_t = a.evidence.evidence_triples[0]
            b_t = b.evidence.evidence_triples[0]

            b_bad = None

            for _ in range(10):

                trial = self.corruptor.corrupt_triple(
                    b_t,
                    triples,
                    triple_set,
                )

                if trial in triple_set or trial == b_t:
                    continue

                b_bad = trial
                break

            if b_bad is None:
                continue

            bad_claim = self.generator.verbalize(b_bad)

            if not self._is_valid_claim(bad_claim):
                continue

            claim = self._make_conjunction_text(
                a.claim.text,
                bad_claim,
            )

            ev = [a_t, b_bad]

            rid = f"{seed_prefix}-{scene_id}-conj-ref-{idx:06d}"

            corpus.add_claim(
                self._make_claim(
                    rec_id=rid,
                    claim=claim,
                    label="REFUTED",
                    claim_triples=[b_bad],
                    evidence_triples=ev,
                    scene_id=scene_id,
                    reasoning="conjunction",
                    notes="corrupted_conjunction",
                )
            )

        return corpus
