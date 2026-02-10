import random
import time
from urllib.parse import unquote

from typing import Any, Dict, List, Optional, Sequence, Callable

from src.core.claims.lables import OutputLabels, ReasoningLabels, SourceTypesLabels
from src.core.claims.types import ClaimCorpus, ClaimInstance
from src.core.graph.types import Term, Triple, TripleList
from src.core.nlg.triple_realizer import TripleRealizer
from src.core.claims.result import ClaimGenerationStats, ClaimGenerationStatsSummary


class ClaimGenerator:

    def __init__(self,
                  *, 
                  realizer: TripleRealizer, 
                  context_id: str,
                  triples: TripleList,
                  source: str = "agent",
                  generator: str = "agent",
                  source_type: str = SourceTypesLabels.UNKNOWN,
                  context_type: Optional[str] = None,
                  seed: Optional[int] = 42,
                  seed_prefix: Optional[str] = "claim",
                  receptacle_mapper: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.realizer = realizer
        self.context_id = context_id
        self.source = source
        self.generator = generator
        self.source_type = source_type
        self.context_type = context_type
        self.triples = triples
        self.seed_prefix = seed_prefix
        self._rng = random.Random(seed)
        self.stats = ClaimGenerationStats()
        self.corpus = ClaimCorpus()
        self.stats.corpus = self.corpus
        self.receptacle_mapper = receptacle_mapper
        self._start_time: Optional[float] = None

    def _random_triple(self, triples: Sequence[Triple]) -> Triple:
        return self._rng.choice(list(triples))
    
    def _random_triple_not_in(self, triples: Sequence[Triple], exclude_triples: Sequence[Triple]) -> Optional[Triple]:
        exclude_set = set(exclude_triples)
        candidates = [t for t in triples if t not in exclude_set]
        if not candidates:
            return None
        return self._rng.choice(candidates)
    
    def _is_valid_text(self, text: str) -> bool:
        return bool(text and text.strip())
    
    def _is_boolean_object(self, o: Term) -> bool:
        object_type = self._object_type_from_entity_id(str(o))
        if isinstance(object_type, str):
            ol = object_type.strip().lower()
            return ol in {"true", "false", "yes", "no"}
        return False
    
    def _is_temperature_object(self, o: Term) -> bool:
        """Check if object is a temperature value that can be corrupted."""
        object_type = self._object_type_from_entity_id(str(o))
        if isinstance(object_type, str):
            ol = object_type.strip()
            return ol in {"Cold", "Hot", "RoomTemp"}
        return False
    
    def _corrupt_temperature(self, o: Term) -> str:
        """Corrupt temperature value to a different temperature."""
        object_type = self._object_type_from_entity_id(str(o))
        ol = object_type.strip()
        # Define temperature corruption mappings
        temperature_alternatives = {
            "Cold": ["Hot", "RoomTemp"],
            "Hot": ["Cold", "RoomTemp"], 
            "RoomTemp": ["Cold", "Hot"]
        }
        
        if ol in temperature_alternatives:
            alternatives = temperature_alternatives[ol]
            return self._rng.choice(alternatives)
        return ol  # Return original if not a known temperature
    
    def _flip_bool(self, o: Term) -> str:
        object_type = self._object_type_from_entity_id(str(o))
        ol = object_type.strip().lower()
        if ol in {"true", "yes"}:
            return "False"
        if ol in {"false", "no"}:
            return "True"
        return "True"
    
    def _short_uri(self, x: str) -> str:
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


    def _object_type_from_entity_id(self, entity_id: str) -> str:
        """
        Fork|+00.62|+01.31|-02.48 -> Fork
        Fork_3 -> Fork
        """
        t = self._short_uri(entity_id)
        if "|" in t:
            return t.split("|", 1)[0]
        if "_" in t:
            return t.split("_", 1)[0]
        return t
    
    def _find_triples_with_object(self, triples: TripleList, obj: Term) -> List[Triple]:
        """Find triples where the object type matches the given object type, returns original triples."""
        target_obj_type = self._object_type_from_entity_id(str(obj))
        return [t for t in triples if self._object_type_from_entity_id(str(t.o)) == target_obj_type]
    
    def _object_present_in_triples(self, triples: TripleList, obj: Term) -> bool:
        """Check if an object type is present in triples using object type comparison."""
        target_obj_type = self._object_type_from_entity_id(str(obj))
        for t in triples:
            if self._object_type_from_entity_id(str(t.o)) == target_obj_type:
                return True
        return False
    
    def _corrupt_triple(self, triple: Triple) -> Triple:
        s, p, o = triple
        copy_triples = self.triples.copy()
        copy_triples.remove(triple)
        
        # Track object type being processed
        object_type = self._object_type_from_entity_id(str(o))
        self.stats.add_object_type(object_type)

        # Try boolean corruption first
        if self._is_boolean_object(o):
            bad_triple = Triple(s=s, p=p, o=self._flip_bool(o))
            if bad_triple not in self.triples:
                self.stats.add_corruption_type("boolean_flip")
                self.stats.successful_corruptions += 1
                return bad_triple
        
        # Try temperature corruption
        if self._is_temperature_object(o):
            corrupted_temp = self._corrupt_temperature(o)
            if corrupted_temp != str(o):  # Ensure we actually changed the temperature
                bad_triple = Triple(s=s, p=p, o=corrupted_temp)
                if bad_triple not in self.triples:
                    self.stats.add_corruption_type("temperature_change")
                    self.stats.successful_corruptions += 1
                    return bad_triple
        
        # Try receptacle-based corruption
        if not self.receptacle_mapper:
            self.stats.failed_corruptions += 1
            return Triple(s=s, p=p, o=o)  # No corruption possible
            
        preferred_receptacles = self.receptacle_mapper(self._object_type_from_entity_id(str(s)))
        #print(f"Attempting receptacle-based corruption for object '{o}' of type '{object_type}'. Preferred receptacles: {preferred_receptacles}")
        if not preferred_receptacles:
            self.stats.failed_corruptions += 1
            return Triple(s=s, p=p, o=o)  # No alternatives found
            
        self._rng.shuffle(preferred_receptacles)
        for rec in preferred_receptacles:
            if not self._object_present_in_triples(copy_triples, rec):
                continue
                
            # if preferred_receptacles:
            #     print(f"{preferred_receptacles}")    
            # Track receptacle mapping usage
            self.stats.add_receptacle_mapping(rec)
            
            find_triples = self._find_triples_with_object(copy_triples, rec)
            self._rng.shuffle(find_triples)
            
            for t in find_triples:
                bad_triple = Triple(s=s, p=p, o=t.o)
                if bad_triple not in self.triples:
                    self.stats.add_corruption_type("receptacle_substitution")
                    #print(f"Corrupting triple {triple} by substituting object with {t.o} from receptacle {rec}")
                    self.stats.successful_corruptions += 1
                    return bad_triple
        
        self.stats.failed_corruptions += 1                
        return Triple(s=s, p=p, o=o)  # Return original if no corruption worked


    def _filter_claims_by_reasoning(self, reasoning: ReasoningLabels, includes_labels: List[OutputLabels]) -> ClaimCorpus:
        grouped = ClaimCorpus()
        for claim in self.corpus.claims:
            if claim.reasoning.structural == reasoning and claim.label in includes_labels:
                grouped.add(claim)
        return grouped
    
    def _is_new_claim(self, claim_instance: ClaimInstance) -> bool:
        claim_layout = claim_instance.get_schema_layout_json()
        for ci in self.corpus.claims:
            if ci.get_schema_layout_json() == claim_layout:
                self.stats.duplicate_claims_filtered += 1
                return False
        return True

    def _make_instance(self,
                       *,
                        rec_id: str,
                        claim_text: str,
                        label: str,
                        claim_triples: Sequence[Triple],
                        structural_reasoning: str,
                        evidence_triples: Sequence[Triple],
                        evidence_urls: Optional[List[str]] = None,
                        split: Optional[str] = None,
                        notes: Optional[str] = None,
                        created_utc: Optional[str] = None,
                        source_type: Optional[SourceTypesLabels] = None
                       ):
        return ClaimInstance.make_instance(
            rec_id=rec_id,
            claim_text=claim_text,
            label=label,
            claim_triples=claim_triples,
            structural_reasoning=structural_reasoning,
            evidence_triples=evidence_triples,
            evidence_source=self.source,
            evidence_source_type= source_type if source_type is not None else self.source_type,
            evidence_urls=evidence_urls,
            context_id=self.context_id,
            generator=self.generator,
            context_type=self.context_type,
            split=split,
            notes=notes,
            created_utc=created_utc,
        )

    def save_to_jsonl(self, file_path: str) -> None:
        self.corpus.save_to_jsonl(file_path)

    def remove_duplicates(self) -> None:
        self.corpus.remove_duplicates()
    
    def start_timing(self) -> None:
        """Start timing the claim generation process."""
        self._start_time = time.time()
    
    def finalize_stats(self) -> None:
        """Finalize statistics and calculate performance metrics."""
        if self._start_time:
            self.stats.processing_time_seconds = time.time() - self._start_time
        
        self.stats.total_processed_triples = len(self.triples)
        self.stats.finalize_timing()
    
    def print_generation_summary(self) -> None:
        """Print comprehensive claim generation summary."""
        self.finalize_stats()
        summary = ClaimGenerationStatsSummary(stats=self.stats)
        summary.print_summary()

    def generate_one_hop(
        self,
        *,
        n_claims: Optional[int] = 10,
        add_corruption: bool = False,
        n_supported: Optional[float] = 0.5, # of n_claims
        max_sup_attempts: Optional[int] = 10,
        max_refuted_attempts: Optional[int] = 15,
    ) -> None:
        """
        Generate claims based on one-hop triples from the context. Optionally add corrupted claims by modifying the object of the triple.
        """
        if not self._start_time:
            self.start_timing()

        # ---------------- SUPPORTED ----------------
        supported_target = int(n_supported * n_claims) if add_corruption else n_claims
        supported_attempts_budget = supported_target * max_sup_attempts

        supported_count = 0
        attempts = 0

        while supported_count < supported_target and attempts < supported_attempts_budget:

            truth_triple = self._random_triple(self.triples)

            text = self.realizer.realize(truth_triple)

            if not self._is_valid_text(text):
                self.stats.skipped_empty += 1
                attempts += 1
                continue

            record_id = f"{self.seed_prefix}-{self.context_id}-onehop-sup-{len(self.corpus.claims):06d}"

            instance = self._make_instance(
                        rec_id=record_id,
                        claim_text=text,
                        label=OutputLabels.SUPPORTED,
                        claim_triples=[truth_triple],
                        structural_reasoning=ReasoningLabels.ONE_HOP,
                        evidence_triples=[truth_triple],
                        evidence_urls=[],
                        split= None,
                        notes = "recorded",
            )

            if not self._is_new_claim(instance):
                attempts += 1
                continue

            self.corpus.add(instance)
            self.stats.supported += 1
            supported_count += 1
            attempts += 1
        
        if not add_corruption:
            return
        
        # ---------------- REFUTED ----------------      
        refuted_target = int((1-n_supported) * n_claims)
        refuted_attempts_budget = refuted_target * max_refuted_attempts

        refuted_count = 0
        attempts = 0

        while refuted_count < refuted_target and attempts < refuted_attempts_budget:

            truth_triple = self._random_triple(self.triples)
            claim_triple = self._corrupt_triple(truth_triple)  # Attempt to corrupt the triple, but it may return the original if no corruption is possible

            if claim_triple == truth_triple:
                attempts += 1
                continue  # No valid corruption found, skip
            text = self.realizer.realize(claim_triple)

            if not self._is_valid_text(text):
                self.stats.skipped_empty += 1
                attempts += 1
                continue
            
            record_id = f"{self.seed_prefix}-{self.context_id}-onehop-ref-{len(self.corpus.claims):06d}"

            instance = self._make_instance(
                        rec_id=record_id,
                        claim_text=text,
                        label=OutputLabels.REFUTED,
                        claim_triples=[claim_triple],
                        structural_reasoning=ReasoningLabels.ONE_HOP,
                        evidence_triples=[truth_triple],
                        evidence_urls=[],
                        split= None,
                        notes = "corrupted"
            )
            self.corpus.add(instance)

            self.stats.refuted += 1
            refuted_count += 1
            attempts += 1

    def generate_conjunction(
        self,
        *,
        n_claims: Optional[int] = 10,
        add_corruption: bool = False,
        n_supported: Optional[float] = 0.5, # of n_claims
        max_sup_attempts: Optional[int] = 10,
        max_refuted_attempts: Optional[int] = 15,
    ) -> None:
        if not self._start_time:
            self.start_timing()

        copied_corpus = ClaimCorpus(claims=self.corpus.claims.copy())
        self._rng.shuffle(copied_corpus.claims)

        # ---------------- SUPPORTED ----------------
        supported_target = int(n_supported * n_claims) if add_corruption else n_claims
        supported_attempts_budget = supported_target * max_sup_attempts

        supported_count = 0
        attempts = 0

        while supported_count < supported_target and attempts < supported_attempts_budget:

            truth_triple_1 = self._random_triple(self.triples)
            truth_triple_2 = self._random_triple_not_in(self.triples, exclude_triples=[truth_triple_1])

            text = self.realizer.realize_conjunction(truth_triple_1, truth_triple_2)

            if not self._is_valid_text(text):
                self.stats.skipped_empty += 1
                attempts += 1
                continue

            record_id = f"{self.seed_prefix}-{self.context_id}-conjunction-sup-{len(self.corpus.claims):06d}"

            instance = self._make_instance(
                        rec_id=record_id,
                        claim_text=text,
                        label=OutputLabels.SUPPORTED,
                        claim_triples=[truth_triple_1, truth_triple_2],
                        structural_reasoning=ReasoningLabels.CONJUNCTION,
                        evidence_triples=[truth_triple_1, truth_triple_2],
                        evidence_urls=[],
                        split= None,
                        notes = "recorded",
                        source_type=SourceTypesLabels.INFERENCE
            )

            if not self._is_new_claim(instance):
                attempts += 1
                continue

            self.corpus.add(instance)
            self.stats.supported += 1
            supported_count += 1
            attempts += 1
        
        if not add_corruption:
            return
        
        # ---------------- REFUTED ----------------
        refuted_target = int(n_supported * n_claims) if add_corruption else n_claims
        refuted_attempts_budget = refuted_target * max_refuted_attempts

        refuted_count = 0
        attempts = 0
        refuted_mode = [0, 1, 2]  # 0: corrupt triple 1, 1: corrupt triple 2, 2: corrupt both

        while refuted_count < refuted_target and attempts < refuted_attempts_budget:
            
            truth_triple_1 = self._random_triple(self.triples)
            truth_triple_2 = self._random_triple_not_in(self.triples, exclude_triples=[truth_triple_1])

            random_mode = self._rng.choice(refuted_mode)
            if random_mode == 0:
                claim_triple_1 = self._corrupt_triple(truth_triple_1)
                claim_triple_2 = truth_triple_2
            elif random_mode == 1:
                claim_triple_1 = truth_triple_1
                claim_triple_2 = self._corrupt_triple(truth_triple_2)
            else:
                claim_triple_1 = self._corrupt_triple(truth_triple_1)
                claim_triple_2 = self._corrupt_triple(truth_triple_2)

            # Check if corruption actually worked - at least one triple should be different
            corruption_worked = False
            if random_mode == 0:
                corruption_worked = claim_triple_1 != truth_triple_1
            elif random_mode == 1:
                corruption_worked = claim_triple_2 != truth_triple_2
            else:
                corruption_worked = (claim_triple_1 != truth_triple_1) or (claim_triple_2 != truth_triple_2)
            
            if not corruption_worked:
                attempts += 1
                continue  # No valid corruption found, skip

            text = self.realizer.realize_conjunction(claim_triple_1, claim_triple_2)

            if not self._is_valid_text(text):
                self.stats.skipped_empty += 1
                attempts += 1
                continue

            record_id = f"{self.seed_prefix}-{self.context_id}-conjunction-ref-{len(self.corpus.claims):06d}"

            instance = self._make_instance(
                        rec_id=record_id,
                        claim_text=text,
                        label=OutputLabels.REFUTED,
                        claim_triples=[claim_triple_1, claim_triple_2],
                        structural_reasoning=ReasoningLabels.CONJUNCTION,
                        evidence_triples=[truth_triple_1, truth_triple_2],
                        evidence_urls=[],
                        split= None,
                        notes = "corrupted",
                        source_type=SourceTypesLabels.INFERENCE
            )

            if not self._is_new_claim(instance):
                attempts += 1
                continue

            self.corpus.add(instance)
            self.stats.refuted += 1
            refuted_count += 1
            attempts += 1

    def generate_negation(
        self,
        *,
        n_claims: Optional[int] = 10,
        add_corruption: bool = False,
        n_supported: Optional[float] = 0.5, # of n_claims
        max_sup_attempts: Optional[int] = 10,
        max_refuted_attempts: Optional[int] = 15,
    ) -> None:
        if not self._start_time:
            self.start_timing()
        
        boolean_triples = [t for t in self.triples if self._is_boolean_object(t.o) ]

        # ---------------- SUPPORTED ----------------
        supported_target = int(n_supported * n_claims) if add_corruption else n_claims
        supported_attempts_budget = supported_target * max_sup_attempts

        supported_count = 0
        attempts = 0

        while supported_count < supported_target and attempts < supported_attempts_budget:
            
            truth_triple = self._random_triple(boolean_triples)
            text = self.realizer.realize_negation(truth_triple)

            if not self._is_valid_text(text):
                self.stats.skipped_empty += 1
                attempts += 1
                continue

            record_id = f"{self.seed_prefix}-{self.context_id}-negation-sup-{len(self.corpus.claims):06d}"

            instance = self._make_instance(
                        rec_id=record_id,
                        claim_text=text,
                        label=OutputLabels.SUPPORTED,
                        claim_triples=[truth_triple],
                        structural_reasoning=ReasoningLabels.NEGATION,
                        evidence_triples=[truth_triple],
                        evidence_urls=[],
                        split= None,
                        notes = "recorded",
            )

            if not self._is_new_claim(instance):
                attempts += 1
                continue

            self.corpus.add(instance)
            self.stats.supported += 1
            supported_count += 1
            attempts += 1

        if not add_corruption:
            return
        
        # ---------------- REFUTED ----------------
        refuted_target = int(n_supported * n_claims) if add_corruption else n_claims
        refuted_attempts_budget = refuted_target * max_refuted_attempts

        refuted_count = 0
        attempts = 0
        while refuted_count < refuted_target and attempts < refuted_attempts_budget:
            # Glass is breakable.
            # Negate Glass is not breakable => refuted claim with evidence Glass is breakable
            truth_triple = self._random_triple(boolean_triples)
            claim_triple = Triple(s=truth_triple.s, p=truth_triple.p, o=self._flip_bool(truth_triple.o))
            text = self.realizer.realize_negation(claim_triple)

            if not self._is_valid_text(text):
                self.stats.skipped_empty += 1
                attempts += 1
                continue

            record_id = f"{self.seed_prefix}-{self.context_id}-negation-ref-{len(self.corpus.claims):06d}"

            instance = self._make_instance(
                        rec_id=record_id,
                        claim_text=text,
                        label=OutputLabels.REFUTED,
                        claim_triples=[claim_triple],
                        structural_reasoning=ReasoningLabels.NEGATION,
                        evidence_triples=[truth_triple],
                        evidence_urls=[],
                        split= None,
                        notes = "corrupted",
            )

            if not self._is_new_claim(instance):
                attempts += 1
                continue

            self.corpus.add(instance)
            self.stats.refuted += 1
            refuted_count += 1
            attempts += 1
            