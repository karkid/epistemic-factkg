from __future__ import annotations

from src.core.registry.entity import EntityRegistry
from src.core.registry.relation import RelationRegistry

from src.adapters.ai2thor.registry.entities import register_ai2thor_entities
from src.adapters.ai2thor.registry.relations import register_ai2thor_relations
from src.adapters.ai2thor.ontology import build_ai2thor_ontology

from src.nlg.entities.formatter import EntityFormatter
from src.nlg.realizers.kg_realizer import KGRealizer
from src.nlg.templates.default_templates import build_default_templates
from src.nlg.lexicon.predicate_prepositions import DEFAULT_PREDICATE_PREP

from src.adapters.ai2thor.semantics.claim_generator import AI2ThorClaimGenerator
from src.adapters.ai2thor.semantics.claim_corruptor import AI2ThorClaimCorruptor

from src.semantics.runtime.bundle import SemanticsBundle


def build_ai2thor_bundle(*, base_iri: str | None = None, seed: int | None = None) -> SemanticsBundle:
    # ---- registries (core objects) ----
    ent_reg = EntityRegistry()
    rel_reg = RelationRegistry()

    register_ai2thor_entities(ent_reg)
    register_ai2thor_relations(rel_reg)

    # ---- ontology (built from registries + namespace policy) ----
    ontology = build_ai2thor_ontology(
        relation_registry=rel_reg,
        entity_registry=ent_reg,
        base_iri=base_iri,
    )

    # ---- NLG ----
    formatter = EntityFormatter(ent_reg, definite=False)

    realizer = KGRealizer(
        templates_by_predicate=build_default_templates(),
        formatter=formatter,
        predicate_to_prep=DEFAULT_PREDICATE_PREP,
    )

    # ---- semantics strategies ----
    claim_generator = AI2ThorClaimGenerator(ontology=ontology, realizer=realizer)
    corruptor = AI2ThorClaimCorruptor(seed=seed)

    return SemanticsBundle(
        entity_registry=ent_reg,
        relation_registry=rel_reg,
        ontology=ontology,
        formatter=formatter,
        realizer=realizer,
        claim_generator=claim_generator,
        corruptor=corruptor,
    )
