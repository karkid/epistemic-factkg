
from dataclasses import dataclass
from typing import Optional
import re

from src.adapters.ai2thor.semantics.semantic_rules import get_preferred_receptacles
from src.core.claims.labels import SourceTypesLabels
from src.core.claims.claim_generator import ClaimGenerator
from src.core.claims.types import ClaimCorpus
from src.adapters.ai2thor.semantics.entity_lexicon import create_ai2thor_object_type_lexicon
from src.adapters.ai2thor.semantics.predicate_lexicon import create_predicate_lexicon
from src.adapters.ai2thor.nlg.template import Ai2ThorTemplate

from src.core.nlg.triple_realizer import TripleRealizer

from src.core.graph.types import Triple, TripleList, TripleSet

from src.infra.rdf.io.ttl import load_triples_from_ttl
from src.infra.rdf.query.engine import TripleQueryEngine
from src.infra.rdf.formatter import short_uri, ai2thor_object_type_from_entity_id

from src.pipelines.result import BuildClaimsResult


def build_claims(
    ttl_path: str = "out/knowledge_graph.ttl",
    output_dir: str = "out",
    max_contexts: Optional[int] = None,
    n_claims: int = 500,
    add_corruption: bool = True,
    verbose: bool = False
) -> BuildClaimsResult:
    """
    Build semantic claims from a knowledge graph TTL file.
    
    Args:
        ttl_path: Path to the TTL file containing the knowledge graph
        output_dir: Directory to save claim JSONL files
        max_contexts: Maximum number of contexts to process (None for all)
        n_claims: Number of claims to generate per method
        add_corruption: Whether to add corrupted claims
        verbose: Enable verbose logging
    
    Returns:
        BuildClaimsResult with processing statistics and output files
    """
    if verbose:
        print(f"Loading triples from: {ttl_path}")
    
    try:
        # 1. Load triples from TTL file
        triples = load_triples_from_ttl(ttl_path)
        query_engine = TripleQueryEngine(triples)
        
        if verbose:
            print(f"Loaded {len(triples)} triples")

        # 2. Group triples by context/floorplan id
        group_by_context = query_engine.group_by_namespace(namespace="contexts")
        contexts = sorted(group_by_context.keys())
        
        if max_contexts and max_contexts > 0:
            contexts = contexts[:max_contexts]
            
        if verbose:
            print(f"Processing {len(contexts)} contexts: {contexts}")

        # 3. Initialize Claim generator and corruptor (if needed)
        pred_lex = create_predicate_lexicon()
        ent_lex = create_ai2thor_object_type_lexicon()
        realizer = TripleRealizer(
            template=Ai2ThorTemplate(predicate_lexicon=pred_lex), 
            pred_lexicon=pred_lex, 
            ent_lexicon=ent_lex, 
            normallizer=ai2thor_object_type_from_entity_id
        )

        # 4. Generate claims for each context and collect all claims
        output_files = []
        total_corpus = ClaimCorpus()
        
        for context in contexts:
            if verbose:
                print(f"Processing context: {context}")
                
            context_triples = group_by_context[context]
            claim_generator = ClaimGenerator(
                realizer=realizer, 
                context_id=context, 
                triples=context_triples, 
                context_type="floorplan", 
                generator="ai2thor-scene-simulator",
                source="sensor",
                source_type=SourceTypesLabels.PERCEPTION,
                receptacle_mapper=get_preferred_receptacles
            )
            
            # Generate different types of claims
            claim_generator.generate_one_hop(n_claims=n_claims, add_corruption=add_corruption)
            claim_generator.generate_conjunction(n_claims=n_claims, add_corruption=add_corruption)
            # Note: generate_negation might not exist yet, commenting out for safety
            claim_generator.generate_negation(n_claims=n_claims, add_corruption=add_corruption)

            # Add to total corpus (don't save individual context files)
            total_corpus.claims.extend(claim_generator.corpus.claims)
            
            if verbose:
                print(f"Generated {len(claim_generator.corpus.claims)} claims for {context}")

        # Save all claims to a single consolidated file
        consolidated_output_file = f"{output_dir}/claims_all.jsonl"
        total_corpus.save_to_jsonl(consolidated_output_file)
        output_files.append(consolidated_output_file)

        if verbose:
            print(f"Total claims generated: {len(total_corpus.claims)}")
            print(f"All claims saved to: {consolidated_output_file}")
            print(f"Output files: {output_files}")

        return BuildClaimsResult(
            ttl_path=ttl_path,
            total_triples=len(triples),
            contexts=contexts,
            context_triples=group_by_context,
            claim_corpus=total_corpus,
            output_files=output_files
        )

    except Exception as e:
        raise RuntimeError(f"Error building claims: {e}") from e