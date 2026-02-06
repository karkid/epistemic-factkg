from typing import Iterator, Dict, List, Any, Tuple
from pathlib import Path
import yaml
import re

from generators.ai2thor.kg.ai2thor_ontology import AI2THOROntology
from knowledge_graph.semantics.source.base import ClaimInstance, SemanticDataSource
from utils.triple_query_engine import TripleQueryEngine

Triple = Tuple[str, str, str]


class AI2THORSemanticDataSource(SemanticDataSource):
    """
    Semantic data source for AI2-THOR generated claims.
    Loads claim instances from YAML files in a specified directory.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to project configs folder
            config_path = (
                Path(__file__).parent.parent.parent.parent.parent
                / "configs"
                / "ai2thor_default.yaml"
            )

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.claim_corpus = self._load_claims()

    def _load_config(self) -> Dict[str, Any]:
        """Load AI2-THOR configuration from YAML."""
        if not self.config_path.exists():
            # Return default config if file doesn't exist
            default_config = {
                "claim_files": [
                    "/Users/Frido/Workspace/Projects/github/epistemic-factkg/output/knowledge_graph.ttl"
                ]
            }
            print(f"Config file not found at {self.config_path}, using default config")
            return default_config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"Loaded config from {self.config_path}")
            return config

    def _load_claims(self) -> List[ClaimInstance]:
        """Load claim instances from configured YAML files."""
        claims: List[ClaimInstance] = []
        query_engine = TripleQueryEngine()

        # Load TTL files
        # claim_files = self.config.get("claim_files", [])
        ##print(f"Looking for claim_files in config: {claim_files}")
        claim_files = [
            "/Users/Frido/Workspace/Projects/github/epistemic-factkg/output/knowledge_graph.ttl"
        ]

        for ttl_file in claim_files:
            try:
                query_engine.load_from_ttl(ttl_file)
            except Exception as e:
                print(f"Warning: Could not load TTL file {ttl_file}: {e}")
                continue

        FLOORPLAN_RE = re.compile(r"FloorPlan\d+", re.IGNORECASE)
        floor_plans_group = query_engine.group_by_entity(
            entity_pattern=FLOORPLAN_RE, forward_rel="hasObject", backward_rel="inScene"
        )
        floor_plans = sorted(floor_plans_group.keys())

        print(floor_plans)

        # for fp in floor_plans:
        #     triples = floor_plans_group[fp]

        return claims

    def get_claims(self) -> Iterator[ClaimInstance]:
        """Yield claim data one by one."""
        claim_ids = self.get_available_claims()
        for claim_id in claim_ids:
            yield self.get_claim_by_id(claim_id)

    def get_claim_by_id(self, claim_id: str) -> ClaimInstance:
        """Get specific claim by ID."""
        for ci in self.claim_corpus:
            if ci.rec_id == claim_id:
                return ci
        return None

    def get_available_claims(self) -> List[str]:
        """List all available claim IDs."""
        return [ci.rec_id for ci in self.claim_corpus]

    def cleanup(self) -> None:
        """Clean up any resources held by the data source."""
        self.claim_corpus.clear()
