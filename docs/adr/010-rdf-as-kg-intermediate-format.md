# ADR-010: RDF/Turtle as Knowledge Graph Intermediate Format

## Status

Accepted

## Context

The AI2-THOR pipeline needs an intermediate representation of the scene knowledge graph — a structured store of entities, properties, and relations extracted from the simulation — before claim generation can begin. This intermediate format must support:

1. Structured storage of typed triples (`entity`, `predicate`, `value`)
2. Efficient querying to generate claims of different structural types (one-hop, conjunction, negation, absence)
3. Human readability for debugging and inspection
4. Extensibility for future ontology additions

Alternatives considered:

| Format | Description | Problem |
|---|---|---|
| **Raw JSONL triples** | One JSON object per triple | No standard query language; custom traversal code required for each claim type |
| **Property graph (Neo4j)** | Graph database with Cypher queries | Heavyweight dependency; requires running a server; overkill for a dataset generation step |
| **JSON-LD** | JSON with semantic web context | More complex to parse than plain Turtle; less human-readable |
| **RDF/Turtle (TTL)** | Standard W3C semantic web format with SPARQL queries | Mature tooling, `rdflib` in Python, standard query language, human-readable |

## Decision

Use **RDF/Turtle** (`.ttl` files) as the knowledge graph intermediate format, with `rdflib` for graph construction and `SPARQLWrapper`/`rdflib` for queries.

Output: `out/knowledge_graph.ttl`

Custom RDF namespaces defined in `src/infra/rdf/namespaces.py`:
- `efkg:` — Epistemic FactKG ontology terms
- `efkg-obj:` — AI2-THOR object URIs
- Standard namespaces: `rdf:`, `rdfs:`, `owl:`, `xsd:`

The RDF builder is in `src/infra/rdf/builder.py`. SPARQL query engine is in `src/infra/rdf/query/engine.py`.

## Consequences

**Positive:**
- SPARQL provides expressive, declarative queries for all claim structural types (`one_hop`, `conjunction`, `negation`, `absence`) without custom traversal code
- `rdflib` is a pure Python library with no server dependency — fits cleanly into the pip/uv dependency stack
- TTL files are human-readable and inspectable in standard semantic web tools (Protégé, GraphDB browser)
- The ontology is extensible: adding new AI2-THOR predicates or object types means updating the ontology files, not the query engine
- Interoperability — the TTL output could be ingested by any semantic web consumer without format conversion

**Negative:**
- TTL files are verbose — the graph for a full AI2-THOR run can be large
- SPARQL has a learning curve; contributors unfamiliar with RDF/SPARQL may find the query code harder to read than custom Python traversal
- `rdflib` adds a dependency; performance for very large graphs may require migration to a dedicated triplestore
