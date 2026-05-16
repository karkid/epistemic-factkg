"""Model architecture components (never trained separately).

V1 architecture: HeteroConv encoder + task heads + symbolic aggregator.

Modules:
- encoder.py    : Shared HeteroConv built from GraphConfig
- heads.py      : StanceHead (H1), ISHead (H2), VerdictHead
- aggregator.py : Stateless EC formula (no parameters)
"""
