"""Reference tab — EC formula, decision logic, Pramana, source trust, models, registry, assumptions."""
from __future__ import annotations

import streamlit as st

from _loaders import load_registry


def render() -> None:
    t_formula, t_registry, t_assumptions = st.tabs(
        ["Formulas & Definitions", "Source Registry", "Design Assumptions"]
    )

    with t_formula:
        _render_formulas()

    with t_registry:
        _render_registry_reference()

    with t_assumptions:
        _render_assumptions()


def _render_formulas() -> None:
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("### EC Formula")
        st.code("EC_i = 1 − (1 − ST_i)^(EW_i × IS_i)", language=None)
        st.markdown(
            "| Symbol | Meaning | Range |\n|--------|---------|-------|\n"
            "| ST | Source Trust | [0, 1] |\n"
            "| EW | Evidence Weight (stance prob.) | [0, 1] |\n"
            "| IS | Inference Strength | [0, 1] |\n"
            "| EC | Epistemic Confidence | [0, 1] |"
        )

        st.markdown("### Decision Logic  `threshold = 0.35`")
        st.markdown(
            "| Condition | Outcome |\n|-----------|---------|\n"
            "| sup > 0.35 **and** ref > 0.35 | Conflicting → VerdictHead |\n"
            "| sup > 0.35 only | Symbolic override → **SUPPORTED** |\n"
            "| ref > 0.35 only | Symbolic override → **REFUTED** |\n"
            "| neither > 0.35 | EC weak → VerdictHead |\n"
            "| baseline model | VerdictHead always |"
        )

        st.markdown("### Source Trust Scale")
        st.markdown(
            "| Source Type | Default ST |\n|-------------|------------|\n"
            "| academic | 0.95 |\n"
            "| government | 0.90 |\n"
            "| news | 0.75 |\n"
            "| unknown | 0.60 |\n"
            "| social_media | 0.40 |"
        )

        st.markdown("### Evidence Weight (EW) by Type")
        st.markdown(
            "| Evidence Type | EW |\n|---------------|----|\n"
            "| perception | 0.95 |\n"
            "| non_apprehension | 0.75 |\n"
            "| testimony | 0.80 |\n"
            "| comparison_analogy | 0.65 |\n"
            "| inference | 0.55 |\n"
            "| postulation_derivation | 0.40 |"
        )

    with c_r:
        st.markdown("### Pramana (Epistemic Modalities)")
        st.markdown(
            "| Modality | Pramana | Meaning |\n|----------|---------|----------|\n"
            "| Web Text / PDF | Shabda | Testimony |\n"
            "| Image / Video / Audio | Pratyaksha | Perception |\n"
            "| Web Table | Upamana | Comparison |"
        )

        st.markdown("### Model Architecture")
        st.markdown(
            "| Model | EC | NLI | Description |\n|-------|-----|-----|-------------|\n"
            "| baseline | ✗ | ✗ | GNN → VerdictHead only |\n"
            "| v1-hgnn  | ✓ | ✗ | Symbolic override at threshold |\n"
            "| v2-hgnn  | ✓ | ✗ | EC scalar also feeds VerdictHead |\n"
            "| v3-nli   | ✓ | ✓ | DeBERTa-v3-small NLI replaces H1 |"
        )

        st.markdown("### Inference Strength (IS)")
        st.markdown(
            "IS is a regression head output ∈ [0, 1]. It represents how strongly "
            "the evidence logically supports or refutes the claim — independent of stance direction. "
            "High IS + strong stance → high EC → likely symbolic override."
        )

        st.markdown("### Verdict Classes")
        st.markdown(
            "| Label | Symbol | Meaning |\n|-------|--------|----------|\n"
            "| supported | ✓ | Evidence confirms the claim |\n"
            "| refuted | ✗ | Evidence contradicts the claim |\n"
            "| not_enough_evidence | ~ | Insufficient or conflicting evidence |"
        )

        st.markdown("### EC Aggregation Formula")
        st.markdown(
            "For each direction d ∈ {support, refute}:\n\n"
            r"$$EC_d = 1 - \prod_{i: \text{stance}_i = d} (1 - EC_i)$$"
        )
        st.markdown(
            "If EC_support > threshold → SUPPORTED  \n"
            "If EC_refute > threshold → REFUTED  \n"
            "Else → VerdictHead classifier decides"
        )


def _render_registry_reference() -> None:
    """Compact read-only view of the source trust registry."""
    records = load_registry()
    if not records:
        st.info("Registry not found — see `data/registry/source_trust_registry.jsonl`")
        return

    import pandas as pd
    st.caption(
        f"**{len(records)} registered sources.** "
        "Full interactive view available in the **Knowledge Base** tab."
    )

    rows = [
        {
            "source_id":  r.get("source_id", ""),
            "name":       r.get("source_name", ""),
            "type":       r.get("source_type", ""),
            "modality":   r.get("modality", ""),
            "ST":         r.get("source_trust", 0.0),
            "prior_ST":   r.get("prior_trust",  0.0),
            "default_IS": r.get("default_inference_strength", 0.0),
        }
        for r in sorted(records, key=lambda x: x.get("source_trust", 0), reverse=True)
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=["ST", "prior_ST", "default_IS"],
                                     cmap="RdYlGn", vmin=0, vmax=1),
        use_container_width=True,
        height=min(600, 38 * len(rows) + 40),
    )


def _render_assumptions() -> None:
    """Key design assumptions and axioms baked into this system."""
    st.markdown("### Epistemic Assumptions")
    st.markdown(
        "These modelling choices are fixed at training / inference time. "
        "Changing any of them requires re-training or re-evaluation."
    )

    sections = [
        (
            "Source Trust (ST)",
            [
                "ST is assigned statically from the registry; it does not adapt per-claim.",
                "Unknown sources receive ST = 0.60 (conservative neutral prior).",
                "Web archives (.web.archive.org) resolve to the original domain's ST.",
                "Social-media sources (twitter, reddit, …) are capped at ST ≤ 0.50.",
            ],
        ),
        (
            "Inference Strength (IS)",
            [
                "IS is a learned regression head output ∈ [0, 1].",
                "IS is detached from the verdict gradient to prevent shortcut learning (ADR-022).",
                "IS captures logical binding strength, not stance direction.",
                "Postulation/derivation evidence is excluded from IS training (ADR-005).",
            ],
        ),
        (
            "Evidence Confidence (EC)",
            [
                "EC_i = 1 − (1 − ST_i)^(EW_i × IS_i)  — multiplicative-complementary form.",
                "EW (evidence weight) = stance probability from the stance head.",
                "The decisive threshold is fixed at 0.35 for symbolic override.",
                "EC aggregate uses product-of-complements: EC_d = 1 − ∏(1 − EC_i) for stance d.",
            ],
        ),
        (
            "Verdict Decision Logic",
            [
                "If EC_support > 0.35 AND EC_refute > 0.35 → treat as conflicting, fall through to VerdictHead.",
                "If only EC_support > 0.35 → hard SUPPORTED override.",
                "If only EC_refute  > 0.35 → hard REFUTED override.",
                "Otherwise → VerdictHead softmax output decides.",
                "Baseline model always uses VerdictHead (no EC computation).",
            ],
        ),
        (
            "Graph Construction",
            [
                "Each claim becomes a single CLAIM node; each evidence item becomes an EVIDENCE node.",
                "HeteroData edges: (EVIDENCE, supports/refutes/neutral, CLAIM).",
                "v3-nli replaces the first GNN hop (H1) with DeBERTa-v3-small NLI features (ADR-024).",
                "Boilerplate, empty, and near-duplicate evidence items are filtered before graph build.",
                "Floor-plan train/test split prevents floorplan leakage (ADR-003).",
            ],
        ),
        (
            "Synthetic Data",
            [
                "Synthetic data covers absence-of-evidence (non_apprehension) patterns (ADR-013).",
                "IS jitter is applied to synthetic data to prevent IS=0 shortcut (ADR-017).",
                "NEI-heavy distribution used: ≥40% NEI in synthetic split (ADR-018).",
                "Synthetic claims are labelled with full Pramana + EC pipeline (ADR-019).",
            ],
        ),
    ]

    for title, bullets in sections:
        with st.expander(f"**{title}**", expanded=False):
            for b in bullets:
                st.markdown(f"- {b}")
