"""Domain knowledge used to validate what the models learned.

These are not model outputs. They are expectations taken from medicinal
chemistry, written down *before* looking at SHAP results, so that the check in
``scripts/explain.py`` is a falsifiable assertion rather than a story told
afterwards.

Blood-brain barrier penetration is governed by a polarity/lipophilicity
trade-off: to cross, a molecule should be small, not very polar, and not carry
many hydrogen-bond donors. Hence the expected signs below.
"""

from __future__ import annotations

# descriptor -> ("positive" | "negative", human-readable rationale)
BBB_EXPECTED_DIRECTION: dict[str, tuple[str, str]] = {
    "TPSA": (
        "negative",
        "Polar surface area above ~90 Å² blocks passive diffusion into the CNS.",
    ),
    "MolWt": (
        "negative",
        "Larger molecules cross less readily; CNS drugs cluster below ~450 Da.",
    ),
    "NumHDonors": (
        "negative",
        "Each hydrogen-bond donor costs desolvation energy at the membrane.",
    ),
    "NHOHCount": (
        "negative",
        "NH and OH groups are donors, so more of them lowers penetration.",
    ),
    "MolLogP": (
        "positive",
        "Moderate lipophilicity helps a molecule partition into the membrane.",
    ),
}

# Lipinski rule-of-5 thresholds, used by the demo app.
LIPINSKI_RULES = [
    ("MolWt", 500, "molecular weight > 500"),
    ("MolLogP", 5, "LogP > 5"),
    ("NumHDonors", 5, "H-bond donors > 5"),
    ("NumHAcceptors", 10, "H-bond acceptors > 10"),
]


def check_direction(descriptor: str, correlation: float) -> tuple[bool, str] | None:
    """Compare an observed value/SHAP correlation with the expectation.

    Returns ``(matches, rationale)``, or None when the descriptor has no
    documented expectation.
    """
    if descriptor not in BBB_EXPECTED_DIRECTION:
        return None
    expected, rationale = BBB_EXPECTED_DIRECTION[descriptor]
    observed = "positive" if correlation > 0 else "negative"
    return observed == expected, rationale
