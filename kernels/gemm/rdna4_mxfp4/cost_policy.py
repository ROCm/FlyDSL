# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU configuration cost checks; this module never launches a GPU."""

import json
from functools import lru_cache
from pathlib import Path

from .cost_candidates import bounded_candidates, expanded_candidates
from .cost_features import COST_FEATURE_NAMES, configuration_features, predict_cost
from .dispatch_features import CONTEXT_FEATURE_NAMES, context_features


@lru_cache(maxsize=1)
def cost_model():
    model = json.loads(Path(__file__).with_name("cost_model.json").read_text())
    if tuple(model["feature_names"]) != (*CONTEXT_FEATURE_NAMES, *COST_FEATURE_NAMES):
        raise ValueError("cost model and CPU feature definitions do not match")
    return model


def rerank_configs(
    m, n, k, cache_policy, regional, adapt, legal, register_defaults, hybrid_defaults,
    *, improvement_margin=0.0,
):
    """Keep the regional first choice unless the cost model finds a clear loss."""
    originals = list(regional(cache_policy))
    default = originals[0]
    model = cost_model()
    hints = (0, 1) if cache_policy == "auto" else (int(cache_policy == "hot"),)
    contexts = [context_features(m, n, k, hint) for hint in hints]
    scores = {}

    def score(config):
        key = tuple(sorted(config.items()))
        if key not in scores:
            extra = configuration_features(m, n, k, config)
            scores[key] = sum(
                predict_cost(model, (*values, *extra)) for values in contexts
            ) / len(contexts)
        return scores[key]

    default_score = score(default)
    if default_score <= model["gate"]["floor"]:
        return originals
    bank = model.get("candidate_bank", "bounded")
    if bank not in ("bounded", "expanded"):
        raise ValueError("unsupported cost-model candidate bank")
    factory = expanded_candidates if bank == "expanded" else bounded_candidates
    candidates = factory(
        m, n, k, regional, adapt, legal, register_defaults, hybrid_defaults
    )
    winner = min(candidates, key=score)
    if default_score - score(winner) <= max(model["gate"]["margin"], improvement_margin):
        return originals
    return [winner, *(config for config in originals if config != winner)]
