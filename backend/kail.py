"""
Knowledge-Adaptive Integration Layer (KAIL) - maps concept confidences to symbolic facts
and computes weighted rule activations.
"""
import math

class KAIL:
    def __init__(self, rule_weights=None):
        # rule_weights: dict rule_name -> weight (float)
        self.rule_weights = rule_weights or {}

    def concept_to_fact(self, concept_vector, concept_names):
        # concept_vector: list/np of floats in [0,1]
        facts = {}
        for i, name in enumerate(concept_names):
            conf = float(concept_vector[i])
            # thresholding with soft confidence
            facts[name] = conf
        return facts

    def rule_activation(self, facts, rule):
        # rule is a lambda/fn that uses facts and returns activation score [0,1]
        try:
            score = rule(facts)
        except Exception:
            score = 0.0
      return max(0.0, min(1.0, score))

    def weighted_reasoning(self, facts, rules):
        # rules: dict name -> (callable rule, base_weight)
        activations = {}
        for rname, (rfn, base_w) in rules.items():
            act = self.rule_activation(facts, rfn)
            w = self.rule_weights.get(rname, base_w)
            activations[rname] = {'activation': act, 'weight': w, 'score': act * w}
        return activations

    def aggregate_decision(self, activations):
        # simple weighted sum normalized into [0,1]
        num = sum(v['score'] for v in activations.values())
        den = sum(v['weight'] for v in activations.values()) or 1.0
        return num / den
