
"""Simple medical rules for prototype. Each rule is a function that accepts facts dict and
returns a float in [0,1] representing degree of match.
"""

def rule_opacity_and_infiltrate(facts):
    # example: high opacity AND presence of infiltrate -> high score
    o = facts.get('opacity', 0.0)
    i = facts.get('infiltrate', 0.0)
    return min(o, i)


def rule_diffuse_patterns(facts):
    d = facts.get('diffuse_pattern', 0.0)
    return d


def rule_airway_changes(facts):
    a = facts.get('airway_changes', 0.0)
    return a * 0.6

RULE_SET = {
    'opacity_infiltrate': (rule_opacity_and_infiltrate, 1.0),
    'diffuse_pattern': (rule_diffuse_patterns, 0.8),
    'airway_change': (rule_airway_changes, 0.6),
}
