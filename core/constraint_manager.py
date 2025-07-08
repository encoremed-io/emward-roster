from types import SimpleNamespace
from typing import Callable

class ConstraintManager:
    def __init__(self, model, state: SimpleNamespace):
        self.model = model
        self.state = state
        self.rules: list[Callable] = []

    def add_rule(self, rule_func: Callable, condition: bool = True):
        """Register a rule with optional enablement condition."""
        if condition:
            self.rules.append(rule_func)

    def apply_all(self):
        """Apply all registered rules in order."""
        for rule in self.rules:
            rule(self.model, self.state)
