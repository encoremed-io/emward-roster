"""
scheduler.rules
---------------

Exposes all scheduling constraints by importing from:

- `fixed`: Rules for handling fixed assignments, leave days and training shifts for the nurse scheduling problem.
- `high`: High priority constraints (e.g., AM coverage, preferences).
- `low`: Low priority constraints (e.g., fairness, weekly hours).

Allows unified access to all rule and constraint definitions via wildcard imports.
"""
from .fixed import *
from .high import *
from .low import *
