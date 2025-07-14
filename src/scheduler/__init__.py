"""
scheduler
---------

Main scheduling module. Initializes key components:

- `builder`: Model construction and constraint setup.
- `runner`: Solving and result handling logic.

Provides high-level access to core scheduling functionality.
"""
from . import builder, runner
