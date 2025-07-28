class NoFeasibleSolutionError(Exception):
    """Raised when no feasible solution is found with the given constraints, before preferences and fairness are considered."""
    pass

class InvalidMCError(Exception):
    """Raised when an invalid MC (Medical Certificate) is encountered in the preferences."""
    pass

class InvalidALError(Exception):
    """Raised when an invalid AL (Annual Leave) is encountered in the preferences."""
    pass

class ConsecutiveMCError(Exception):
    """Raised when the maximum consecutive MC days is exceeded."""
    pass

class ConsecutiveALError(Exception):
    """Raised when the maximum consecutive AL days is exceeded."""
    pass

class InputMismatchError(Exception):
    """Raised when there is a mismatch between the nurse profiles and preferences."""
    pass

class InvalidPreviousScheduleError(Exception):
    """Raised when the previous schedule is invalid."""

class FileReadingError(Exception):
    """Raised when there is an error reading a file."""
    pass

class FileContentError(Exception):
    """Raised when the content of a file is not as expected."""
    pass
