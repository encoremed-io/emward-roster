"""
src.core
---------

Core scheduling engine components:

- HardRule & define_hard_rules:  
  Define and instantiate non-negotiable ("hard") constraints for the CP-SAT model.

- ConstraintManager:  
  Register and apply constraint functions in a controlled sequence.

- ScheduleState:  
  Encapsulate all inputs, parameters, and intermediate collections needed to build
  and solve the nurse scheduling problem.
"""
