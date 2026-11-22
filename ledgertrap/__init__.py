"""LedgerTrap: a reproducible RL environment for training financial-audit agents.

Public entry point: `LedgerTrapEnv` with reset(seed, config) / step(action).
"""

from .env import LedgerTrapEnv
from .version import ENV_VERSION, GENERATOR_VERSION, SCHEMA_VERSION

__all__ = ["LedgerTrapEnv", "ENV_VERSION", "GENERATOR_VERSION", "SCHEMA_VERSION"]
