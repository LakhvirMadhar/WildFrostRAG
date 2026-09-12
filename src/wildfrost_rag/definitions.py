"""Top-level Dagster Definitions object, wiring together defs/'s assets/resources.

Launch locally with:
    poetry run dagster dev -m wildfrost_rag.definitions
"""

from dagster import Definitions

defs = Definitions(assets=[], resources={})
