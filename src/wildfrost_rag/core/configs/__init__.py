"""One settings class per external system.

Neo4j, OpenAI, MLflow, embeddings, scraping, and paths - each loading its
own env vars independently. core/config.py is the only composition point:
it imports each class from here and builds one Settings instance holding
all of them.
"""
