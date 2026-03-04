"""
Shared graph traversal patterns for WildFrostRAG retrievers.

Used by VectorThenCypherRetriever and FulltextThenCypherRetriever to enrich
initial search results with structured graph data.
"""

# Single enriched traversal query covering all Wildfrost knowledge graph relationships.
# Expects variables `doc` and `score` to already be bound by the calling query
# (e.g., from vector search or fulltext search YIELD).
#
# Uses pattern comprehension to pick exactly one Card per Document, avoiding row
# explosion when multiple Cards link to the same Document (e.g., Leaders page).
# All multi-valued relationships also use pattern comprehensions to avoid
# cartesian products from chained OPTIONAL MATCHes.
GRAPH_TRAVERSAL_QUERY = """
    WITH doc, score,
         head([(doc)<-[:HAS_DOCUMENT]-(c:Card) | c]) as card
    RETURN doc, card,
           head([(card)-[:BELONGS_TO_TRIBE]->(t:Tribe) | t]) as tribe,
           head([(card)-[:HAS_CARD_TYPE]->(ct:CardType) | ct]) as cardtype,
           [(card)-[:HAS_KEYWORD]->(k:Keyword) | k.name] as keywords,
           [(card)-[hs:HAS_STAT]->(s:Stat) | {stat_name: s.name, value: hs.value}] as stats,
           [(card)-[:SUMMONS]->(sm:Card) | sm.card_name] as summons,
           [(card)-[:TRANSFORMS_INTO]->(tr:Card) | tr.card_name] as transforms_into,
           [(card)-[:CAN_BE_RECRUITED_AS]->(rc:Card) | rc.card_name] as can_recruit_as,
           score
    ORDER BY score DESC
"""
