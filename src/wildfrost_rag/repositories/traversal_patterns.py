"""Shared graph traversal patterns for WildFrostRAG retrievers.

Used by VectorThenCypherRetriever and FulltextThenCypherRetriever to enrich
initial search results with structured graph data.
"""

# Enriched traversal query covering all Wildfrost knowledge graph relationships.
# Expects variables `doc` and `score` to already be bound by the calling query
# (e.g., from vector search or fulltext search YIELD).
#
# Identifies which entity type owns the Document, then enriches with
# entity-specific relationships. Uses pattern comprehension to avoid
# cartesian products from chained OPTIONAL MATCHes.
GRAPH_TRAVERSAL_QUERY = """
    WITH doc, score,
         head([(doc)<-[:HAS_DOCUMENT]-(c:Card) | c]) as card,
         head([(doc)<-[:HAS_DOCUMENT]-(b:Bell) | b]) as bell,
         head([(doc)<-[:HAS_DOCUMENT]-(ch:Charm) | ch]) as charm,
         head([(doc)<-[:HAS_DOCUMENT]-(f:Fight) | f]) as fight,
         head([(doc)<-[:HAS_DOCUMENT]-(cr:Crown) | cr]) as crown,
         head([(doc)<-[:HAS_DOCUMENT]-(st:Stat) | st]) as stat,
         head([(doc)<-[:HAS_DOCUMENT]-(z:Zone) | z]) as zone,
         head([(doc)<-[:HAS_DOCUMENT]-(me:MapEvent) | me]) as map_event,
         head([(doc)<-[:HAS_DOCUMENT]-(sh:Shop) | sh]) as shop
    RETURN doc, score,
           card, bell, charm, fight, crown, stat, zone, map_event, shop,
           // Card-specific traversals
           head([(card)-[:BELONGS_TO_TRIBE]->(t:Tribe) | t]) as tribe,
           head([(card)-[:HAS_CARD_TYPE]->(ct:CardType) | ct]) as cardtype,
           [(card)-[:HAS_KEYWORD]->(k:Keyword) | k.name] as keywords,
           [(card)-[:HAS_STAT]->(s:Stat) | s.name] as stats,
           [(card)-[:SUMMONS]->(sm:Card) | sm.card_name] as summons,
           [(card)-[:TRANSFORMS_INTO]->(tr:Card) | tr.card_name] as transforms_into,
           [(card)-[:CAN_BE_RECRUITED_AS]->(rc:Card) | rc.card_name] as can_recruit_as,
           // Fight-specific traversals
           [(fight)-[:FEATURES_ENEMY]->(e:Card) | e.card_name] as fight_enemies,
           // Bell-specific traversals
           [(bell)-[:APPLIES_CHARM]->(ac:Charm) | ac.name] as bell_charms,
           [(bell)-[:ADDS_TO_FIGHT]->(af:Card) | af.card_name] as bell_adds_cards,
           [(bell)-[:GRANTS_KEYWORD]->(gk:Keyword) | gk.name] as bell_grants_keywords
    ORDER BY score DESC
"""
