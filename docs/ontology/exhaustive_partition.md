# Exhaustive Partition (Covering Axiom)

A core ontology design pattern from OWL/RDF knowledge engineering.

## The Problem

When a parent class has subclasses that account for **every possible instance**, no instance should sit directly on the parent. The parent is abstract — every member must belong to exactly one subclass.

If instances DO sit directly on the parent alongside subclasses, the partition is **incomplete** — you're missing a subclass.

## Formal Definition

An **exhaustive (complete) partition** of a class requires two axioms:

1. **Covering axiom** (`owl:unionOf`): The parent class is equivalent to the union of its subclasses. Every instance of the parent belongs to at least one subclass.
2. **Disjointness axiom** (`owl:disjointWith`): Subclasses are mutually exclusive. No instance belongs to more than one subclass.

In OWL 2, both are combined into a single statement: `owl:disjointUnionOf`.

```
Parent ≡ SubclassA ∪ SubclassB ∪ SubclassC
SubclassA ⊓ SubclassB = ∅
SubclassA ⊓ SubclassC = ∅
SubclassB ⊓ SubclassC = ∅
```

## Two Types of Decomposition

| Type | Direct instances on parent? | When to use |
|---|---|---|
| **Exhaustive (complete)** | No — parent is abstract | Subclasses cover ALL instances |
| **Non-exhaustive (incomplete)** | Yes — some instances don't fit any subclass | Subclasses only cover SOME instances |

## How We Discovered This

### The bell targeting problem

Bells in Wildfrost target "non-boss enemies." We modeled this as:

```
Bell -[:TARGETS_CARD_TYPE]-> CardType:enemies
```

But `enemies` was both a **leaf node** (regular enemy cards sat directly on it) and a **parent node** (bosses, minibosses, enemy_clunkers were subtypes). This meant bell targeting implied bosses were targeted too — incorrect.

### The root cause

The `enemies` CardType was an incomplete partition. Every enemy in Wildfrost is either a regular enemy, boss, miniboss, or enemy clunker — the subclasses are exhaustive. But we were missing the `non_boss_enemies` subclass, so regular enemies sat directly on the parent.

### Before (incomplete partition)

```
enemies (parent + leaf — regular enemies sit here)
├── bosses          SUBTYPE_OF enemies
├── minibosses      SUBTYPE_OF enemies
└── enemy_clunkers  SUBTYPE_OF enemies AND clunkers
```

Problem: `TARGETS_CARD_TYPE -> enemies` implies all subtypes, including bosses.

### After (exhaustive partition)

```
enemies (abstract parent — no direct cards)
├── non_boss_enemies  SUBTYPE_OF enemies  ← regular enemies move here
│   └── enemy_clunkers  SUBTYPE_OF non_boss_enemies AND clunkers
├── bosses            SUBTYPE_OF enemies
└── minibosses        SUBTYPE_OF enemies
```

Now `TARGETS_CARD_TYPE -> non_boss_enemies` correctly excludes bosses/minibosses. Traversing SUBTYPE_OF from `non_boss_enemies` finds only `enemy_clunkers` — which ARE non-boss enemies.

### Query implications

Naive queries like `MATCH (c:Card)-[:HAS_CARD_TYPE]->(ct:CardType {name: "enemies"})` return empty after the restructure because `enemies` is abstract. The correct pattern traverses the hierarchy:

```cypher
MATCH (ct:CardType {name: "enemies"})<-[:SUBTYPE_OF*0..]-(sub:CardType)<-[:HAS_CARD_TYPE]-(c:Card)
RETURN c
```

This was already needed before the restructure — the naive query missed bosses/minibosses/enemy_clunkers even when regular enemies sat on the parent.

## Key Takeaway

If a parent class has subclasses and you find instances sitting directly on the parent, ask: **do the subclasses cover all possible instances?** If yes, you have a missing subclass. Add it, make the parent abstract, and the taxonomy becomes clean.

## References

- [W3C OWL Guide - Section 5.3: Disjoint Classes](https://www.w3.org/TR/owl-guide/) — `owl:unionOf` + `owl:disjointWith` for partitions
- [W3C Value Partitions Best Practice](https://www.w3.org/2001/sw/BestPractices/OEP/SpecifiedValues-20040721/) — Step-by-step design pattern with OWL code
- [OWL 2 Primer](https://www.w3.org/2007/OWL/draft/owl2-primer/) — `owl:disjointUnionOf` (combined axiom)
- [Ontology Design Patterns](https://www.emergentmind.com/topics/ontology-design-patterns-odps) — Reusable patterns for ontology engineering
