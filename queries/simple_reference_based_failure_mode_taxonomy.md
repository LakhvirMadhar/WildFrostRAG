# Failure Mode Taxonomy - Axial Codes

In this document, we present a taxonomy of failure modes identified in the evaluation of a language model (LLM). The approach involves axial coding, which organizes open codes into higher-level categories based on common themes and patterns. This structure aids in understanding the nature of the failures and can inform strategies for improvement.

## Inaccurate Information

This axial code encompasses instances where the model provides incorrect or fabricated information about game elements, including abilities, stats, and mechanics. These inaccuracies can mislead users and hinder their understanding of the game.

### Sub-categories:
- **Incorrect Stats and Abilities**: Instances where the model generates wrong values or descriptions for characters, cards, or abilities.
  - Open Codes: 1, 3, 6, 7, 8, 12, 19, 20, 29, 36, 38, 40

- **Fabricated Elements**: Instances where the model invents non-existent game elements or attributes.
  - Open Codes: 16, 17, 26, 27, 30, 31, 32, 41, 43, 45

## Vague or Non-specific Responses

This axial code captures responses that lack clarity or specificity, often leading to confusion or misinterpretation. These responses may contain some correct information but fail to provide the necessary context or detail.

### Sub-categories:
- **Vague Advice**: Responses that offer advice or commentary that is correct but lacks specificity, making it less useful.
  - Open Codes: 2, 5, 10, 11, 13, 14, 15, 21, 22, 24, 25, 33, 34, 46, 48

- **Non-answer Answers**: Responses that do not adequately address the query, often resorting to guessing or providing irrelevant information.
  - Open Codes: 9, 44, 47

## Misunderstanding Game Mechanics

This axial code includes instances where the model demonstrates a lack of understanding of the game's mechanics or rules, leading to incorrect interpretations or advice.

### Sub-categories:
- **Misinterpretation of Game Elements**: Instances where the model confuses game elements, such as treating enemies as allies or misunderstanding fight locations.
  - Open Codes: 4, 8, 12, 36, 40

- **Incorrect Contextual Understanding**: Responses that misplace game elements within the wrong context or fail to connect them correctly to the game mechanics.
  - Open Codes: 9, 10, 11, 12, 43

## Summary

The analysis of failure modes in the LLM evaluation reveals three primary axial codes: Inaccurate Information, Vague or Non-specific Responses, and Misunderstanding Game Mechanics. Each category highlights specific issues that can lead to user confusion and misinformation. Addressing these failure modes is crucial for enhancing the model's reliability and effectiveness in providing accurate and contextually relevant information. By refining the model's understanding of game mechanics and improving the clarity of its responses, we can significantly enhance user experience and trust in the system.