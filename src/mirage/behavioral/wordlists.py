"""Word lists for the behavioral templates.

Adjectives are unambiguous, common in movie/product reviews, and >= 4 letters so a one-character
typo still leaves a recognisable word. Extend freely; counts in configs/behavioral.yaml are drawn
from the cartesian product, so longer lists only add variety.
"""

POS_ADJ = [
    "great",
    "good",
    "excellent",
    "wonderful",
    "amazing",
    "brilliant",
    "fantastic",
    "superb",
    "lovely",
    "enjoyable",
    "delightful",
    "charming",
    "impressive",
    "beautiful",
    "perfect",
    "terrific",
    "outstanding",
    "pleasant",
    "memorable",
    "fun",
]

NEG_ADJ = [
    "bad",
    "terrible",
    "awful",
    "horrible",
    "boring",
    "dull",
    "poor",
    "disappointing",
    "dreadful",
    "weak",
    "mediocre",
    "annoying",
    "painful",
    "tedious",
    "lousy",
    "pathetic",
    "forgettable",
    "unpleasant",
    "bland",
    "messy",
]

NOUNS = [
    "movie",
    "film",
    "plot",
    "acting",
    "story",
    "script",
    "ending",
    "soundtrack",
    "cast",
    "dialogue",
    "performance",
    "direction",
    "experience",
    "show",
    "sequel",
]

# Two groups so the name-swap test crosses Indian and Western names.
NAMES_A = [
    "Priya",
    "Rahul",
    "Ananya",
    "Arjun",
    "Kavya",
    "Rohan",
    "Meera",
    "Vikram",
    "Diya",
    "Aditya",
]
NAMES_B = [
    "John",
    "Emily",
    "Michael",
    "Sarah",
    "David",
    "Jessica",
    "James",
    "Laura",
    "Daniel",
    "Emma",
]

POS_VERBS = ["loved", "enjoyed", "liked", "adored"]
NEG_VERBS = ["hated", "disliked", "loathed", "despised"]
