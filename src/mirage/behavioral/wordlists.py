"""Word lists for the behavioral templates.

Adjectives are unambiguous and common in movie reviews. The typo test uses only those with >= 4
letters, so a one-character typo still leaves a recognisable word. Extend freely; counts in
configs/behavioral.yaml are drawn from the cartesian product, so longer lists only add variety.
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
    "beautiful",
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
    "sloppy",  # replaced "messy" (D33): only 12 training uses, and sometimes affectionate
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

# Unseen-names probe: Indian vs Western first names, paired by position AND gender (D32), so a
# changed prediction can't come from a gender swap. None of NAMES_A occurs in SST-2's training
# data, so for the classical models this mainly checks that unfamiliar names are ignored.
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
    "Emily",
    "John",
    "Sarah",
    "Michael",
    "Jessica",
    "David",
    "Laura",
    "James",
    "Emma",
    "Daniel",
]

# Mild adjectives for the name-shortcut test: weak enough that a name could tip the balance.
# Nouns that name a whole work, so "{name}'s latest {noun}" and "Another {name} {noun}" read
# naturally ("Another Spielberg acting" would not).
WORK_NOUNS = ["movie", "film", "sequel", "drama", "comedy", "thriller", "project", "picture"]
MILD_POS = ["decent", "solid", "watchable", "pleasant", "nice"]
MILD_NEG = ["flat", "uneven", "forgettable", "bland", "weak"]

POS_VERBS = ["loved", "enjoyed", "liked", "adored"]
NEG_VERBS = ["hated", "disliked", "loathed", "despised"]
