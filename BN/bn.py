from collections import defaultdict
import random

DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

START = "<START>"
END = "<END>"


def tokenize(sentence):
    return [START] + sentence.lower().split() + [END]


sentences = [tokenize(sentence) for sentence in DATASET]


# First-order model

first_counts = defaultdict(lambda: defaultdict(int))

for sentence in sentences:
    for current, next_word in zip(sentence, sentence[1:]):
        first_counts[current][next_word] += 1


def make_probabilities(counts):
    probabilities = {}

    for current, next_counts in counts.items():
        total = sum(next_counts.values())
        probabilities[current] = {
            word: count / total
            for word, count in sorted(next_counts.items())
        }

    return probabilities


first_probabilities = make_probabilities(first_counts)


def most_probable(probabilities, current):
    options = probabilities.get(current, {})

    if not options:
        return None

    return max(options, key=options.get)


def sample_next(probabilities, current):
    options = probabilities.get(current, {})

    if not options:
        return None

    words = list(options.keys())
    weights = list(options.values())

    return random.choices(words, weights=weights, k=1)[0]


def generate_greedy(probabilities, max_length=30):
    current = START
    result = []

    for _ in range(max_length):
        next_word = most_probable(probabilities, current)

        if next_word is None or next_word == END:
            break

        result.append(next_word)
        current = next_word

    return " ".join(result)


def generate_sampled(probabilities, max_length=30):
    current = START
    result = []

    for _ in range(max_length):
        next_word = sample_next(probabilities, current)

        if next_word is None or next_word == END:
            break

        result.append(next_word)
        current = next_word

    return " ".join(result)


# Second-order model

second_counts = defaultdict(lambda: defaultdict(int))

for sentence in sentences:
    for first, second, next_word in zip(
        sentence,
        sentence[1:],
        sentence[2:],
    ):
        second_counts[(first, second)][next_word] += 1


second_probabilities = make_probabilities(second_counts)


def generate_second_order(probabilities, max_length=30):
    current = [START]
    result = []

    next_word = sample_next(first_probabilities, current[0])

    if next_word is None or next_word == END:
        return ""

    result.append(next_word)
    current.append(next_word)

    for _ in range(max_length - 1):
        context = tuple(current[-2:])
        next_word = sample_next(probabilities, context)

        if next_word is None or next_word == END:
            break

        result.append(next_word)
        current.append(next_word)

    return " ".join(result)


# Values

print("DATASET")
for sentence in sentences:
    print(" ".join(sentence))

print("\nFIRST-ORDER CONDITIONAL PROBABILITIES")
for word, probabilities in first_probabilities.items():
    values = ", ".join(
        f"{next_word}={probability:.4f}"
        for next_word, probability in probabilities.items()
    )
    print(f"{word}: {values}")

print("\nFIRST-ORDER NORMALISATION")
for word, probabilities in first_probabilities.items():
    print(f"{word}: {sum(probabilities.values()):.4f}")

print("\nMOST PROBABLE NEXT WORD")
for word in ["the", "cat", "dog", "sat", "ran"]:
    print(f"{word} -> {most_probable(first_probabilities, word)}")


random.seed(42)

print("\nGREEDY GENERATION")
for _ in range(5):
    print(generate_greedy(first_probabilities))

print("\nSAMPLING GENERATION")
for _ in range(5):
    print(generate_sampled(first_probabilities))


print("\nSECOND-ORDER CONDITIONAL PROBABILITIES")
for context, probabilities in second_probabilities.items():
    values = ", ".join(
        f"{next_word}={probability:.4f}"
        for next_word, probability in probabilities.items()
    )
    print(f"{context}: {values}")

print("\nSECOND-ORDER NORMALISATION")
for context, probabilities in second_probabilities.items():
    print(f"{context}: {sum(probabilities.values()):.4f}")

print("\nSECOND-ORDER SAMPLING")
for _ in range(5):
    print(generate_second_order(second_probabilities))

