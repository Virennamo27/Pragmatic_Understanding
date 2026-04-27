#!/usr/bin/env python3

import json
import random
from collections import Counter
from statistics import mean


TRAIN_PATH = "data/train.json"
VALIDATION_PATH = "data/validation.json"
TEST_PATH = "data/test.json"
RANDOM_SEED = 42
SHORT_LENGTH_THRESHOLD = 5
RANDOM_EXAMPLES_PER_SPLIT = 10


def load_split(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    return data


def pair_key(sample):
    return (sample.get("context", ""), sample.get("reply", ""))


def label_distribution(samples):
    return dict(Counter(sample.get("label") for sample in samples))


def short_sample_counts(samples):
    context_short = sum(1 for s in samples if len(s.get("context", "")) < SHORT_LENGTH_THRESHOLD)
    reply_short = sum(1 for s in samples if len(s.get("reply", "")) < SHORT_LENGTH_THRESHOLD)
    return context_short, reply_short


def average_lengths(samples):
    if not samples:
        return 0.0, 0.0
    context_avg = mean(len(s.get("context", "")) for s in samples)
    reply_avg = mean(len(s.get("reply", "")) for s in samples)
    return context_avg, reply_avg


def print_random_examples(split_name, samples, k=RANDOM_EXAMPLES_PER_SPLIT):
    print(f"\n{split_name}: {k} random examples")
    if not samples:
        print("  (no samples)")
        return
    count = min(k, len(samples))
    chosen = random.sample(samples, count)
    for idx, sample in enumerate(chosen, 1):
        context = sample.get("context", "")
        reply = sample.get("reply", "")
        label = sample.get("label", "")
        print(f"  [{idx}] label={label}")
        print(f"      context: {context}")
        print(f"      reply  : {reply}")


def main():
    random.seed(RANDOM_SEED)

    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    test = load_split(TEST_PATH)

    print("=" * 60)
    print("SARCASM DATASET INTEGRITY REPORT")
    print("=" * 60)

    print("\n1) Dataset sizes")
    print(f"  train      : {len(train)}")
    print(f"  validation : {len(validation)}")
    print(f"  test       : {len(test)}")

    print("\n2) Label distribution per split")
    print(f"  train      : {label_distribution(train)}")
    print(f"  validation : {label_distribution(validation)}")
    print(f"  test       : {label_distribution(test)}")

    print("\n3) Split overlap check (context + reply)")
    train_pairs = {pair_key(s) for s in train}
    validation_pairs = {pair_key(s) for s in validation}
    test_pairs = {pair_key(s) for s in test}

    train_val_overlap = train_pairs & validation_pairs
    train_test_overlap = train_pairs & test_pairs
    val_test_overlap = validation_pairs & test_pairs

    print(f"  train vs validation duplicates: {len(train_val_overlap)}")
    print(f"  train vs test duplicates      : {len(train_test_overlap)}")
    print(f"  validation vs test duplicates : {len(val_test_overlap)}")

    print("\n4) Extremely short samples (< 5 chars)")
    train_context_short, train_reply_short = short_sample_counts(train)
    val_context_short, val_reply_short = short_sample_counts(validation)
    test_context_short, test_reply_short = short_sample_counts(test)
    print(
        f"  train      : context<{SHORT_LENGTH_THRESHOLD}={train_context_short}, "
        f"reply<{SHORT_LENGTH_THRESHOLD}={train_reply_short}"
    )
    print(
        f"  validation : context<{SHORT_LENGTH_THRESHOLD}={val_context_short}, "
        f"reply<{SHORT_LENGTH_THRESHOLD}={val_reply_short}"
    )
    print(
        f"  test       : context<{SHORT_LENGTH_THRESHOLD}={test_context_short}, "
        f"reply<{SHORT_LENGTH_THRESHOLD}={test_reply_short}"
    )

    print("\n5) Average text lengths")
    train_context_avg, train_reply_avg = average_lengths(train)
    val_context_avg, val_reply_avg = average_lengths(validation)
    test_context_avg, test_reply_avg = average_lengths(test)
    print(
        f"  train      : avg context={train_context_avg:.2f}, "
        f"avg reply={train_reply_avg:.2f}"
    )
    print(
        f"  validation : avg context={val_context_avg:.2f}, "
        f"avg reply={val_reply_avg:.2f}"
    )
    print(
        f"  test       : avg context={test_context_avg:.2f}, "
        f"avg reply={test_reply_avg:.2f}"
    )

    print("\n6) Random examples")
    print_random_examples("train", train)
    print_random_examples("validation", validation)
    print_random_examples("test", test)

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)


if __name__ == "__main__":
    main()
