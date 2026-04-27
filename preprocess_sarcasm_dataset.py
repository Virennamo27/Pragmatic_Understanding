#!/usr/bin/env python3

import json
import os
import re

import pandas as pd


INPUT_CSV = "archive/train-balanced-sarcasm.csv"
OUTPUT_DIR = "data"
TRAIN_OUT = os.path.join(OUTPUT_DIR, "train.json")
VALIDATION_OUT = os.path.join(OUTPUT_DIR, "validation.json")
TEST_OUT = os.path.join(OUTPUT_DIR, "test.json")
SAMPLE_SIZE = 100000
RANDOM_SEED = 42


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_banned_token(text: str) -> bool:
    return text in {"[deleted]", "[removed]"}


def contains_sarcasm_indicator(text: str) -> bool:
    # Match "/s" as a token-like indicator and "sarcasm" anywhere in text.
    return bool(re.search(r"(^|\s)/s(\s|$)", text)) or ("sarcasm" in text)


def save_json(records, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = pd.read_csv(INPUT_CSV, usecols=["label", "comment", "parent_comment"])
    total_before_cleaning = len(df)

    # Remove rows with null comment or parent_comment.
    df = df.dropna(subset=["comment", "parent_comment"])

    # Normalize early so token filters are applied consistently.
    df["comment"] = df["comment"].astype(str).map(normalize_text)
    df["parent_comment"] = df["parent_comment"].astype(str).map(normalize_text)

    # Remove deleted/removed comments in comment or parent_comment.
    mask_deleted_removed = (
        df["comment"].map(contains_banned_token)
        | df["parent_comment"].map(contains_banned_token)
    )
    df = df[~mask_deleted_removed]

    # Remove comments containing sarcasm indicators.
    mask_sarcasm_indicator = df["comment"].map(contains_sarcasm_indicator)
    df = df[~mask_sarcasm_indicator]

    rows_after_cleaning = len(df)

    # Convert labels to string classes.
    df["label"] = df["label"].map({0: "not_sarcastic", 1: "sarcastic"})
    df = df[df["label"].notna()]

    # Create final schema fields.
    df = df.rename(columns={"parent_comment": "context", "comment": "reply"})
    df = df[["context", "reply", "label"]]

    # Shuffle dataset.
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"Not enough rows after cleaning to sample {SAMPLE_SIZE}. "
            f"Available: {len(df)}"
        )

    # Sample exactly 100000 rows.
    df = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)
    final_subset_size = len(df)

    # Split 80/10/10.
    train_end = int(0.8 * final_subset_size)
    validation_end = train_end + int(0.1 * final_subset_size)

    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(train_df.to_dict(orient="records"), TRAIN_OUT)
    save_json(validation_df.to_dict(orient="records"), VALIDATION_OUT)
    save_json(test_df.to_dict(orient="records"), TEST_OUT)

    label_distribution = df["label"].value_counts().to_dict()

    print(f"total rows before cleaning: {total_before_cleaning}")
    print(f"rows after cleaning: {rows_after_cleaning}")
    print(f"final subset size: {final_subset_size}")
    print(f"train size: {len(train_df)}")
    print(f"validation size: {len(validation_df)}")
    print(f"test size: {len(test_df)}")
    print(f"label distribution: {label_distribution}")


if __name__ == "__main__":
    main()
