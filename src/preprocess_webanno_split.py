#!/usr/bin/env python3
import os
import json
from typing import List, Dict
from sklearn.model_selection import train_test_split

# Set these paths
INPUT_TSV_DIR = "data/train"
OUTPUT_DIR = "training_data"

def parse_webanno_file(tsv_path: str) -> List[Dict]:
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    headers = []
    for line in lines:
        if line.startswith("#T_SP="):
            header = line.strip().split("webanno.custom.")[-1].split("|")[0]
            headers.append(header)

    tokens, labels = [], []
    sentence_tokens, sentence_labels = [], []
    previous_entity = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            continue

        token = parts[2]
        annotations = parts[3:]
        annotations += ["_"] * (len(headers) - len(annotations))  # pad

        # Determine active entity using priority
        active_entity = None
        priority = ["Timeperiodofstudy", "Locationofstudy", "Mainhypothesisandcorrespondingresults"]

        found_entities = [headers[idx] for idx, ann in enumerate(annotations) if ann.startswith("*[")]
        for p in priority:
            if p in found_entities:
                active_entity = p
                break
        if active_entity is None and found_entities:
            active_entity = found_entities[0]

        # Assign label
        if active_entity:
            label = f"I-{active_entity}" if previous_entity == active_entity else f"B-{active_entity}"
            previous_entity = active_entity
        else:
            label = "O"
            previous_entity = None

        sentence_tokens.append(token)
        sentence_labels.append(label)

        # End of sentence (heuristic)
        if token.endswith(".") or token in {"?", "!", "\r"}:
            tokens.append(sentence_tokens)
            labels.append(sentence_labels)
            sentence_tokens, sentence_labels = [], []
            previous_entity = None

    # Catch any leftover sentence
    if sentence_tokens:
        tokens.append(sentence_tokens)
        labels.append(sentence_labels)

    return [{"tokens": t, "labels": l} for t, l in zip(tokens, labels)]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []
    for file in os.listdir(INPUT_TSV_DIR):
        if file.endswith(".tsv"):
            filepath = os.path.join(INPUT_TSV_DIR, file)
            all_data.extend(parse_webanno_file(filepath))

    # Split dataset
    train, test = train_test_split(all_data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    # Write output
    with open(os.path.join(OUTPUT_DIR, "ner_train.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "ner_val.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "ner_test.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
