from convert_averitec import load_json_or_jsonl, convert_one_averitec, write_jsonl

records = load_json_or_jsonl("averitec/train.jsonl")
unified = [
    convert_one_averitec(r, f"averitec-train-{i:06d}")
    for i, r in enumerate(records, start=1)
]
write_jsonl("unified/averitec_train_unified.jsonl", unified)


records = load_json_or_jsonl("averitec/dev.jsonl")
unified = [
    convert_one_averitec(r, f"averitec-dev-{i:06d}")
    for i, r in enumerate(records, start=1)
]
write_jsonl("unified/averitec_dev_unified.jsonl", unified)

records = load_json_or_jsonl("averitec/test.jsonl")
unified = [
    convert_one_averitec(r, f"averitec-test-{i:06d}")
    for i, r in enumerate(records, start=1)
]
write_jsonl("unified/averitec_test_unified.jsonl", unified)

import json
from collections import Counter

ctr = Counter()
with open("unified/averitec_train_unified.jsonl") as f:
    for line in f:
        ctr[json.loads(line)["verdict"]["label"]] += 1

print(ctr)


ctr = Counter()
with open("unified/averitec_train_unified.jsonl") as f:
    for line in f:
        ctr[json.loads(line)["epistemic"]["proof_type"]] += 1

print(ctr)

