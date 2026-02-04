#!/usr/bin/env python3
"""
Dataset Quality Checker

Check quality and structure of epistemic-factkg datasets.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def check_dataset_quality(dataset_path: Path):
    """Check dataset quality and structure."""

    print(f"🔍 Checking dataset quality: {dataset_path}")

    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return False

    line_count = 0
    label_counts = Counter()
    reasoning_counts = Counter()
    evidence_types = Counter()
    generator_counts = Counter()
    pramana_type_count = 0
    relation_type_count = 0
    inscene_hasobject_count = 0

    try:
        with open(dataset_path, "r") as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Count labels
                if "label" in record:
                    label_counts[record["label"]] += 1

                # Count reasoning types
                if "structural_reasoning" in record:
                    reasoning_counts[record["structural_reasoning"]] += 1

                # Count evidence types
                if "evidence_type" in record:
                    evidence_types[record["evidence_type"]] += 1

                # Count generators
                if "context" in record and "generator" in record["context"]:
                    generator_counts[record["context"]["generator"]] += 1

                # Check for problematic fields
                line_str = line
                if '"pramana_type"' in line_str:
                    pramana_type_count += 1
                if "relation type" in line_str:
                    relation_type_count += 1
                if '"inScene"' in line_str or '"hasObject"' in line_str:
                    inscene_hasobject_count += 1

        # Report results
        print("\n📊 Dataset Statistics:")
        print(f"   Total records: {line_count:,}")

        print("\n🏷️  Labels:")
        for label, count in label_counts.most_common():
            pct = (count / line_count) * 100 if line_count > 0 else 0
            print(f"   {label}: {count:,} ({pct:.1f}%)")

        print("\n🧠 Reasoning Types:")
        for reasoning, count in reasoning_counts.most_common():
            pct = (count / line_count) * 100 if line_count > 0 else 0
            print(f"   {reasoning}: {count:,} ({pct:.1f}%)")

        print("\n📋 Evidence Types:")
        for etype, count in evidence_types.most_common():
            pct = (count / line_count) * 100 if line_count > 0 else 0
            print(f"   {etype}: {count:,} ({pct:.1f}%)")

        print("\n🤖 Generators:")
        for gen, count in generator_counts.most_common():
            pct = (count / line_count) * 100 if line_count > 0 else 0
            print(f"   {gen}: {count:,} ({pct:.1f}%)")

        print("\n🔧 Quality Checks:")

        # pramana_type check
        if pramana_type_count == 0:
            print("   ✅ No pramana_type found (expected)")
        else:
            print(
                f"   ❌ Found {pramana_type_count} pramana_type instances (unexpected)"
            )

        # evidence_type check
        total_evidence = sum(evidence_types.values())
        if total_evidence > 0:
            print(f"   ✅ Evidence types present: {total_evidence:,} instances")
        else:
            print("   ❌ No evidence types found")

        # relation type check
        if relation_type_count == 0:
            print("   ✅ No 'relation type' found (expected)")
        else:
            print(f"   ❌ Found {relation_type_count} 'relation type' instances")

        # inScene/hasObject relations
        if inscene_hasobject_count > 0:
            print(f"   📊 inScene/hasObject relations: {inscene_hasobject_count:,}")
        else:
            print("   ⚠️  No inScene/hasObject relations found")

        return True

    except Exception as e:
        print(f"❌ Error during quality check: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Check Dataset Quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("dataset_file", type=Path, help="Dataset JSONL file to check")

    args = parser.parse_args()

    success = check_dataset_quality(args.dataset_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
