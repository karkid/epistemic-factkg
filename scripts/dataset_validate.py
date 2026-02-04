#!/usr/bin/env python3
"""
Dataset Validator

Validate epistemic-factkg datasets.
"""

import argparse
import json
import sys
from pathlib import Path


def validate_dataset(dataset_path: Path, strict: bool = False) -> bool:
    """Validate dataset format and content."""

    print(f"🔍 Validating dataset: {dataset_path}")

    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return False

    required_fields = {
        "id",
        "claim",
        "label",
        "structural_reasoning",
        "evidence",
        "context",
        "meta",
    }

    evidence_fields = {"triples", "source", "source_type", "urls"}

    errors = []
    warnings = []
    line_num = 0

    try:
        with open(dataset_path, "r") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue

                # Check required fields
                missing = required_fields - set(record.keys())
                if missing:
                    errors.append(f"Line {line_num}: Missing fields: {missing}")

                # Check evidence structure
                if "evidence" in record:
                    if not isinstance(record["evidence"], dict):
                        errors.append(f"Line {line_num}: 'evidence' must be a dict")
                    else:
                        missing_ev = evidence_fields - set(record["evidence"].keys())
                        if missing_ev:
                            errors.append(
                                f"Line {line_num}: Missing evidence fields: {missing_ev}"
                            )

                # Check label values
                if "label" in record:
                    if record["label"] not in ["SUPPORTED", "REFUTED"]:
                        warnings.append(
                            f"Line {line_num}: Unexpected label: {record['label']}"
                        )

                if line_num % 1000 == 0:
                    print(f"   📊 Validated {line_num} records...")

        print("\n📊 Validation Results:")
        print(f"   Total records: {line_num}")
        print(f"   Errors: {len(errors)}")
        print(f"   Warnings: {len(warnings)}")

        if errors:
            print("\n❌ Errors found:")
            for error in errors[:10]:  # Show first 10
                print(f"   - {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")

        if warnings and not strict:
            print("\n⚠️  Warnings:")
            for warning in warnings[:5]:  # Show first 5
                print(f"   - {warning}")
            if len(warnings) > 5:
                print(f"   ... and {len(warnings) - 5} more warnings")

        success = len(errors) == 0 and (not strict or len(warnings) == 0)

        if success:
            print("\n✅ Dataset validation passed!")
        else:
            print("\n❌ Dataset validation failed!")

        return success

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Validate Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "dataset_file", type=Path, help="Dataset JSONL file to validate"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Fail on warnings as well as errors"
    )

    args = parser.parse_args()

    success = validate_dataset(args.dataset_file, args.strict)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
