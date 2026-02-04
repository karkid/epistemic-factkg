#!/usr/bin/env python3
"""
Test script for AI2THORSemanticDataSource

This script helps test the AI2THORSemanticDataSource while implementing it.
"""

import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_basic_functionality():
    """Test basic functionality of AI2THORSemanticDataSource"""
    print("=" * 50)
    print("Testing AI2THORSemanticDataSource")
    print("=" * 50)

    try:
        from generators.ai2thor.semantic.semantic_data_source import (
            AI2THORSemanticDataSource,
        )

        # Test initialization
        print("1. Testing initialization...")
        data_source = AI2THORSemanticDataSource()
        print("   ✓ Successfully initialized data source")

        # Test getting available claims
        print("2. Testing get_available_claims()...")
        claim_ids = data_source.get_available_claims()
        print(f"   ✓ Found {len(claim_ids)} claims")
        if claim_ids:
            print(f"   - Sample claim IDs: {claim_ids[:3]}")

        # Test getting individual claims
        print("3. Testing get_claim_by_id()...")
        if claim_ids:
            first_claim_id = claim_ids[0]
            claim = data_source.get_claim_by_id(first_claim_id)
            if claim:
                print(f"   ✓ Successfully retrieved claim: {first_claim_id}")
                print(f"   - Claim text: {claim.claim.text}")
                print(f"   - Label: {claim.label}")
                print(f"   - Scene ID: {claim.context.scene_id}")
                print(f"   - Number of claim triples: {len(claim.claim.claim_triples)}")
            else:
                print(f"   ✗ Failed to retrieve claim: {first_claim_id}")
        else:
            print("   - No claims available to test")

        # Test iterating through claims
        print("4. Testing get_claims() iterator...")
        claim_count = 0
        for claim in data_source.get_claims():
            claim_count += 1
            if claim_count <= 2:  # Show details for first 2
                print(f"   - Claim {claim_count}: {claim.rec_id}")
                print(f"     Text: {claim.claim.text}")
            elif claim_count == 3:
                print(f"   - ... (showing first 2 of {len(claim_ids)} total)")
                break
        print("   ✓ Successfully iterated through claims")

        # Test cleanup
        print("5. Testing cleanup()...")
        data_source.cleanup()
        print("   ✓ Successfully cleaned up")

        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_detailed_claim_structure():
    """Test detailed structure of a claim"""
    print("\\nTesting detailed claim structure...")

    try:
        from generators.ai2thor.semantic.semantic_data_source import (
            AI2THORSemanticDataSource,
        )

        data_source = AI2THORSemanticDataSource()
        claim_ids = data_source.get_available_claims()

        if claim_ids:
            claim = data_source.get_claim_by_id(claim_ids[0])
            print(f"\\nDetailed structure of claim '{claim.rec_id}':")
            print(f"  - rec_id: {claim.rec_id}")
            print(f"  - label: {claim.label}")
            print(f"  - claim.text: {claim.claim.text}")
            print(f"  - claim.claim_triples: {len(claim.claim.claim_triples)} triples")
            if claim.claim.claim_triples:
                print(f"    Sample triple: {claim.claim.claim_triples[0]}")
            print(f"  - reasoning.structural: {claim.reasoning.structural}")
            print(f"  - evidence.evidence_source: {claim.evidence.evidence_source}")
            print(
                f"  - evidence.evidence_source_type: {claim.evidence.evidence_source_type}"
            )
            print(
                f"  - evidence.evidence_triples: {len(claim.evidence.evidence_triples)} triples"
            )
            print(f"  - context.scene_id: {claim.context.scene_id}")
            print(f"  - context.generator: {claim.context.generator}")
            print(f"  - meta.created_utc: {claim.meta.created_utc}")
        else:
            print("No claims available for detailed testing")

    except Exception as e:
        print(f"Error during detailed testing: {e}")
        import traceback

        traceback.print_exc()


def test_configuration():
    """Test configuration loading"""
    print("\\nTesting configuration...")

    try:
        from generators.ai2thor.semantic.semantic_data_source import (
            AI2THORSemanticDataSource,
        )

        # Test with default config
        print("1. Testing with default config...")
        data_source = AI2THORSemanticDataSource()
        print(f"   Config loaded: {len(data_source.config)} keys")
        if "claim_files" in data_source.config:
            print(f"   Claim files: {data_source.config['claim_files']}")

        # Test with non-existent config (should use defaults)
        print("2. Testing with non-existent config...")
        data_source2 = AI2THORSemanticDataSource("nonexistent.yaml")
        print(f"   Fallback config loaded: {len(data_source2.config)} keys")

    except Exception as e:
        print(f"Error during configuration testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("AI2THORSemanticDataSource Test Suite")
    print("====================================")

    # Run all tests
    success = test_basic_functionality()

    if success:
        test_configuration()
        test_detailed_claim_structure()
        print("\\n🎉 Test suite completed successfully!")
    else:
        print("\\n❌ Test suite failed!")
        sys.exit(1)
