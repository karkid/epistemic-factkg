#!/usr/bin/env python3
"""
Semantic Data Source Tester

Test and explore AI2-THOR semantic claims.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "../src"))

from generators.ai2thor.semantic.semantic_data_source import AI2THORSemanticDataSource


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Test Semantic Claims",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", "-c", help="Config file path (optional)")
    parser.add_argument("--list", "-l", action="store_true", help="List all claim IDs")
    parser.add_argument("--count", action="store_true", help="Show claim count")
    parser.add_argument("--show", help="Show specific claim by ID")
    parser.add_argument("--validate", action="store_true", help="Validate all claims")

    args = parser.parse_args()

    try:
        print("🔄 Loading semantic data source...")

        data_source = (
            AI2THORSemanticDataSource(config_path=args.config)
            if args.config
            else AI2THORSemanticDataSource()
        )

        if args.count:
            claims = data_source.get_available_claims()
            print(f"📊 Total claims: {len(claims)}")

        elif args.list:
            claims = data_source.get_available_claims()
            print(f"📋 Available claims ({len(claims)} total):")
            for claim_id in claims:
                print(f"   • {claim_id}")

        elif args.show:
            claim = data_source.get_claim_by_id(args.show)
            if claim:
                print(f"📄 Claim: {args.show}")
                print(f"   Text: {claim.claim.text}")
                print(f"   Label: {claim.label}")
            else:
                print(f"❌ Claim not found: {args.show}")

        elif args.validate:
            print("✅ Validating claims...")
            claims = list(data_source.get_claims())
            print(f"✅ All {len(claims)} claims validated successfully")

        else:
            # Default: show summary
            claims = data_source.get_available_claims()
            print("📊 Semantic data source ready")
            print(f"   Claims available: {len(claims)}")
            print("   Use --list to see all claims")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
