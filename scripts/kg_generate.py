#!/usr/bin/env python3
"""
Knowledge Graph Generator

Main script for generating knowledge graphs from various sources.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "../src"))

from generators.ai2thor.kg.ontology import create_ai2thor_ontology
from knowledge_graph.core.builder import KnowledgeGraphBuilder
from utils.exceptions import DataSourceError, ConfigurationError, BuildError


def main():
    parser = argparse.ArgumentParser(
        description="🏗️ Generate Knowledge Graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--generator",
        "-g",
        choices=["ai2thor"],
        default="ai2thor",
        help="Generator type (default: ai2thor)",
    )
    parser.add_argument("--config", "-c", help="Config file path (optional)")
    parser.add_argument("--scenes", "-s", nargs="+", help="Specific scenes to process")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/knowledge_graph.ttl"),
        help="Output file path (default: output/knowledge_graph.ttl)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["turtle", "json-ld"],
        default="turtle",
        help="Output format (default: turtle)",
    )

    args = parser.parse_args()

    try:
        print(f"🔄 Initializing {args.generator} generator...")

        if args.generator == "ai2thor":
            from generators.ai2thor.kg.data_source import AI2THORDataSource

            # Create data source
            if args.config:
                data_source = AI2THORDataSource(config_path=args.config)
                print(f"   Using config: {args.config}")
            else:
                data_source = AI2THORDataSource()
                print("   Using default config")

            # Create ontology and builder
            ontology = create_ai2thor_ontology()
            builder = KnowledgeGraphBuilder(ontology=ontology)

            print("🏗️ Building knowledge graph...")

            # Build graph
            if args.scenes:
                # TODO: Handle multiple scenes
                scene_data = data_source.get_scene_by_id(args.scenes[0])
                result = builder.build_from_scene(scene_data)
            else:
                result = builder.build_from_source(data_source)

            # Report results
            print("\n📊 Build Results:")
            print(f"   Scenes processed: {len(result.scenes_processed)}")
            print(f"   Objects: {result.num_objects}")
            print(f"   Relations: {result.num_relations}")
            print(f"   Total triples: {result.total_triples}")

            if result.warnings:
                print(f"\n⚠️  Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:
                    print(f"   - {warning}")
                if len(result.warnings) > 3:
                    print(f"   ... and {len(result.warnings) - 3} more")

            if result.errors:
                print(f"\n❌ Errors: {len(result.errors)}")
                for error in result.errors[:3]:
                    print(f"   - {error}")
                if len(result.errors) > 3:
                    print(f"   ... and {len(result.errors) - 3} more")

            if result.success:
                # Create output directory
                args.output.parent.mkdir(parents=True, exist_ok=True)

                print(f"\n💾 Saving to: {args.output}")
                result.graph.serialize(destination=args.output, format=args.format)
                print(f"✅ Complete! {result.total_triples} triples generated")
            else:
                print(f"\n❌ Build failed with {len(result.errors)} errors")
                sys.exit(1)
        else:
            print(f"❌ Unsupported generator: {args.generator}")
            sys.exit(1)

    except (ConfigurationError, DataSourceError, BuildError) as e:
        print(f"❌ {type(e).__name__}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
