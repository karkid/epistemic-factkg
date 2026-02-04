#!/usr/bin/env python3
"""
Knowledge Graph Visualizer

Generate interactive HTML visualizations from knowledge graphs.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "../src"))

from visualizer.builder import build_rdf_graph
from utils.exceptions import BuildError, DataSourceError


def main():
    parser = argparse.ArgumentParser(
        description="🎨 Visualize Knowledge Graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_file", type=Path, help="Knowledge graph file (TTL, RDF, JSON-LD)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/visualizer/knowledge_graph.html"),
        help="Output HTML file",
    )

    args = parser.parse_args()

    try:
        print(f"🎨 Visualizing: {args.input_file}")
        print(f"📁 Output: {args.output}")

        build_rdf_graph(args.input_file, args.output)

        print("✅ Visualization complete!")
        print(f"🌐 Open {args.output} in your browser")

    except (DataSourceError, BuildError) as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
