import argparse
import os
from pathlib import Path

from visualizer.builder import build_rdf_graph
from utils.exceptions import BuildError, DataSourceError


def main():
    """Enhanced command line interface with error handling."""
    parser = argparse.ArgumentParser(
        description="Create interactive knowledge graph visualizations with comprehensive error handling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input RDF file (turtle, json-ld, etc.)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/visualizer/knowledge_graph.html"),
        help="Output HTML file for the visualization",
    )

    args = parser.parse_args()

    try:
        print(f"🔍 Building visualization for: {args.input}")
        print(f"📁 Output path: {args.output}")

        # Create output folder
        folder = args.output.parent
        if folder:
            os.makedirs(folder, exist_ok=True)

        build_rdf_graph(args.input, output_html=args.output)

        print("✅ Visualization complete!")
        print(f"🌐 Open {args.output} in your browser")

    except DataSourceError as e:
        print(f"❌ Data source error: {e}")
        exit(1)
    except BuildError as e:
        print(f"❌ Build error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
