import argparse
import os
from pathlib import Path

from src.visualizer.builder import build_rdf_graph 


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Create interactive knowledge graph visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input RDF file (turtle, json-ld, etc.)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output HTML file for the visualization"
    )
        
    args = parser.parse_args()
    
    # Validate required input argument
    if not args.input:
        parser.error('--input/-i is required when not listing themes')

    # Validate required output argument
    if not args.output:
        parser.error('--output/-o is required when not listing themes')

    # Create output folder
    folder = args.output.parent
    if folder:
        os.makedirs(folder, exist_ok=True)
    
    build_rdf_graph(args.input, output_html=args.output)


if __name__ == "__main__":
    main()