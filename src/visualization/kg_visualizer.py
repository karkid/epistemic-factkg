#!/usr/bin/env python3
"""Clean, maintainable knowledge graph visualizer."""

import argparse
from pathlib import Path

from .network_builder import NetworkBuilder
from .html_generator import HTMLGenerator


def create_visualization(
    graph_file: Path,
    output_file: Path,
    title: str = "Knowledge Graph",
    show_literals: bool = True,
    max_nodes: int = None,
    theme: str = 'light'
) -> None:
    """Create an interactive knowledge graph visualization.
    
    Args:
        graph_file: Path to RDF graph file (turtle, json-ld, etc.)
        output_file: Path for output HTML file
        title: Title for the visualization
        show_literals: Whether to include literal values
        max_nodes: Maximum number of entity nodes (None for unlimited)
        theme: Theme name ('light', 'dark', 'monochrome_light', 'monochrome_dark')
    """
    # Build network
    builder = NetworkBuilder(theme=theme)
    builder.load_graph(graph_file)
    
    net = builder.create_network(show_literals=show_literals, max_nodes=max_nodes)
    
    # Get complete data for JavaScript
    all_nodes, all_edges = builder.get_all_nodes_and_edges_data()
    
    # Get container data from the network builder
    container_data = getattr(net, 'container_data', [])
    
    # Ensure output is in 'out' directory
    if not output_file.is_absolute() and output_file.parent == Path('.'):
        output_file = Path('out') / output_file
    
    # Generate HTML
    output_file.parent.mkdir(parents=True, exist_ok=True)
    html_string = net.generate_html()
    
    # Inject focused interaction JavaScript with data
    html_generator = HTMLGenerator()
    js_code = html_generator.get_focused_interaction_js(all_nodes, all_edges, container_data)
    html_string = html_generator.inject_javascript(html_string, js_code)
    
    # Save HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    # Print statistics
    stats = builder.get_statistics()
    print(f"Visualization saved to {output_file}")
    print(f"Open {output_file} in your browser to explore the knowledge graph")
    print()
    print("Graph Statistics:")
    print(f"- Total nodes: {stats['total_nodes']}")
    print(f"- Entity nodes: {stats['entity_nodes']}")
    print(f"- Total edges: {stats['total_edges']}")
    print(f"- Unique relations: {stats['unique_relations']}")
    print(f"- Literals: {stats['literals']} ({'shown' if show_literals else 'hidden'})")
    print(f"- Relation types: {', '.join(stats['relation_types'])}")
    for node_type, count in stats['node_types'].items():
        print(f"- {node_type.title()} nodes: {count}")


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
        help="Output HTML file (default: input_file_visualization.html)"
    )
    
    parser.add_argument(
        "--title", "-t",
        default="Knowledge Graph Visualization",
        help="Title for the visualization"
    )
    
    parser.add_argument(
        "--no-literals",
        action="store_true",
        help="Hide literal values to reduce clutter"
    )
    
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Maximum number of entity nodes to show"
    )
    
    parser.add_argument(
        '--theme',
        type=str,
        default='light',
        choices=['light', 'dark', 'monochrome_light', 'monochrome_dark'],
        help='Color theme for the visualization'
    )
    
    parser.add_argument(
        '--list-themes',
        action='store_true',
        help='List available themes and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list themes
    if args.list_themes:
        from .themes import get_theme_info
        print("Available themes:")
        for theme_name, theme_desc in get_theme_info().items():
            print(f"  {theme_name}: {theme_desc}")
        return
    
    # Validate required input argument
    if not args.input:
        parser.error('--input/-i is required when not listing themes')
    
    # Set default output file in 'out' directory
    if not args.output:
        args.output = Path('out') / 'kg_visualization.html'
    
    # Create visualization
    create_visualization(
        graph_file=args.input,
        output_file=args.output,
        title=args.title,
        show_literals=not args.no_literals,
        max_nodes=args.max_nodes,
        theme=args.theme
    )


if __name__ == "__main__":
    main()