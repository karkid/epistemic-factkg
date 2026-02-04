import os
import shutil
import time
from pathlib import Path
from visualizer.network_builder import NetworkBuilder
from utils.exceptions import BuildError, DataSourceError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_template():
    """Load HTML template with error handling."""
    try:
        path = os.path.join(BASE_DIR, "templates", "template.html")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise DataSourceError(f"Template file not found: {path}")
    except Exception as e:
        raise BuildError(f"Failed to load template: {e}")


def split_html(html):
    """Split HTML content at body tag with error handling."""
    try:
        head = html.split("<body>")[0] + "</head>"
        body = "<body>" + html.split("<body>")[1]
        body = body.replace("</body>", "")
        return head, body
    except IndexError:
        raise BuildError("Invalid HTML structure - missing body tag")


def build_rdf_graph(
    ttl_file: Path, output_html: Path = Path("output/rdf_graph.html")
) -> None:
    """Build visualization graph with comprehensive error handling."""
    try:
        # Validate input file
        if not ttl_file.exists():
            raise DataSourceError(f"Input TTL file not found: {ttl_file}")

        # Force clean rebuild - remove output folder first
        if output_html.parent.exists():
            shutil.rmtree(output_html.parent)
        output_html.parent.mkdir(parents=True, exist_ok=True)

        node_builder = NetworkBuilder()

        # Configure filtering to reduce noise
        node_builder.set_filters(
            show_instances=True,
            show_classes=True,
            show_literals=True,
            max_nodes=300,  # Limit to prevent overwhelming display
        )

        # Hide some noisy relations that don't add much value
        node_builder.hide_relations(["comment", "label", "sameAs", "description"])

        # Load RDF
        node_builder.load_graph(ttl_file)
        net = node_builder.build()

        print("Triples:", len(node_builder.graph))
        print("Nodes displayed:", len(node_builder.nodes))

        # Generate PyVis HTML
        pyvis_html = net.generate_html()

        # Load template
        template = load_template()

        # Split PyVis html
        head, body = split_html(pyvis_html)

        # Add cache buster timestamp
        cache_buster = str(int(time.time()))

        # Inject into template
        final_html = (
            template.replace("{{ head }}", head)
            .replace("{{ body }}", body)
            .replace("{{ cache_buster }}", cache_buster)
        )

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(final_html)

        # Copy static files
        static_folder = os.path.join(BASE_DIR, "static")
        if (output_html.parent / "static").exists():
            shutil.rmtree(output_html.parent / "static")
        shutil.copytree(static_folder, output_html.parent / "static")

        # lib_folder = os.path.join(BASE_DIR, "lib")
        # if (output_html.parent/"lib").exists():
        #     shutil.rmtree(output_html.parent/"lib")
        # shutil.copytree(lib_folder, output_html.parent/"lib")

        print("Saved:", output_html)

    except Exception as e:
        raise BuildError(f"Failed to build visualization: {e}") from e
