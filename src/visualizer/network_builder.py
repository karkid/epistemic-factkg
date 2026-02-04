from pathlib import Path
from rdflib import Graph, Literal
from pyvis.network import Network


class NetworkBuilder:
    """
    RDF → Interactive Knowledge Graph Builder
    Using rdflib + pyvis (vis.js)
    """

    # =================================================
    # Init
    # =================================================

    def __init__(self):

        self.graph = Graph()

        self.nodes = set()
        self.max_nodes = 500

        # Filters
        self.hidden_nodes = set(["Floor"])
        self.hidden_relations = set(["type"])

        self.show_instances = True
        self.show_classes = True
        self.show_literals = True  # Show literal values as nodes

        self.color_map = {
            "True": "#22c55e",
            "False": "#ef4444",
            "Hot": "#ef4444",
            "Cold": "#3b82f6",
            "RoomTemp": "#23e0ea",
            "Metal": "#9ca3af",
            "Wood": "#9ca3af",
            "Plastic": "#9ca3af",
            "Glass": "#9ca3af",
            "Ceramic": "#9ca3af",
            "Stone": "#9ca3af",
            "Fabric": "#9ca3af",
            "Rubber": "#9ca3af",
            "Food": "#9ca3af",
            "Paper": "#9ca3af",
            "Wax": "#9ca3af",
            "Soap": "#9ca3af",
            "Sponge": "#9ca3af",
            "Organic": "#9ca3af",
        }

    # =================================================
    # Public API
    # =================================================

    def load_graph(self, graph_file: Path) -> None:
        """Load RDF graph"""

        suffix = graph_file.suffix.lower()

        if suffix == ".jsonld":
            fmt = "json-ld"

        elif suffix in [".ttl", ".turtle"]:
            fmt = "turtle"

        elif suffix in [".rdf", ".xml"]:
            fmt = "xml"

        else:
            raise ValueError(f"Unsupported format: {suffix}")

        self.graph.parse(graph_file, format=fmt)

    def set_filters(
        self,
        show_instances: bool = True,
        show_classes: bool = True,
        show_literals: bool = True,
        max_nodes: int = 500,
    ):

        self.show_instances = show_instances
        self.show_classes = show_classes
        self.show_literals = show_literals
        self.max_nodes = max_nodes

    def hide_relations(self, relations):

        if isinstance(relations, str):
            self.hidden_relations.add(relations)

        else:
            self.hidden_relations.update(relations)

    def hide_nodes(self, nodes):

        if isinstance(nodes, str):
            self.hidden_nodes.add(nodes)

        else:
            self.hidden_nodes.update(nodes)

    def build(self) -> Network:
        """Build PyVis network"""

        net = self._create_network()

        self._process_triples(net)

        return net

    # =================================================
    # Core Pipeline
    # =================================================

    def _should_process_triple(self, s: str, p: str, o: str) -> bool:
        """Determine if a triple should be processed based on filters"""

        if len(self.nodes) >= self.max_nodes:
            return False

        # Skip literals unless show_literals is enabled
        if isinstance(o, Literal) and not self.show_literals:
            return False

        s_label = self._short_name(str(s))
        o_label = self._short_name(str(o))
        p_label = self._short_name(str(p))

        # Check hidden nodes
        if s_label in self.hidden_nodes or o_label in self.hidden_nodes:
            return False

        # Check hidden relations
        if p_label in self.hidden_relations:
            return False

        # Check instance/class filters
        s_is_instance = self._is_instance(s_label)
        o_is_instance = self._is_instance(o_label)

        if (s_is_instance and not self.show_instances) or (
            o_is_instance and not self.show_instances
        ):
            return False

        return True

    def _process_triples(self, net: Network):

        count = 0

        for s, p, o in self.graph:
            if not self._should_process_triple(s, p, o):
                continue

            if count >= self.max_nodes:
                break

            self._add_edge(net, s, o, p)

            count += 1

    # =================================================
    # Node Handling
    # =================================================

    def _add_literal_node(self, net: Network, p, literal: Literal):
        """Add a literal value node to the network"""

        node_id = str(literal)

        if node_id in self.nodes:
            return

        label = str(literal)

        # Special styling for literals
        style = self._literal_style(literal)
        value_label = self._short_name(p)

        net.add_node(
            node_id,
            id=node_id,
            label="",  # Empty label - only show on hover
            title=f"{value_label}: {label}",
            color=style["color"],
            size=style["size"],
            shape=style["shape"],
            font=style["font"],
        )

        self.nodes.add(node_id)

    def _add_node(self, net: Network, node):

        node_id = str(node)

        if node_id in self.nodes:
            return

        # Handle URI nodes (existing logic)
        label = self._short_name(node_id)

        if label in self.hidden_nodes:
            return

        is_instance = self._is_instance(label)

        if (is_instance and not self.show_instances) or (
            not is_instance and not self.show_classes
        ):
            return

        display = self._clean_label(label)[:30]

        style = self._node_style(label)

        net.add_node(
            node_id,
            id=node_id,
            label=display,
            title=f"{label}\n{node_id}",
            color=style["color"],
            size=style["size"],
            shape=style["shape"],
            font=style["font"],
        )

        self.nodes.add(node_id)

    # =================================================
    # Edge Handling
    # =================================================

    def _add_edge(self, net: Network, s, o, p):

        # Add nodes first
        self._add_node(net, s)

        if isinstance(o, Literal):
            if self.show_literals:
                self._add_literal_node(net, p, o)
        else:
            self._add_node(net, o)

        # Check if both nodes were successfully added
        s_str = str(s)
        o_str = str(o)

        if s_str not in self.nodes or o_str not in self.nodes:
            return

        rel = self._short_name(str(p))

        if rel in self.hidden_relations:
            return

        style = self._edge_style(rel)

        net.add_edge(
            s_str,
            o_str,
            label=rel,
            color=style["color"],
            width=style["width"],
            font={"size": style["font_size"], "color": "#9ca3af"},
            arrows="to",
            smooth={"type": "dynamic"},
        )

    # =================================================
    # Styling
    # =================================================

    def _node_style(self, name: str) -> dict:

        name = name.lower()

        # Instance
        if self._is_instance(name):
            return {
                "color": "#38bdf8",
                "size": 18,
                "shape": "dot",
                "font": {"size": 10, "color": "#e5e7eb"},
            }

        # Room / Area
        if "room" in name or "area" in name:
            return {
                "color": "#34d399",
                "size": 22,
                "shape": "dot",
                "font": {"size": 11, "color": "#ecfeff"},
            }

        # Sensor
        if "sensor" in name:
            return {
                "color": "#fbbf24",
                "size": 20,
                "shape": "dot",
                "font": {"size": 10, "color": "#fefce8"},
            }

        # Default
        return {
            "color": "#818cf8",
            "size": 19,
            "shape": "dot",
            "font": {"size": 10, "color": "#eef2ff"},
        }

    def _literal_style(self, literal: Literal) -> dict:
        """Style for literal value nodes"""

        value_str = str(literal).lower()

        # Boolean values (states)
        if value_str in ["true", "false"]:
            color = "#22c55e" if value_str == "true" else "#ef4444"
            return {
                "color": color,
                "size": 15,
                "shape": "square",
                "font": {"size": 9, "color": "#ffffff"},
            }

        # Numeric values
        try:
            float(value_str)
            return {
                "color": "#f59e0b",
                "size": 16,
                "shape": "triangle",
                "font": {"size": 9, "color": "#fffbeb"},
            }
        except ValueError:
            pass

        # String values (like "RoomTemp")
        color = self.color_map.get(literal, "#6b7280")

        return {
            "color": color,
            "size": 14,
            "shape": "ellipse",
            "font": {"size": 9, "color": "#faf5ff"},
        }

    def _edge_style(self, rel: str) -> dict:

        r = rel.lower()

        if "type" in r:
            return {"color": "#f87171", "width": 1.5, "font_size": 8}

        if "subclass" in r:
            return {"color": "#c084fc", "width": 2, "font_size": 9}

        if "contain" in r or "located" in r:
            return {"color": "#34d399", "width": 1.5, "font_size": 8}

        return {"color": "#9ca3af", "width": 1, "font_size": 7}

    # =================================================
    # Network Setup
    # =================================================

    def _create_network(self) -> Network:

        net = Network(
            height="100vh",
            width="100vw",
            directed=True,
            notebook=False,
            bgcolor="#0f172a",
            font_color="#e5e7eb",
        )

        net.set_options("""
        {
          "nodes": {
            "shape": "dot",
            "labelHighlightBold": false,
            "font": {
              "strokeWidth": 0
            }
          },

          "edges": {
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          },

          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.15,
              "springLength": 140,
              "springConstant": 0.04,
              "damping": 0.25,
              "avoidOverlap": 0.6
            },
            "stabilization": {
              "enabled": true,
              "iterations": 300
            }
          },

          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "multiselect": true,
            "selectConnectedEdges": false
          },

          "layout": {
            "improvedLayout": true
          }
        }
        """)

        return net

    # =================================================
    # Utils
    # =================================================

    def _short_name(self, uri: str) -> str:

        return uri.split("/")[-1].split("#")[-1]

    def _clean_label(self, label: str) -> str:

        label = label.replace("%7C", "|").replace("%2B", "+").replace("%2D", "-")

        if "|" in label:
            parts = label.split("|")

            if len(parts) > 4:
                return parts[-1]

            return parts[0]

        return label

    def _is_instance(self, label: str) -> bool:

        return "|" in label or "floorplan" in label.lower()
