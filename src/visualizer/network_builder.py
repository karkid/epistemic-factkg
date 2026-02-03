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
        max_nodes: int = 500
    ):

        self.show_instances = show_instances
        self.show_classes = show_classes
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


        # Skip literals
        if isinstance(o, Literal):
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


        if (s_is_instance and not self.show_instances) or \
           (o_is_instance and not self.show_instances):
            return False
        
        return True
    
    def _process_triples(self, net: Network):

        count = 0

        for s, p, o in self.graph:

            if not self._should_process_triple(s, p, o):
                continue

            if count >= self.max_nodes:
                break


            # Skip literals
            if isinstance(o, Literal):
                continue


            s = str(s)
            p = str(p)
            o = str(o)


            self._add_node(net, s)
            self._add_node(net, o)

            self._add_edge(net, s, o, p)

            count += 1

    # =================================================
    # Node Handling
    # =================================================

    def _add_node(self, net: Network, uri: str):

        if uri in self.nodes:
            return


        if len(self.nodes) >= self.max_nodes:
            return


        label = self._short_name(uri)


        if label in self.hidden_nodes:
            return


        is_instance = self._is_instance(label)


        if (is_instance and not self.show_instances) or \
           (not is_instance and not self.show_classes):
            return


        display = self._clean_label(label)[:30]


        style = self._node_style(label)


        net.add_node(
            uri,
            id=uri,
            label=display,
            title=f"{label}\n{uri}",

            color=style["color"],
            size=style["size"],
            shape=style["shape"],

            font=style["font"]
        )


        self.nodes.add(uri)


    # =================================================
    # Edge Handling
    # =================================================

    def _add_edge(self, net: Network, s: str, o: str, p: str):

        if s not in self.nodes or o not in self.nodes:
            return


        rel = self._short_name(p)


        if rel in self.hidden_relations:
            return


        style = self._edge_style(rel)


        net.add_edge(
            s,
            o,

            label=rel,

            color=style["color"],
            width=style["width"],

            font={
                "size": style["font_size"],
                "color": "#9ca3af"
            },

            arrows="to",

            smooth={"type": "dynamic"}
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
                "font": {
                    "size": 10,
                    "color": "#e5e7eb"
                }
            }


        # Room / Area
        if "room" in name or "area" in name:

            return {
                "color": "#34d399",
                "size": 22,
                "shape": "dot",
                "font": {
                    "size": 11,
                    "color": "#ecfeff"
                }
            }


        # Sensor
        if "sensor" in name:

            return {
                "color": "#fbbf24",
                "size": 20,
                "shape": "dot",
                "font": {
                    "size": 10,
                    "color": "#fefce8"
                }
            }


        # Default
        return {
            "color": "#818cf8",
            "size": 19,
            "shape": "dot",
            "font": {
                "size": 10,
                "color": "#eef2ff"
            }
        }


    def _edge_style(self, rel: str) -> dict:

        r = rel.lower()


        if "type" in r:

            return {
                "color": "#f87171",
                "width": 1.5,
                "font_size": 8
            }


        if "subclass" in r:

            return {
                "color": "#c084fc",
                "width": 2,
                "font_size": 9
            }


        if "contain" in r or "located" in r:

            return {
                "color": "#34d399",
                "width": 1.5,
                "font_size": 8
            }


        return {
            "color": "#9ca3af",
            "width": 1,
            "font_size": 7
        }


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
            font_color="#e5e7eb"
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

        label = (
            label.replace("%7C", "|")
                 .replace("%2B", "+")
                 .replace("%2D", "-")
        )


        if "|" in label:

            parts = label.split("|")

            if len(parts) > 4:
                return parts[-1]

            return parts[0]


        return label


    def _is_instance(self, label: str) -> bool:

        return "|" in label or "floorplan" in label.lower()
