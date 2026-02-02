"""Build PyVis network from RDF graph data."""

from pathlib import Path
from rdflib import Graph, URIRef, RDF, RDFS, Literal
from pyvis.network import Network
import json
from .config import get_network_config, get_physics_options, get_node_styles, get_edge_styles
from ..knowledge_graph.core.namespaces import BASE, ENTITIES, RELATIONS, SCENES


class NetworkBuilder:
    """Handles building PyVis networks from RDF data."""
    
    def __init__(self, theme='light'):
        self.graph = Graph()
        self.theme = theme
        self.nodes = {}
        self.edges = []
        self.literal_count = 0
        self.entity_count = 0
        self.node_types = {}
        self.node_positions = {}  # Store spatial positions
        
        # Relationships to hide from visualization
        self.hidden_relationships = {'position', 'rotation'}
        
        # Node types to filter out (noise reduction)
        self.filtered_node_types = {'Floor', 'FloorLamp'}
    
    def load_graph(self, graph_file: Path) -> None:
        """Load RDF graph from file."""
        # Auto-detect format
        if graph_file.suffix.lower() == '.jsonld':
            rdf_format = 'json-ld'
        elif graph_file.suffix.lower() in ['.ttl', '.turtle']:
            rdf_format = 'turtle'
        elif graph_file.suffix.lower() in ['.rdf', '.xml']:
            rdf_format = 'xml'
        else:
            rdf_format = 'turtle'  # default
        
        print(f"Loading graph from {graph_file} (format: {rdf_format})")
        self.graph.parse(graph_file, format=rdf_format)
        print(f"Graph loaded with {len(self.graph)} triples")
    
    def _determine_node_type(self, uri_str: str) -> str:
        """Determine the type of a node based on its URI and context."""
        uri = URIRef(uri_str)
        
        # Check if it's a scene
        if 'FloorPlan' in uri_str or 'Scene' in uri_str:
            return 'scene'
        
        # Check if it's an entity (has rdf:type)
        for _, _, obj in self.graph.triples((uri, RDF.type, None)):
            return 'entity'
        
        # Check if it appears as subject in object properties
        for pred, obj in self.graph.predicate_objects(uri):
            if isinstance(obj, URIRef):
                return 'entity'
        
        # Default to entity if it's a URI
        return 'entity'
    
    def _should_filter_node(self, uri_str: str, node_type: str) -> bool:
        """Check if node should be filtered out from visualization."""
        # Don't filter scene nodes - we need them for the dropdown
        if node_type == 'scene' or 'FloorPlan' in uri_str:
            return False
            
        # Filter out noisy node types
        for filtered_type in self.filtered_node_types:
            if filtered_type in uri_str:
                return True
        return False
    
    def _extract_nodes_and_edges(self, show_literals: bool = True, max_nodes: int = None) -> None:
        """Extract nodes and edges from the RDF graph."""
        self.nodes = {}
        self.edges = []
        
        # First pass: extract position data for spatial layout
        for subj, pred, obj in self.graph:
            subj_str = str(subj)
            pred_str = str(pred).split('#')[-1].split('/')[-1]
            
            if pred_str == 'position' and isinstance(obj, Literal):
                # Parse position string like "(-4.3, 0.01, -0.38)"
                try:
                    pos_str = str(obj).strip('()')
                    x, y, z = map(float, pos_str.split(', '))
                    self.node_positions[subj_str] = {'x': x, 'z': z}  # Use x,z for 2D layout
                except:
                    pass  # Skip malformed position data
        
        # Second pass: collect node types from RDF type relationships
        node_types = {}
        for subj, pred, obj in self.graph:
            if str(pred).endswith('type') or 'type' in str(pred).lower():
                if isinstance(obj, URIRef):
                    obj_label = str(obj).split('#')[-1].split('/')[-1]
                    node_types[str(subj)] = obj_label
        
        # Process all triples with filtering
        for subj, pred, obj in self.graph:
            subj_str = str(subj)
            pred_str = str(pred).split('#')[-1].split('/')[-1]  # Get local name
            
            # Skip type relationships as edges - we'll use them for node labels instead
            if pred_str.lower() == 'type' or 'type' in pred_str.lower():
                continue
                
            # Skip hidden relationships (position, rotation) from edges
            if pred_str in self.hidden_relationships:
                continue
            
            # Add subject node (with filtering)
            if subj_str not in self.nodes:
                node_type = self._determine_node_type(subj_str)
                # Filter out noisy node types
                if not self._should_filter_node(subj_str, node_type):
                    self.nodes[subj_str] = node_type
            
            if isinstance(obj, URIRef):
                # Object is another resource
                obj_str = str(obj)
                if obj_str not in self.nodes:
                    obj_node_type = self._determine_node_type(obj_str)
                    # Filter out noisy node types
                    if not self._should_filter_node(obj_str, obj_node_type):
                        self.nodes[obj_str] = obj_node_type
                
                # Only add edge if both nodes are included
                if subj_str in self.nodes and obj_str in self.nodes:
                    self.edges.append((subj_str, pred_str, obj_str))
            else:
                # Object is a literal - only add if subject node is included
                if subj_str in self.nodes and show_literals:
                    obj_str = str(obj)
                    self.edges.append((subj_str, pred_str, obj_str))
                    self.literal_count += 1
        
        # Store node types for labeling
        self.node_types = node_types
        
        # Count entities
        self.entity_count = len([n for n, t in self.nodes.items() if t == 'entity'])
        
        # Apply max_nodes limit if specified
        if max_nodes and self.entity_count > max_nodes:
            self._limit_nodes(max_nodes)
        
        print(f"Extracted {len(self.nodes)} nodes and {len(self.edges)} edges (type relationships filtered out)")
    
    def _limit_nodes(self, max_nodes: int) -> None:
        """Limit the number of nodes to the most connected entities."""
        # Count connections for each entity
        entity_connections = {}
        for subj, pred, obj in self.edges:
            if self.nodes.get(subj) == 'entity':
                entity_connections[subj] = entity_connections.get(subj, 0) + 1
            if obj in self.nodes and self.nodes.get(obj) == 'entity':
                entity_connections[obj] = entity_connections.get(obj, 0) + 1
        
        # Get top connected entities
        top_entities = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_entity_uris = set(uri for uri, _ in top_entities)
        
        # Filter nodes and edges
        self.nodes = {uri: node_type for uri, node_type in self.nodes.items() 
                     if node_type != 'entity' or uri in top_entity_uris}
        
        filtered_edges = []
        for subj, pred, obj in self.edges:
            if (subj in self.nodes and 
                (obj in self.nodes or isinstance(obj, str) and obj not in self.nodes)):
                filtered_edges.append((subj, pred, obj))
        self.edges = filtered_edges
    
    def create_network(self, show_literals: bool = True, max_nodes: int = None) -> Network:
        """Create PyVis network with hierarchical container layout."""
        # Extract data
        self._extract_nodes_and_edges(show_literals, max_nodes)
        
        # Get theme-based configuration
        network_config = get_network_config(self.theme)
        physics_options = get_physics_options(self.theme)
        node_styles = get_node_styles(self.theme)
        
        # Create network
        net = Network(**network_config)
        net.show_buttons(filter_=False)  # Disable control buttons
        
        # Calculate hierarchical positions
        positions = self._calculate_container_positions()
        
        # Store container information for HTML overlays
        container_data = []
        
        # Add only entity nodes and collect container data
        for uri, node_type in self.nodes.items():
            # Use type from RDF if available, otherwise use URI local name
            if hasattr(self, 'node_types') and uri in self.node_types:
                label = self.node_types[uri]
            else:
                label = uri.split('#')[-1].split('/')[-1]  # Get local name
            
            # Enhance node type detection for containers
            if 'floorplan' in label.lower():
                enhanced_type = 'scene'  # FloorPlan* nodes are scenes
            elif 'floor' in label.lower():
                enhanced_type = 'floor'  # Floor nodes are floors  
            elif 'scene' in label.lower() or node_type == 'scene' or label.lower() == 'scene':
                enhanced_type = 'scene'
            else:
                enhanced_type = 'entity'
            
            # Use spatial position if available, otherwise use hierarchical position
            if uri in self.node_positions:
                position = {
                    'x': self.node_positions[uri]['x'] * 10,  # Scale for better visualization
                    'y': self.node_positions[uri]['z'] * 10   # Use z as y coordinate
                }
            else:
                position = positions.get(uri, {})
            
            if enhanced_type in ['floor', 'scene']:
                # Store container information for HTML overlay creation (don't add to network)
                container_data.append({
                    'id': uri,
                    'label': label,
                    'group': enhanced_type,
                    'x': position.get('x', 0),
                    'y': position.get('y', 0)
                })
                print(f"📦 Storing {enhanced_type} container: {label} (excluded from network)")
            else:
                # Only add entity nodes to the network
                style = node_styles.get(enhanced_type, node_styles['entity'])
                
                # Create clean style without conflicts
                clean_style = {}
                if 'color' in style:
                    clean_style['color'] = style['color']
                if 'shape' in style:
                    clean_style['shape'] = style['shape']
                if 'font' in style:
                    clean_style['font'] = style['font']
                if 'borderWidth' in style:
                    clean_style['borderWidth'] = style['borderWidth']
                
                # Add entity node with positioning
                net.add_node(
                    uri,
                    label=label,
                    group=enhanced_type,
                    x=position.get('x', 0),
                    y=position.get('y', 0),
                    size=style.get('size', 15),
                    physics=True,
                    **clean_style
                )
        
        # Store container data for later use
        net.container_data = container_data
        print(f"Container data prepared: {len(container_data)} containers (floor/scene nodes excluded from network)")
        
        # Don't add edges initially - clean focused start
        
        # Configure network options
        net.set_options(json.dumps(physics_options))
        
        return net
    
    def get_all_nodes_and_edges_data(self):
        """Get all nodes and edges data for JavaScript to use."""
        node_styles = get_node_styles(self.theme)
        all_nodes = []
        all_edges = []
        
        # Add all nodes (excluding floor and scene nodes)
        for uri, node_type in self.nodes.items():
            # Use type from RDF if available, otherwise use URI local name
            if hasattr(self, 'node_types') and uri in self.node_types:
                label = self.node_types[uri]
            else:
                label = uri.split('#')[-1].split('/')[-1]  # Get local name
                
            # Enhance node type detection for containers
            if 'floor' in label.lower() or 'floorplan' in label.lower():
                display_type = 'floor'
            elif node_type == 'scene' or 'scene' in label.lower():
                display_type = 'scene'
            else:
                display_type = 'entity'
            
            # Skip floor and scene nodes - they're shown as containers
            if display_type in ['floor', 'scene']:
                continue
                
            style = node_styles.get(display_type, node_styles['entity'])
            
            node_data = {
                'id': uri,
                'label': label,
                'group': display_type,
                **style
            }
            all_nodes.append(node_data)
        
        # Add all edges
        edge_id = 0
        for subj, pred, obj in self.edges:
            if obj in self.nodes:  # Resource edge
                edge_data = {
                    'id': f"edge_{edge_id}",
                    'from': subj,
                    'to': obj,
                    'label': pred
                }
                all_edges.append(edge_data)
            else:  # Literal edge
                # Create literal node
                literal_id = f"literal_{edge_id}"
                literal_node = {
                    'id': literal_id,
                    'label': str(obj)[:50] + ('...' if len(str(obj)) > 50 else ''),
                    'group': 'literals',
                    **node_styles['literals']
                }
                all_nodes.append(literal_node)
                
                # Create edge to literal
                edge_data = {
                    'id': f"edge_{edge_id}",
                    'from': subj,
                    'to': literal_id,
                    'label': pred
                }
                all_edges.append(edge_data)
            
            edge_id += 1
        
        return all_nodes, all_edges
    
    def _calculate_container_positions(self) -> dict:
        """Calculate hierarchical positions based on actual relationships."""
        positions = {}
        
        # Find spatial relationships
        floor_scenes = {}  # floor -> [scenes]
        scene_objects = {}  # scene -> [objects] 
        
        for subj, pred, obj in self.edges:
            if pred.lower() == 'inscene':
                # Object is in scene
                if obj not in scene_objects:
                    scene_objects[obj] = []
                scene_objects[obj].append(subj)
            elif pred.lower() == 'hasobject':
                # Scene/Floor has object
                if subj not in scene_objects:
                    scene_objects[subj] = []
                scene_objects[subj].append(obj)
        
        # Find floors and scenes from nodes
        floors = []
        scenes = []
        entities = []
        
        for uri, node_type in self.nodes.items():
            label = uri.split('#')[-1].split('/')[-1]
            if 'floor' in label.lower() or 'floorplan' in label.lower():
                floors.append((uri, label))
            elif node_type == 'scene' or 'scene' in label.lower():
                scenes.append((uri, label))
            elif node_type == 'entity':
                entities.append((uri, label))
        
        # Layout floors as large containers
        floor_spacing = 800
        floor_size = 600
        for i, (floor_uri, floor_label) in enumerate(floors):
            positions[floor_uri] = {
                'x': i * floor_spacing,
                'y': 0,
                'width': floor_size,
                'height': 400
            }
        
        # Layout scenes inside floors
        scene_size = 250
        for i, (scene_uri, scene_label) in enumerate(scenes):
            # Find which floor this scene belongs to (if any)
            floor_index = i % max(1, len(floors))  # Distribute scenes across floors
            floor_x = positions.get(floors[floor_index][0], {}).get('x', 0) if floors else 0
            
            scene_x = floor_x + (i % 2) * (scene_size + 50) - floor_size//4
            scene_y = -50
            
            positions[scene_uri] = {
                'x': scene_x,
                'y': scene_y,
                'width': scene_size,
                'height': 200
            }
        
        # Layout entities inside their scenes based on relationships
        for scene_uri, objects in scene_objects.items():
            if scene_uri in positions:
                scene_pos = positions[scene_uri]
                scene_x, scene_y = scene_pos['x'], scene_pos['y']
                
                # Arrange objects in a grid within the scene
                objects_per_row = 4
                for i, obj_uri in enumerate(objects):
                    if obj_uri in [uri for uri, _ in entities]:  # Only position entities
                        row = i // objects_per_row
                        col = i % objects_per_row
                        
                        obj_x = scene_x + (col - objects_per_row/2) * 40
                        obj_y = scene_y + 150 + row * 40
                        
                        positions[obj_uri] = {
                            'x': obj_x,
                            'y': obj_y
                        }
        
        # Position remaining entities around floors
        positioned_entities = set(positions.keys())
        remaining_entities = [(uri, label) for uri, label in entities if uri not in positioned_entities]
        
        for i, (entity_uri, entity_label) in enumerate(remaining_entities):
            floor_index = i % max(1, len(floors))
            floor_x = positions.get(floors[floor_index][0], {}).get('x', 0) if floors else 0
            
            # Position around the floor
            angle = (i * 2 * 3.14159) / max(1, len(remaining_entities))
            radius = 300
            entity_x = floor_x + radius * 0.8 * (i % 6 - 3) / 3
            entity_y = 300 + (i // 6) * 60
            
            positions[entity_uri] = {
                'x': entity_x,
                'y': entity_y
            }
        
        return positions

    def get_statistics(self) -> dict:
        """Get statistics about the processed graph."""
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        
        # Count by type
        node_types = {}
        for node_type in self.nodes.values():
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Get relation types
        relation_types = set(pred for _, pred, _ in self.edges)
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'entity_nodes': self.entity_count,
            'literals': self.literal_count,
            'node_types': node_types,
            'relation_types': sorted(relation_types),
            'unique_relations': len(relation_types)
        }
