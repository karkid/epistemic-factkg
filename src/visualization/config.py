"""
Configuration settings for the knowledge graph visualizer.
Now uses a theme-based approach for easy color customization.
"""

from .themes import get_theme

def get_network_config(theme_name='light'):
    """Get network configuration for specified theme."""
    theme = get_theme(theme_name)
    
    return {
        "height": "900px",
        "width": "100%",
        "bgcolor": theme['canvas'],
        "font_color": theme['text_primary']
    }

def get_physics_options(theme_name='light'):
    """Get physics options for specified theme."""
    theme = get_theme(theme_name)
    
    return {
        "configure": {"enabled": False},
        "physics": {
            "enabled": True,
            "stabilization": {"iterations": 100, "updateInterval": 10},
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.1,
                "springLength": 200,
                "springConstant": 0.02,
                "damping": 0.5
            }
        },
        "layout": {
            "improvedLayout": False,
            "hierarchical": {"enabled": False}
        },
        "interaction": {
            "hover": True,
            "selectConnectedEdges": False,
            "hideEdgesOnDrag": False
        },
        "nodes": {
            "font": {"size": 12, "color": theme['text_primary']},
            "borderWidth": 2,
            "physics": True
        },
        "edges": {
            "font": {"size": 12, "color": theme['edge_text']},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "width": 2,
            "physics": True,
            "smooth": {"enabled": False},
            "color": {"color": theme['edge_color'], "highlight": theme['edge_highlight']}
        }
    }

def get_node_styles(theme_name='light'):
    """Get node styling configuration for specified theme."""
    theme = get_theme(theme_name)
    
    return {
        'entity': {
            'color': {
                'background': theme['entity_bg'],
                'border': theme['entity_border'],
                'highlight': {
                    'background': theme['entity_hover_bg'],
                    'border': theme['entity_hover_border']
                }
            },
            'shape': 'dot',
            'size': 15,
            'font': {'color': theme['entity_text']}
        },
        'entities': {
            'color': {
                'background': theme['entity_bg'],
                'border': theme['entity_border'],
                'highlight': {
                    'background': theme['entity_hover_bg'],
                    'border': theme['entity_hover_border']
                }
            },
            'shape': 'dot',
            'size': 15,
            'font': {'color': theme['entity_text']}
        },
        'scene': {
            'color': {'background': theme['scene_bg'], 'border': theme['scene_border']},
            'shape': 'box',
            'size': 100,
            'borderWidth': 3,
            'font': {'size': 14, 'color': theme['scene_text']}
        },
        'scenes': {
            'color': {'background': theme['scene_bg'], 'border': theme['scene_border']},
            'shape': 'box',
            'size': 100,
            'borderWidth': 3,
            'font': {'size': 14, 'color': theme['scene_text']}
        },
        'floor': {
            'color': {'background': theme['floor_bg'], 'border': theme['floor_border']},
            'shape': 'box',
            'size': 200,
            'borderWidth': 4,
            'font': {'size': 16, 'color': theme['floor_text'], 'bold': True}
        },
        'floors': {
            'color': {'background': theme['floor_bg'], 'border': theme['floor_border']},
            'shape': 'box',
            'size': 200,
            'borderWidth': 4,
            'font': {'size': 16, 'color': theme['floor_text'], 'bold': True}
        },
        'relation': {
            'color': {'background': theme['relation_bg'], 'border': theme['relation_border']},
            'shape': 'box',
            'size': 12,
            'font': {'color': theme['relation_text']}
        },
        'literals': {
            'color': {'background': theme['literal_bg'], 'border': theme['literal_border']},
            'shape': 'ellipse',
            'size': 10,
            'font': {'color': theme['literal_text']}
        }
    }

def get_edge_styles(theme_name='light'):
    """Get edge styling configuration for specified theme."""
    theme = get_theme(theme_name)
    
    return {
        'default': {
            'color': theme['edge_color'],
            'width': 2
        },
        'preview': {
            'color': theme['edge_highlight'],
            'width': 1,
            'dashes': [5, 5]
        },
        'selected': {
            'color': theme['selection_bg'],
            'width': 3
        }
    }

# Backwards compatibility - use light theme by default
NETWORK_CONFIG = get_network_config('light')
PHYSICS_OPTIONS = get_physics_options('light')
NODE_STYLES = get_node_styles('light')
EDGE_STYLES = get_edge_styles('light')
