"""
Theme definitions for the knowledge graph visualizer.
Easy theme switching and experimentation.
"""

THEMES = {
    'light': {
        'name': 'Light Theme',
        'canvas': "#D8E2F4",
        'background': '#ffffff',
        'text_primary': '#000000',
        'text_secondary': '#000000',
        'text_light': '#333333',
        'text_shadow': '1px 1px 2px rgba(255,255,255,0.8), -1px -1px 2px rgba(255,255,255,0.8)',
        
        # Nodes
        'entity_bg': '#3b82f6',
        'entity_border': '#2563eb',
        'entity_hover_bg': '#60a5fa',
        'entity_hover_border': '#1d4ed8',
        'entity_text': '#1f2937',
        
        'scene_bg': 'rgba(59, 130, 246, 0.15)',
        'scene_border': '#3b82f6',
        'scene_text': '#3b82f6',
        
        'floor_bg': 'rgba(16, 185, 129, 0.15)',
        'floor_border': '#10b981',
        'floor_text': '#10b981',
        
        'relation_bg': '#f59e0b',
        'relation_border': '#d97706',
        'relation_text': '#ffffff',
        
        'literal_bg': '#84cc16',
        'literal_border': '#65a30d',
        'literal_text': '#ffffff',
        
        # Edges
        'edge_color': '#d1d5db',
        'edge_highlight': '#3b82f6',
        'edge_text': '#6b7280',
        
        # Selection
        'selection_bg': '#ef4444',
        'selection_border': '#dc2626',
        
        # UI
        'panel_bg': 'rgba(255, 255, 255, 0.95)',
        'panel_border': '#e5e7eb',
        'panel_header_bg': 'rgba(249, 250, 251, 0.8)',
        
        'button_bg': 'rgba(249, 250, 251, 0.95)',
        'button_border': '#e5e7eb',
        'button_text': '#374151',
        'button_hover_bg': 'rgba(59, 130, 246, 0.8)',
        'button_hover_border': '#3b82f6',
        'button_hover_text': '#ffffff',
        'button_active_bg': '#3b82f6',
        
        'tree_bg': 'rgba(243, 244, 246, 0.5)',
        'tree_border': '#e5e7eb',
        'tree_hover': 'rgba(59, 130, 246, 0.1)',
        
        'stats_bg': 'rgba(243, 244, 246, 0.5)',
        'stats_border': '#e5e7eb'
    },
    
'dark': {
    'name': 'Dark Theme',
    'canvas': "#8592a8",
    'background': '#2d3748',
    
    'text_primary': '#ffffff',
    'text_secondary': '#ffffff', 
    'text_light': '#ffffff',
    'text_shadow': '2px 2px 4px rgba(0,0,0,1), -2px -2px 4px rgba(0,0,0,1), 0px 0px 6px rgba(0,0,0,1)',

    # Nodes - High contrast with bright colors
    'entity_bg': '#4299e1',              # Bright blue
    'entity_border': '#2b77cb',          
    'entity_hover_bg': '#63b3ed',
    'entity_hover_border': '#3182ce',
    'entity_text': '#ffffff',

    'scene_bg': 'rgba(156, 163, 175, 0.4)',   
    'scene_border': '#9ca3af',
    'scene_text': '#ffffff',

    'floor_bg': 'rgba(52, 211, 153, 0.4)',    
    'floor_border': '#34d399',
    'floor_text': '#ffffff',

    'relation_bg': '#f6ad55',            # Orange
    'relation_border': '#ed8936',
    'relation_text': '#000000',          # Black text on orange

    'literal_bg': '#68d391',             # Green  
    'literal_border': '#48bb78',
    'literal_text': '#000000',           # Black text on green

    # Edges
    'edge_color': '#a0aec0',
    'edge_highlight': '#4299e1',
    'edge_text': '#ffffff',

    # Selection
    'selection_bg': '#f56565',
    'selection_border': '#e53e3e',

    # UI - Clean dark theme
    'panel_bg': 'rgba(45, 55, 72, 0.95)',
    'panel_border': '#718096',
    'panel_header_bg': 'rgba(26, 32, 44, 0.95)',

    'button_bg': 'rgba(74, 85, 104, 0.9)',
    'button_border': '#a0aec0', 
    'button_text': '#ffffff',

    'button_hover_bg': 'rgba(66, 153, 225, 0.95)',
    'button_hover_border': '#4299e1',
    'button_hover_text': '#ffffff',

    'button_active_bg': '#4299e1',

    'tree_bg': 'rgba(74, 85, 104, 0.85)',
    'tree_border': '#a0aec0',
    'tree_hover': 'rgba(66, 153, 225, 0.3)',

    'stats_bg': 'rgba(74, 85, 104, 0.85)',
    'stats_border': '#a0aec0'
},

            
    'monochrome_light': {
        'name': 'Light Monochrome',
        'canvas': '#f5f5f5',
        'background': '#f5f5f5',
        'text_primary': '#333333',
        'text_secondary': '#666666',
        'text_light': '#999999',
        'text_shadow': '1px 1px 2px rgba(255,255,255,0.8), -1px -1px 2px rgba(255,255,255,0.8)',
        
        # Nodes
        'entity_bg': '#666666',
        'entity_border': '#404040',
        'entity_hover_bg': '#525252',
        'entity_hover_border': '#333333',
        'entity_text': '#ffffff',
        
        'scene_bg': 'rgba(102, 102, 102, 0.2)',
        'scene_border': '#666666',
        'scene_text': '#666666',
        
        'floor_bg': 'rgba(128, 128, 128, 0.2)',
        'floor_border': '#808080',
        'floor_text': '#555555',
        
        'relation_bg': '#808080',
        'relation_border': '#666666',
        'relation_text': '#ffffff',
        
        'literal_bg': '#999999',
        'literal_border': '#777777',
        'literal_text': '#ffffff',
        
        # Edges
        'edge_color': '#999999',
        'edge_highlight': '#666666',
        'edge_text': '#666666',
        
        # Selection
        'selection_bg': '#333333',
        'selection_border': '#222222',
        
        # UI
        'panel_bg': 'rgba(255, 255, 255, 0.95)',
        'panel_border': '#e0e0e0',
        'panel_header_bg': 'rgba(248, 248, 248, 0.8)',
        
        'button_bg': 'rgba(248, 248, 248, 0.95)',
        'button_border': '#e0e0e0',
        'button_text': '#333333',
        'button_hover_bg': 'rgba(102, 102, 102, 0.8)',
        'button_hover_border': '#666666',
        'button_hover_text': '#ffffff',
        'button_active_bg': '#666666',
        
        'tree_bg': 'rgba(240, 240, 240, 0.5)',
        'tree_border': '#e0e0e0',
        'tree_hover': 'rgba(102, 102, 102, 0.1)',
        
        'stats_bg': 'rgba(240, 240, 240, 0.5)',
        'stats_border': '#e0e0e0'
    },
    
    'monochrome_dark': {
        'name': 'Dark Monochrome',
        'canvas': '#000000',
        'background': '#000000',
        'text_primary': '#ffffff',
        'text_secondary': '#ffffff',
        'text_light': '#ffffff',
        'text_shadow': '2px 2px 4px rgba(0,0,0,1), -2px -2px 4px rgba(0,0,0,1), 0px 0px 8px rgba(0,0,0,1)',
        
        # Nodes
        'entity_bg': '#666666',
        'entity_border': '#ffffff',
        'entity_hover_bg': '#888888',
        'entity_hover_border': '#ffffff',
        'entity_text': '#ffffff',
        
        'scene_bg': 'rgba(255, 255, 255, 0.3)',
        'scene_border': '#ffffff',
        'scene_text': '#ffffff',
        
        'floor_bg': 'rgba(255, 255, 255, 0.3)',
        'floor_border': '#ffffff',
        'floor_text': '#ffffff',
        
        'relation_bg': '#525252',
        'relation_border': '#404040',
        'relation_text': '#ffffff',
        
        'literal_bg': '#666666',
        'literal_border': '#525252',
        'literal_text': '#ffffff',
        
        # Edges
        'edge_color': '#525252',
        'edge_highlight': '#737373',
        'edge_text': '#a3a3a3',
        
        # Selection
        'selection_bg': '#d4d4d4',
        'selection_border': '#a3a3a3',
        
        # UI
        'panel_bg': 'rgba(38, 38, 38, 0.95)',
        'panel_border': '#525252',
        'panel_header_bg': 'rgba(17, 17, 17, 0.8)',
        
        'button_bg': 'rgba(64, 64, 64, 0.95)',
        'button_border': '#404040',
        'button_text': '#d4d4d4',
        'button_hover_bg': 'rgba(115, 115, 115, 0.8)',
        'button_hover_border': '#737373',
        'button_hover_text': '#ffffff',
        'button_active_bg': '#737373',
        
        'tree_bg': 'rgba(64, 64, 64, 0.5)',
        'tree_border': '#525252',
        'tree_hover': 'rgba(115, 115, 115, 0.2)',
        
        'stats_bg': 'rgba(64, 64, 64, 0.5)',
        'stats_border': '#525252'
    }
}

def get_theme(theme_name='light'):
    """Get theme configuration by name."""
    return THEMES.get(theme_name, THEMES['light'])

def list_themes():
    """List all available theme names."""
    return list(THEMES.keys())

def get_theme_info():
    """Get theme names and descriptions."""
    return {name: config['name'] for name, config in THEMES.items()}