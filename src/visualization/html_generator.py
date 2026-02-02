"""Generate HTML with embedded JavaScript for focused interaction - Clean version."""

from .themes import get_theme


class HTMLGenerator:
    """Handles HTML generation and JavaScript injection."""
    
    def __init__(self, theme='light'):
        self.theme = theme
    
    @staticmethod
    def cleanup_pyvis_html(html_string: str) -> str:
        """Clean up PyVis-generated HTML by removing problematic references."""
        # Remove utils.js reference that doesn't exist
        html_string = html_string.replace(
            '<script src="lib/bindings/utils.js"></script>',
            ''
        )
        
        # Clean up any double line breaks
        html_string = html_string.replace('\n\n\n', '\n\n')
        
        return html_string
    
    def get_focused_interaction_js(self, all_nodes=None, all_edges=None, container_data=None) -> str:
        """Return JavaScript for focused entity interaction - Clean implementation."""
        import json
        
        # Get theme configuration
        theme = get_theme(self.theme)
        
        # Convert Python data to JavaScript
        nodes_js = "[]" if not all_nodes else json.dumps(all_nodes)
        edges_js = "[]" if not all_edges else json.dumps(all_edges)
        containers_js = "[]" if not container_data else json.dumps(container_data)
        
        # Embed all themes for client-side switching
        from .themes import THEMES
        themes_js = json.dumps(THEMES)
        current_theme = self.theme
        
        # Use simple string building to avoid template literal issues
        js_code = f"""
        <style>
            :root {{
                --bg-color: {theme['background']};
                --canvas-color: {theme['canvas']};
                --text-primary: {theme['text_primary']};
                --text-secondary: {theme['text_secondary']};
                --panel-bg: {theme['panel_bg']};
                --panel-border: {theme['panel_border']};
                --button-bg: {theme['button_bg']};
                --button-border: {theme['button_border']};
                --button-text: {theme['button_text']};
                --button-hover-bg: {theme['button_hover_bg']};
                --button-hover-border: {theme['button_hover_border']};
                --button-hover-text: {theme['button_hover_text']};
                --button-active-bg: {theme['button_active_bg']};
            }}
            /* Reset and override PyVis/Bootstrap styles */
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ margin: 0; padding: 0; overflow: hidden; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg-color); color: var(--text-primary); }}
            #mynetwork {{ 
                width: 100% !important; 
                height: 100vh !important; 
                background-color: var(--canvas-color) !important;
                border: none !important;
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                float: none !important;
            }}
            /* Ensure node labels are visible with stronger overrides */
            .vis-label {{
                color: {theme['text_primary']} !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                stroke: {theme['canvas']} !important;
                stroke-width: 2px !important;
                text-rendering: optimizeLegibility !important;
                font-family: Arial, sans-serif !important;
                text-anchor: middle !important;
                dominant-baseline: central !important;
            }}
            .vis-network .vis-label {{
                color: {theme['text_primary']} !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                text-rendering: optimizeLegibility !important;
                font-family: Arial, sans-serif !important;
                text-anchor: middle !important;
                dominant-baseline: central !important;
            }}
            /* Override any vis.js text styling */
            #mynetwork text {{
                fill: {theme['text_primary']} !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                text-rendering: optimizeLegibility !important;
                stroke: {theme['canvas']} !important;
                stroke-width: 2px !important;
                font-family: Arial, sans-serif !important;
            }}
            
            /* Force ALL text elements to be visible */
            #mynetwork * {{
                color: {theme['text_primary']} !important;
                font-weight: 600 !important;
                font-family: Arial, sans-serif !important;
            }}
            
            /* Canvas text rendering overrides */
            #mynetwork canvas {{
                font-weight: 600 !important;
                font-size: 14px !important;
                text-rendering: optimizeLegibility !important;
                font-family: Arial, sans-serif !important;
            }}
            
            /* SVG text elements */
            #mynetwork svg text {{
                fill: {theme['text_primary']} !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                font-family: Arial, sans-serif !important;
                text-anchor: middle !important;
                dominant-baseline: central !important;
            }}
            #loadingBar {{ display: none !important; }}
            .vis-loading {{ display: none !important; }}
            .container, .container-fluid, .row, .col {{ margin: 0 !important; padding: 0 !important; }}
            h1, h2, h3, h4, h5, h6 {{ margin: 0 !important; padding: 0 !important; }}
            
            /* Side Menu Toggle Button - Attached to left middle of sidebar */
            .kg-menu-toggle {{
                position: fixed;
                top: 50%;
                left: -2px;
                transform: translateY(-50%);
                width: 40px;
                height: 60px;
                border: 1px solid var(--button-border);
                border-left: none;
                background: var(--button-bg);
                backdrop-filter: blur(10px);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                color: var(--button-text);
                transition: all 0.3s ease;
                border-radius: 0 12px 12px 0;
                z-index: 3000;
                box-shadow: 2px 0 8px rgba(0,0,0,0.1);
            }}
            
            .kg-menu-toggle:hover {{
                background: var(--button-hover-bg);
                border-color: var(--button-hover-border);
                color: var(--button-hover-text);
                left: 0px;
                box-shadow: 4px 0 12px rgba(0,0,0,0.2);
            }}
            
            .kg-menu-toggle.active {{
                left: 358px;
                background: var(--button-active-bg);
                border-color: var(--button-active-bg);
                color: var(--button-hover-text);
                box-shadow: 0 0 0 2px rgba(115, 115, 115, 0.3);
            }}
            
            /* Bottom Right Zoom Controls */
            .kg-zoom-controls {{
                position: fixed;
                bottom: 32px;
                right: 32px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                z-index: 3000;
            }}
            
            /* Bottom Right Navigation Controls */
            .kg-nav-controls {{
                position: fixed;
                bottom: 32px;
                right: 100px;
                display: flex;
                flex-direction: row;
                gap: 8px;
                z-index: 3000;
            }}
            
            .kg-icon-btn {{
                width: 48px;
                height: 48px;
                border: 1px solid {theme['button_border']};
                background: {theme['button_bg']};
                backdrop-filter: blur(10px);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                color: {theme['button_text']};
                transition: all 0.2s ease;
                margin: 0;
                padding: 0;
                border-radius: 12px;
            }}
            
            .kg-icon-btn:hover {{
                background: {theme['button_hover_bg']};
                border-color: {theme['button_hover_border']};
                color: {theme['button_hover_text']};
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(115, 115, 115, 0.3);
            }}
            
            .kg-icon-btn.active {{
                background: {theme['button_active_bg']};
                border-color: {theme['button_active_bg']};
                color: {theme['button_hover_text']};
                box-shadow: 0 0 0 2px rgba(115, 115, 115, 0.3);
            }}
            
            /* Sidebar Panel */
            .kg-control-panel {{
                position: fixed;
                top: 0;
                left: -380px;
                background: var(--panel-bg);
                backdrop-filter: blur(20px);
                border-right: 1px solid var(--panel-border);
                width: 360px;
                height: 100vh;
                overflow: hidden;
                transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                z-index: 2500;
            }}
            
            .kg-control-panel.open {{ left: 0; }}
            
            .kg-panel-header {{
                padding: 24px;
                border-bottom: 1px solid {theme['panel_border']};
                background: {theme['panel_header_bg']};
            }}
            
            .kg-panel-content {{
                padding: 24px;
                height: calc(100vh - 120px);
                overflow-y: auto;
            }}
            
            .kg-control-title {{
                font-size: 20px;
                font-weight: 600;
                color: {theme['text_primary']};
                margin: 0;
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            
            .kg-control-group {{
                margin-bottom: 28px;
            }}
            
            .kg-control-group:last-child {{
                margin-bottom: 0;
            }}
            
            .kg-control-label {{
                font-size: 12px;
                font-weight: 600;
                color: {theme['text_secondary']};
                margin-bottom: 16px;
                display: block;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            /* Tree */
            .kg-tree {{
                background: {theme['tree_bg']};
                border-radius: 12px;
                padding: 20px;
                border: 1px solid {theme['tree_border']};
            }}
            
            .kg-tree-item {{
                margin: 6px 0;
                padding: 12px 16px;
                border-radius: 8px;
                transition: all 0.2s ease;
                cursor: pointer;
                user-select: none;
            }}
            
            .kg-tree-item:hover {{
                background: {theme['tree_hover']};
                transform: translateX(4px);
            }}
            
            .kg-tree-floor {{
                font-weight: 600;
                color: {theme['floor_text']};
                background: {theme['floor_bg']};
                border-left: 3px solid {theme['floor_border']};
            }}
            
            .kg-tree-scene {{
                font-weight: 500;
                color: {theme['scene_text']};
                background: {theme['scene_bg']};
                border-left: 3px solid {theme['scene_border']};
                margin-left: 20px;
            }}
            
            .kg-tree-entity {{
                color: {theme['text_secondary']};
                background: {theme['entity_bg'][:4] + '16'};
                border-left: 2px solid {theme['entity_border']};
                margin-left: 40px;
                font-size: 13px;
            }}
            
            /* Statistics */
            .kg-stats {{
                border: 1px solid {theme['stats_border']};
                background: {theme['stats_bg']};
                border-radius: 12px;
                overflow: hidden;
            }}
            
            .kg-stats-item {{
                display: flex;
                justify-content: space-between;
                padding: 12px 16px;
                border-bottom: 1px solid {theme['stats_border']};
                font-size: 13px;
                color: {theme['text_secondary']};
            }}
            
            .kg-stats-item:last-child {{ border-bottom: none; }}
            
            .kg-stats-value {{
                font-weight: 600;
                color: {theme['text_primary']};
            }}
            
            /* Theme Selector */
            .kg-theme-select {{
                width: 100%;
                padding: 8px 12px;
                border: 1px solid {theme['panel_border']};
                background: {theme['background']};
                color: {theme['text_primary']};
                border-radius: 6px;
                font-size: 13px;
                cursor: pointer;
            }}
            
            .kg-theme-select:hover {{
                border-color: {theme['button_hover_border']};
            }}
            
            .kg-theme-select:focus {{
                outline: none;
                border-color: {theme['button_active_bg']};
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
            }}
        </style>
        
        <script type="text/javascript">
            (function() {{
                console.log('🚀 Starting focused KG initialization...');
                
                let kgAllNodesData = {nodes_js};
                let kgAllEdgesData = {edges_js};
                let kgContainerData = {containers_js};
                let kgThemes = {themes_js};
                let kgCurrentTheme = '{current_theme}';
                
                console.log('📊 Loaded data:', kgAllNodesData.length, 'nodes,', kgAllEdgesData.length, 'edges,', kgContainerData.length, 'containers');
                
                let kgOriginalNodes = null;
                let kgOriginalEdges = null;
                let kgSelectedEntities = new Set();
                let kgInitialized = false;
                let kgBaseNodes = [];
                let kgBaseEdges = [];                
                // Define theme switching function globally
                window.applyTheme = function(themeName) {{
                    const theme = kgThemes[themeName];
                    if (!theme) {{
                        console.error('Theme not found:', themeName);
                        return;
                    }}
                    
                    console.log('🎨 Applying theme:', theme.name);
                    
                    // Update CSS custom properties
                    const root = document.documentElement;
                    root.style.setProperty('--bg-color', theme.background);
                    root.style.setProperty('--canvas-color', theme.canvas);
                    root.style.setProperty('--text-primary', theme.text_primary);
                    root.style.setProperty('--text-secondary', theme.text_secondary);
                    root.style.setProperty('--panel-bg', theme.panel_bg);
                    root.style.setProperty('--panel-border', theme.panel_border);
                    root.style.setProperty('--button-bg', theme.button_bg);
                    root.style.setProperty('--button-border', theme.button_border);
                    root.style.setProperty('--button-text', theme.button_text);
                    root.style.setProperty('--button-hover-bg', theme.button_hover_bg);
                    root.style.setProperty('--button-hover-border', theme.button_hover_border);
                    root.style.setProperty('--button-hover-text', theme.button_hover_text);
                    root.style.setProperty('--button-active-bg', theme.button_active_bg);
                    
                    // Update network background
                    const networkDiv = document.getElementById('mynetwork');
                    if (networkDiv) {{
                        networkDiv.style.backgroundColor = theme.canvas;
                    }}
                    
                    // Update body background  
                    document.body.style.backgroundColor = theme.background;
                    document.body.style.color = theme.text_primary;
                    
                    console.log('✅ Theme applied successfully');
                }};                
                function initializeFocusedMode() {{
                    console.log('🔧 Initializing focused mode...');
                    if (kgInitialized) {{
                        console.log('⚠️ Already initialized');
                        return;
                    }}
                    
                    if (typeof network === 'undefined' || !network) {{
                        console.log('❌ Network not ready yet');
                        return;
                    }}
                    
                    console.log('✅ Network is ready, proceeding...');
                    
                    try {{
                        if (kgAllNodesData.length > 0) {{
                            kgOriginalNodes = kgAllNodesData;
                            kgOriginalEdges = kgAllEdgesData;
                            kgBaseNodes = kgOriginalNodes.filter(node => node.group === 'entity');
                            kgBaseEdges = kgOriginalEdges;
                            console.log('📁 Using pre-loaded data');
                        }} else {{
                            console.log('📥 Getting data from network...');
                            kgOriginalNodes = nodes.get();
                            kgOriginalEdges = edges.get();
                        }}
                        
                        console.log('📊 Working with', kgOriginalNodes.length, 'nodes and', kgOriginalEdges.length, 'edges');
                        
                        if (kgOriginalNodes.length === 0) {{
                            console.error('❌ No nodes found! Cannot initialize.');
                            return;
                        }}
                        
                        // Start with clean entity-only view
                        const entityNodes = kgOriginalNodes.filter(n => 
                            n.group === 'entity' || n.group === 'entities');
                        
                        nodes.clear();
                        edges.clear();
                        nodes.add(entityNodes);
                        
                        console.log('✅ Clean start: showing', entityNodes.length, 'entity nodes');
                        
                        // Wait for nodes to stabilize before fitting
                        setTimeout(function() {{
                            network.fit({{
                                animation: {{
                                    duration: 1000,
                                    easingFunction: 'easeInOutQuad'
                                }}
                            }});
                        }}, 200);
                        
                        createUI();
                        updateStats();
                        
                        // Set up event handlers
                        network.on('click', handleKgNodeClick);
                        
                        kgInitialized = true;
                        console.log('✅ Focused mode initialized successfully');
                        
                    }} catch (error) {{
                        console.error('💥 Failed to initialize focused mode:', error);
                    }}
                }}
                
                function createUI() {{
                    // Create side menu toggle button
                    const menuToggle = document.createElement('button');
                    menuToggle.className = 'kg-menu-toggle';
                    menuToggle.id = 'kg-toggle-panel';
                    menuToggle.title = 'Toggle Menu';
                    menuToggle.textContent = '☰';
                    document.body.appendChild(menuToggle);
                    
                    // Create bottom right zoom controls
                    const zoomControls = document.createElement('div');
                    zoomControls.className = 'kg-zoom-controls';
                    
                    const zoomInBtn = document.createElement('button');
                    zoomInBtn.className = 'kg-icon-btn';
                    zoomInBtn.title = 'Zoom In';
                    zoomInBtn.textContent = '+';
                    zoomInBtn.onclick = function() {{ 
                        const scale = network.getScale() * 1.2;
                        network.moveTo({{scale: scale}});
                    }};
                    
                    const zoomOutBtn = document.createElement('button');
                    zoomOutBtn.className = 'kg-icon-btn';
                    zoomOutBtn.title = 'Zoom Out';
                    zoomOutBtn.textContent = '−';
                    zoomOutBtn.onclick = function() {{ 
                        const scale = network.getScale() / 1.2;
                        network.moveTo({{scale: scale}});
                    }};
                    
                    const fitBtn = document.createElement('button');
                    fitBtn.className = 'kg-icon-btn';
                    fitBtn.title = 'Fit View';
                    fitBtn.textContent = '⌘';
                    fitBtn.onclick = function() {{ network.fit(); }};
                    
                    zoomControls.appendChild(zoomInBtn);
                    zoomControls.appendChild(zoomOutBtn);
                    zoomControls.appendChild(fitBtn);
                    document.body.appendChild(zoomControls);
                    
                    // Create bottom left navigation controls
                    const navControls = document.createElement('div');
                    navControls.className = 'kg-nav-controls';
                    
                    const centerBtn = document.createElement('button');
                    centerBtn.className = 'kg-icon-btn';
                    centerBtn.title = 'Center View';
                    centerBtn.textContent = '◎';
                    centerBtn.onclick = function() {{ 
                        network.moveTo({{position: {{x: 0, y: 0}}, scale: 1}});
                    }};
                    
                    const resetBtn = document.createElement('button');
                    resetBtn.className = 'kg-icon-btn';
                    resetBtn.title = 'Reset View';
                    resetBtn.textContent = '↻';
                    resetBtn.onclick = function() {{ 
                        console.log('🔄 Reset button clicked');
                        kgSelectedEntities.clear();
                        console.log('✅ Reset complete - calling updateVisibleNetwork');
                        updateVisibleNetwork();
                        setTimeout(() => network.fit(), 100);
                    }};
                    
                    navControls.appendChild(centerBtn);
                    navControls.appendChild(resetBtn);
                    document.body.appendChild(navControls);
                    
                    // Create control panel
                    const controlPanel = document.createElement('div');
                    controlPanel.className = 'kg-control-panel';
                    controlPanel.id = 'kg-main-panel';
                    
                    const panelHeader = document.createElement('div');
                    panelHeader.className = 'kg-panel-header';
                    const title = document.createElement('div');
                    title.className = 'kg-control-title';
                    title.textContent = 'Graph Explorer';
                    panelHeader.appendChild(title);
                    
                    const panelContent = document.createElement('div');
                    panelContent.className = 'kg-panel-content';
                    
                    // Hierarchy section
                    const hierarchyGroup = document.createElement('div');
                    hierarchyGroup.className = 'kg-control-group';
                    const hierarchyLabel = document.createElement('label');
                    hierarchyLabel.className = 'kg-control-label';
                    hierarchyLabel.textContent = 'Hierarchy';
                    const treeDiv = document.createElement('div');
                    treeDiv.className = 'kg-tree';
                    treeDiv.id = 'kg-hierarchy-tree';
                    treeDiv.innerHTML = '<div class="kg-tree-item">Loading...</div>';
                    
                    hierarchyGroup.appendChild(hierarchyLabel);
                    hierarchyGroup.appendChild(treeDiv);
                    
                    // Scene Filter section
                    const sceneGroup = document.createElement('div');
                    sceneGroup.className = 'kg-control-group';
                    const sceneLabel = document.createElement('label');
                    sceneLabel.className = 'kg-control-label';
                    sceneLabel.textContent = 'Scene Filter';
                    const sceneSelect = document.createElement('select');
                    sceneSelect.className = 'kg-theme-select';
                    sceneSelect.id = 'kg-scene-filter';
                    sceneSelect.innerHTML = '<option value="">All Scenes</option>';
                    
                    // Populate scene options from container data
                    if (kgContainerData && kgContainerData.length > 0) {{
                        const scenes = kgContainerData.filter(c => c.group === 'scene');
                        console.log('🎯 Found scenes for dropdown:', scenes);
                        scenes.forEach(function(scene) {{
                            const option = document.createElement('option');
                            option.value = scene.label;
                            option.textContent = scene.label;
                            sceneSelect.appendChild(option);
                        }});
                    }}
                    
                    // Also extract unique scenes from edge data as fallback
                    const sceneNames = new Set();
                    kgOriginalEdges.filter(edge => edge.label === 'inScene').forEach(edge => {{
                        if (edge.to) {{
                            const sceneName = edge.to.split('/').pop(); // Get last part of URI
                            if (sceneName.includes('FloorPlan')) {{
                                sceneNames.add(sceneName);
                            }}
                        }}
                    }});
                    
                    console.log('🔍 Found scene names from edges:', Array.from(sceneNames));
                    sceneNames.forEach(function(sceneName) {{
                        // Check if not already added
                        const exists = Array.from(sceneSelect.options).some(opt => opt.value === sceneName);
                        if (!exists) {{
                            const option = document.createElement('option');
                            option.value = sceneName;
                            option.textContent = sceneName;
                            sceneSelect.appendChild(option);
                        }}
                    }});
                    
                    sceneSelect.onchange = function() {{
                        const selectedScene = this.value;
                        console.log('🎯 Filtering by scene:', selectedScene);
                        filterByScene(selectedScene);
                    }};
                    
                    sceneGroup.appendChild(sceneLabel);
                    sceneGroup.appendChild(sceneSelect);
                    
                    // Stats section
                    const statsGroup = document.createElement('div');
                    statsGroup.className = 'kg-control-group';
                    const statsLabel = document.createElement('label');
                    statsLabel.className = 'kg-control-label';
                    statsLabel.textContent = 'Statistics';
                    const statsDiv = document.createElement('div');
                    statsDiv.className = 'kg-stats';
                    statsDiv.innerHTML = 
                        '<div class="kg-stats-item"><span>Entities</span><span class="kg-stats-value" id="kg-entity-count">-</span></div>' +
                        '<div class="kg-stats-item"><span>Visible</span><span class="kg-stats-value" id="kg-visible-count">-</span></div>' +
                        '<div class="kg-stats-item"><span>Edges</span><span class="kg-stats-value" id="kg-edge-count">-</span></div>' +
                        '<div class="kg-stats-item"><span>Selected</span><span class="kg-stats-value" id="kg-selected-count">0</span></div>';
                    
                    statsGroup.appendChild(statsLabel);
                    statsGroup.appendChild(statsDiv);
                    
                    // Theme switcher section
                    const themeGroup = document.createElement('div');
                    themeGroup.className = 'kg-control-group';
                    const themeLabel = document.createElement('label');
                    themeLabel.className = 'kg-control-label';
                    themeLabel.textContent = 'Theme';
                    const themeSelect = document.createElement('select');
                    themeSelect.className = 'kg-theme-select';
                    themeSelect.innerHTML = 
                        '<option value="light">Light Theme</option>' +
                        '<option value="dark">Dark Theme</option>' +
                        '<option value="monochrome_light">Light Monochrome</option>' +
                        '<option value="monochrome_dark">Dark Monochrome</option>';
                    themeSelect.value = kgCurrentTheme;
                    themeSelect.onchange = function() {{
                        const newTheme = this.value;
                        console.log('🎨 Changing theme to:', newTheme);
                        window.applyTheme(newTheme);
                        kgCurrentTheme = newTheme;
                    }};
                    
                    themeGroup.appendChild(themeLabel);
                    themeGroup.appendChild(themeSelect);
                    
                    panelContent.appendChild(hierarchyGroup);
                    panelContent.appendChild(sceneGroup);
                    panelContent.appendChild(statsGroup);
                    panelContent.appendChild(themeGroup);
                    
                    controlPanel.appendChild(panelHeader);
                    controlPanel.appendChild(panelContent);
                    document.body.appendChild(controlPanel);
                    
                    // Panel toggle functionality
                    menuToggle.addEventListener('click', function() {{
                        const panel = document.getElementById('kg-main-panel');
                        const btn = this;
                        panel.classList.toggle('open');
                        btn.classList.toggle('active');
                    }});
                }}
                
                function updateStats() {{
                    setTimeout(function() {{
                        document.getElementById('kg-entity-count').textContent = kgAllNodesData.length;
                        document.getElementById('kg-visible-count').textContent = kgOriginalNodes ? kgOriginalNodes.filter(n => n.group === 'entity').length : '0';
                        document.getElementById('kg-edge-count').textContent = kgAllEdgesData.length;
                        
                        // Build simple tree
                        const treeContainer = document.getElementById('kg-hierarchy-tree');
                        if (kgContainerData && kgContainerData.length > 0) {{
                            let treeHtml = '';
                            const floors = kgContainerData.filter(c => c.group === 'floor');
                            const scenes = kgContainerData.filter(c => c.group === 'scene');
                            const entityCount = kgAllNodesData.length;
                            
                            floors.forEach(function(floor) {{
                                treeHtml += '<div class="kg-tree-item kg-tree-floor">' + floor.label + ' (' + entityCount + ')</div>';
                                scenes.forEach(function(scene) {{
                                    const sceneEntities = Math.floor(entityCount / scenes.length);
                                    treeHtml += '<div class="kg-tree-item kg-tree-scene">' + scene.label + ' (' + sceneEntities + ')</div>';
                                    treeHtml += '<div class="kg-tree-item kg-tree-entity" onclick="focusEntities()">Objects</div>';
                                }});
                            }});
                            
                            treeContainer.innerHTML = treeHtml;
                        }} else {{
                            treeContainer.innerHTML = '<div class="kg-tree-item kg-tree-entity" onclick="focusEntities()">' + kgAllNodesData.length + ' Entities</div>';
                        }}
                    }}, 500);
                }}
                
                window.focusEntities = function() {{
                    if (kgOriginalNodes) {{
                        const entityNodes = kgOriginalNodes.filter(n => n.group === 'entity');
                        nodes.clear();
                        edges.clear();
                        nodes.add(entityNodes);
                        network.fit();
                    }}
                }};
                
                window.filterByScene = function(sceneName) {{
                    console.log('🔍 filterByScene called with:', sceneName);
                    
                    // Clear previous selection and reset network state
                    if (network) {{
                        network.setSelection({{
                            nodes: [],
                            edges: []
                        }});
                        network.unselectAll();
                    }}
                    
                    if (!kgOriginalNodes || !kgOriginalEdges) {{
                        console.error('❌ Missing original data');
                        return;
                    }}
                    
                    if (!sceneName) {{
                        // Show all entities
                        const entityNodes = kgOriginalNodes.filter(n => n.group === 'entity');
                        nodes.clear();
                        edges.clear();
                        nodes.add(entityNodes);
                        console.log('✅ Showing all entities:', entityNodes.length);
                    }} else {{
                        // Debug: show what we're working with
                        console.log('📊 Available nodes:', kgOriginalNodes.length, 'edges:', kgOriginalEdges.length);
                        console.log('🔍 Looking for scene:', sceneName);
                        
                        // First, find all nodes that belong to this scene
                        const sceneNodes = kgOriginalNodes.filter(node => {{
                            if (node.group !== 'entity') return false;
                            
                            // Check if this node has an inScene relationship to our target scene
                            const hasInSceneEdge = kgOriginalEdges.some(edge => {{
                                const isInSceneEdge = edge.label === 'inScene' && edge.from === node.id;
                                if (isInSceneEdge) {{
                                    // Check if the target contains our scene name
                                    const targetContainsScene = edge.to && (
                                        edge.to.includes(sceneName) || 
                                        edge.to.endsWith('/' + sceneName) ||
                                        edge.to === sceneName
                                    );
                                    return targetContainsScene;
                                }}
                                return false;
                            }});
                            
                            return hasInSceneEdge;
                        }});
                        
                        console.log('🎯 Found', sceneNodes.length, 'nodes in scene', sceneName);
                        
                        if (sceneNodes.length === 0) {{
                            console.warn('⚠️ No nodes found for scene:', sceneName);
                            // Show all nodes if no specific scene nodes found
                            const entityNodes = kgOriginalNodes.filter(n => n.group === 'entity');
                            nodes.clear();
                            edges.clear(); 
                            nodes.add(entityNodes);
                            return;
                        }}
                        
                        // Get edges between scene nodes and their properties
                        const sceneNodeIds = new Set(sceneNodes.map(n => n.id));
                        const sceneEdges = kgOriginalEdges.filter(edge => {{
                            // Skip position/rotation relationships (they're hidden)
                            if (edge.label === 'position' || edge.label === 'rotation') return false;
                            
                            // Include edges:
                            // 1. Between nodes in this scene  
                            // 2. From scene nodes to literals (properties)
                            const fromInScene = sceneNodeIds.has(edge.from);
                            const toInScene = edge.to && sceneNodeIds.has(edge.to);
                            const toLiteral = typeof edge.to === 'string' && !edge.to.startsWith('http');
                            
                            return fromInScene && (toInScene || toLiteral);
                        }});
                        
                        nodes.clear();
                        edges.clear();
                        nodes.add(sceneNodes);
                        edges.add(sceneEdges);
                        
                        console.log('✅ Scene filter applied:', sceneNodes.length, 'nodes,', sceneEdges.length, 'edges');
                        
                        // Debug: show a few node and edge examples
                        if (sceneNodes.length > 0) {{
                            console.log('📝 Sample nodes:', sceneNodes.slice(0, 3).map(n => n.label || n.id));
                        }}
                        if (sceneEdges.length > 0) {{
                            console.log('📝 Sample edges:', sceneEdges.slice(0, 3).map(e => e.label));
                        }}
                    }}
                    
                    setTimeout(() => network.fit(), 100);
                }};
                
                function handleKgNodeClick(params) {{
                    console.log('Node clicked:', params);
                    if (!params.nodes || params.nodes.length === 0) return;
                    
                    const clickedNodeId = params.nodes[0];
                    const clickedNode = kgOriginalNodes.find(n => n.id === clickedNodeId);
                    if (!clickedNode) return;
                    
                    if (!clickedNode.group || clickedNode.group !== 'entity') return;
                    
                    // Toggle selection
                    if (kgSelectedEntities.has(clickedNodeId)) {{
                        kgSelectedEntities.delete(clickedNodeId);
                    }} else {{
                        kgSelectedEntities.add(clickedNodeId);
                    }}
                    
                    document.getElementById('kg-selected-count').textContent = kgSelectedEntities.size.toString();
                    updateVisibleNetwork();
                }}
                
                function updateVisibleNetwork() {{
                    try {{
                        const baseNodes = kgOriginalNodes.filter(node => node.group === 'entity');
                        let targetEdges = [];
                        let additionalNodes = [];
                        
                        if (kgSelectedEntities.size > 0) {{
                            const relatedEdges = kgOriginalEdges.filter(edge => 
                                kgSelectedEntities.has(edge.from) || kgSelectedEntities.has(edge.to));
                            
                            const relatedNodeIds = new Set();
                            relatedEdges.forEach(edge => {{
                                relatedNodeIds.add(edge.from);
                                relatedNodeIds.add(edge.to);
                            }});
                            
                            additionalNodes = kgOriginalNodes.filter(node => 
                                relatedNodeIds.has(node.id) && !baseNodes.some(bn => bn.id === node.id));
                            
                            targetEdges = relatedEdges;
                        }}
                        
                        edges.clear();
                        if (targetEdges.length > 0) {{
                            edges.add(targetEdges);
                        }}
                        
                        const currentNodes = nodes.get();
                        const currentNodeIds = new Set(currentNodes.map(n => n.id));
                        const targetNodeIds = new Set([...baseNodes, ...additionalNodes].map(n => n.id));
                        
                        const nodesToAdd = [...baseNodes, ...additionalNodes].filter(node => !currentNodeIds.has(node.id));
                        const nodesToRemove = currentNodes.filter(node => !targetNodeIds.has(node.id)).map(n => n.id);
                        
                        if (nodesToAdd.length > 0) nodes.add(nodesToAdd);
                        if (nodesToRemove.length > 0) nodes.remove(nodesToRemove);
                        
                        // Highlight selected entities
                        const updates = baseNodes.map(node => ({{
                            id: node.id,
                            color: kgSelectedEntities.has(node.id) ? 
                                {{ background: '{theme['selection_bg']}', border: '{theme['selection_border']}' }} : node.color
                        }}));
                        nodes.update(updates);
                        
                    }} catch (error) {{
                        console.error('Error updating network:', error);
                    }}
                }}
                
                // Initialize when network is ready
                let checkCount = 0;
                const checkNetwork = function() {{
                    console.log('🔍 Check', checkCount + 1, '- Network ready?', typeof network !== 'undefined' && !!network);
                    if (typeof network !== 'undefined' && network) {{
                        console.log('🎯 Network found! Initializing in 500ms...');
                        setTimeout(initializeFocusedMode, 500);
                    }} else if (checkCount < 20) {{
                        checkCount++;
                        setTimeout(checkNetwork, 500);
                    }} else {{
                        console.error('❌ Network not found after 20 attempts');
                    }}
                }};
                
                console.log('⏰ Starting network check in 100ms...');
                setTimeout(checkNetwork, 100);
                
            }})();
        </script>
        """
        
        return js_code
    
    @staticmethod
    def inject_javascript(html_string: str, js_code: str) -> str:
        """Inject JavaScript into HTML before closing body tag."""
        # First clean up the PyVis HTML
        html_string = HTMLGenerator.cleanup_pyvis_html(html_string)
        
        # Check if our code is already injected to avoid duplicates
        if 'kgInitialized' in html_string:
            print("⚠️ JavaScript already injected, skipping")
            return html_string
            
        body_end = html_string.rfind('</body>')
        if body_end != -1:
            html_string = html_string[:body_end] + js_code + '\n' + html_string[body_end:]
            print("✅ JavaScript injected successfully")
        else:
            print("⚠️ Warning: Could not find closing body tag for JavaScript injection")
        
        return html_string