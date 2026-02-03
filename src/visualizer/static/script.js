// =================================================
// Global State
// =================================================

let isDragging = false;
let offsetX = 0;
let offsetY = 0;
let filteredNodes = null;
let filteredEdges = null;

// Filter states
let nodeFilters = {
    showScenes: true
};


// =================================================
// Utility
// =================================================

function registerEvent(selector, event, handler) {

    const el = document.querySelector(selector);

    if (el) {
        el.addEventListener(event, handler);
    } else {
        console.warn("Missing element:", selector);
    }
}


// =================================================
// Initialization
// =================================================

document.addEventListener("DOMContentLoaded", () => {

    restoreTheme();

    registerUIEvents();

    initToolbarDrag();

    initGraphEvents();
    
    initFilters();
    
    updateStatistics();

});


// =================================================
// Theme
// =================================================

function restoreTheme() {

    if (localStorage.getItem("darkMode") === "true") {
        document.body.classList.add("dark-mode");
        updateThemeIcon(true);
    }
}


function toggleTheme() {

    document.body.classList.toggle("dark-mode");

    const isDark = document.body.classList.contains("dark-mode");

    updateThemeIcon(isDark);

    applyGraphTheme(isDark);

    localStorage.setItem("darkMode", isDark);
}


function updateThemeIcon(isDark) {

    const btn = document.getElementById("themeBtn");

    if (btn) {
        btn.textContent = isDark ? "☀️" : "🌙";
    }
}


function applyGraphTheme(isDark) {

    if (!window.network) return;


    const dark = {
        nodes: {
            color: {
                background: "#1e293b",
                border: "#475569"
            },
            font: { color: "#f8fafc" }
        },

        edges: {
            color: "#94a3b8",
            font: { color: "#cbd5e1" }
        }
    };


    const light = {
        nodes: {
            color: {
                background: "#97c2fc",
                border: "#2b7ce9"
            },
            font: { color: "#222" }
        },

        edges: {
            color: "#848484",
            font: { color: "#222" }
        }
    };


    network.setOptions(isDark ? dark : light);
}


// =================================================
// Graph Controls
// =================================================

function fitView() {

    if (!window.network) return;

    network.fit({
        animation: { duration: 500 }
    });
}


function togglePhysics() {

    if (!window.network) return;

    const enabled = network.physics.options.enabled;

    network.setOptions({
        physics: { enabled: !enabled }
    });
}


function zoomIn() {

    if (!window.network) return;

    const scale = network.getScale();

    network.moveTo({ scale: scale * 1.2 });
}


function zoomOut() {

    if (!window.network) return;

    const scale = network.getScale();

    network.moveTo({ scale: scale * 0.8 });
}


function centerGraph() {

    if (!window.network) return;

    network.moveTo({
        position: network.getViewPosition(),
        scale: network.getScale()
    });
}


// =================================================
// Search
// =================================================

function searchNode() {

    if (!window.nodes) return;

    const text =
        document.getElementById("searchBox")
            ?.value
            ?.toLowerCase();


    if (!text) {
        alert("Enter node name");
        return;
    }


    const all = nodes.get();

    for (let n of all) {

        if (n.label.toLowerCase().includes(text)) {

            network.selectNodes([n.id]);

            network.focus(n.id, {
                scale: 1.5,
                animation: true
            });

            return;
        }
    }

    alert("Node not found");
}


// =================================================
// UI Events
// =================================================

function registerUIEvents() {

    registerEvent("#themeBtn", "click", toggleTheme);
    registerEvent("#searchBtn", "click", searchNode);

    registerEvent("#zoomInBtn", "click", zoomIn);
    registerEvent("#zoomOutBtn", "click", zoomOut);
    registerEvent("#homeBtn", "click", fitView);
    registerEvent("#centerBtn", "click", centerGraph);
    
    // Scene filter checkbox
    registerEvent("#showScenes", "change", updateFilters);
}


// =================================================
// Graph Events
// =================================================

function initGraphEvents() {

    if (!window.network) return;

    network.on("click", (params) => {

        if (params.nodes.length > 0) {

            const node = nodes.get(params.nodes[0]);

            console.log("Clicked:", node);
        }
    });
}


// =================================================
// Draggable Toolbar (Bounded)
// =================================================

function initToolbarDrag() {

    const toolbar = document.getElementById("floatingToolbar");

    if (!toolbar) return;


    toolbar.addEventListener("mousedown", (e) => {

        if (e.target.closest("button")) return;

        isDragging = true;

        offsetX = e.clientX - toolbar.offsetLeft;
        offsetY = e.clientY - toolbar.offsetTop;

        toolbar.style.opacity = "0.8";
    });


    document.addEventListener("mousemove", (e) => {

        if (!isDragging) return;

        const w = toolbar.offsetWidth;
        const h = toolbar.offsetHeight;

        const maxX = window.innerWidth - w;
        const maxY = window.innerHeight - h;


        let x = e.clientX - offsetX;
        let y = e.clientY - offsetY;


        x = Math.max(0, Math.min(x, maxX));
        y = Math.max(0, Math.min(y, maxY));


        toolbar.style.left = x + "px";
        toolbar.style.top = y + "px";
        toolbar.style.right = "auto";
    });


    document.addEventListener("mouseup", () => {

        isDragging = false;

        toolbar.style.opacity = "1";
    });

}


// =================================================
// Filtering and Layout Functions
// =================================================

function initFilters() {
    if (window.network && window.nodes && window.edges) {
        // Store original data
        filteredNodes = nodes;
        filteredEdges = edges;
    }
}

function updateFilters() {
    if (!window.network || !filteredNodes) return;
    
    // Update filter state
    nodeFilters.showScenes = document.getElementById("showScenes")?.checked ?? true;
    
    // Filter nodes
    const allNodes = filteredNodes.get();
    const visibleNodes = allNodes.filter(node => shouldShowNode(node));
    
    // Filter edges to only show those between visible nodes
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
    const allEdges = filteredEdges.get();
    const visibleEdges = allEdges.filter(edge => 
        visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to)
    );
    
    // Update the network
    nodes.clear();
    edges.clear();
    nodes.add(visibleNodes);
    edges.add(visibleEdges);
    
    updateStatistics();
}

function shouldShowNode(node) {
    const label = node.label.toLowerCase();
    
    // Only filter scenes
    if (!nodeFilters.showScenes && (label.includes("floorplan") || label.includes("scene"))) {
        return false;
    }
    
    return true;
}

function hasKeywords(text, keywords) {
    return keywords.some(keyword => text.includes(keyword));
}

function setLayout(layoutType) {
    if (!window.network) return;
    
    // Remove active class from all layout buttons
    document.querySelectorAll(".control-group button").forEach(btn => {
        btn.classList.remove("active");
    });
    
    let layoutOptions = {};
    
    switch(layoutType) {
        case "hierarchicalRepulsion":
            layoutOptions = {
                layout: {
                    hierarchical: {
                        enabled: true,
                        direction: "UD",
                        sortMethod: "directed",
                        levelSeparation: 150,
                        nodeSpacing: 200
                    }
                }
            };
            break;
            
        case "forceAtlas2Based":
            layoutOptions = {
                physics: {
                    forceAtlas2Based: {
                        gravitationalConstant: -50,
                        centralGravity: 0.01,
                        springLength: 100,
                        springConstant: 0.08,
                        damping: 0.4,
                        avoidOverlap: 0
                    },
                    maxVelocity: 50,
                    solver: "forceAtlas2Based",
                    timestep: 0.35,
                    stabilization: {iterations: 150}
                }
            };
            break;
            
        case "barnesHut":
        default:
            layoutOptions = {
                physics: {
                    barnesHut: {
                        gravitationalConstant: -30000,
                        centralGravity: 0.1,
                        springLength: 200,
                        springConstant: 0.05,
                        damping: 0.3,
                        avoidOverlap: 0.2
                    },
                    stabilization: {iterations: 500}
                }
            };
            break;
    }
    
    network.setOptions(layoutOptions);
    fitView();
}

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const layout = document.getElementById("layout");
    
    if (sidebar.style.display === "none") {
        sidebar.style.display = "block";
        layout.style.gridTemplateColumns = "240px 1fr";
    } else {
        sidebar.style.display = "none";
        layout.style.gridTemplateColumns = "0 1fr";
    }
}

function updateStatistics() {
    if (window.nodes && window.edges) {
        const nodeCount = document.getElementById("nodeCount");
        const edgeCount = document.getElementById("edgeCount");
        
        if (nodeCount) nodeCount.textContent = `Nodes: ${nodes.length}`;
        if (edgeCount) edgeCount.textContent = `Edges: ${edges.length}`;
    }
}
