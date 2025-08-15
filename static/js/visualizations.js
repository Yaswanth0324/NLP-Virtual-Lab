// Advanced visualization utilities for NLP Virtual Lab
class VisualizationManager {
    constructor() {
        this.charts = {};
        this.svgElements = {};
    }
    
    // Create a syntax tree visualization using D3.js
    createSyntaxTree(containerId, treeData) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Clear previous content
        container.innerHTML = '';
        
        // Create SVG container
        const width = container.offsetWidth || 600;
        const height = 400;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('class', 'syntax-tree');
        
        // Create tree layout
        const treeLayout = d3.tree().size([width - 100, height - 100]);
        
        // Convert tree data to D3 hierarchy
        const root = d3.hierarchy(treeData);
        const treeData_d3 = treeLayout(root);
        
        // Create group for tree elements
        const g = svg.append('g')
            .attr('transform', 'translate(50, 50)');
        
        // Create links
        g.selectAll('.link')
            .data(treeData_d3.links())
            .enter()
            .append('path')
            .attr('class', 'link')
            .attr('d', d3.linkHorizontal()
                .x(d => d.y)
                .y(d => d.x))
            .style('fill', 'none')
            .style('stroke', 'var(--bs-secondary)')
            .style('stroke-width', '2px');
        
        // Create nodes
        const nodes = g.selectAll('.node')
            .data(treeData_d3.descendants())
            .enter()
            .append('g')
            .attr('class', 'node')
            .attr('transform', d => `translate(${d.y}, ${d.x})`);
        
        // Add circles to nodes
        nodes.append('circle')
            .attr('r', 8)
            .style('fill', 'var(--bs-primary)')
            .style('stroke', 'var(--bs-light)')
            .style('stroke-width', '2px');
        
        // Add text labels
        nodes.append('text')
            .attr('dy', '0.35em')
            .attr('x', d => d.children ? -12 : 12)
            .style('text-anchor', d => d.children ? 'end' : 'start')
            .style('font-size', '12px')
            .style('fill', 'var(--bs-body-color)')
            .text(d => d.data.name);
        
        this.svgElements[containerId] = svg;
    }
    
    // Create word cloud visualization
    createWordCloud(containerId, words, frequencies) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        // Create simple word cloud layout
        const maxFreq = Math.max(...Object.values(frequencies));
        const minFreq = Math.min(...Object.values(frequencies));
        
        const wordCloudDiv = document.createElement('div');
        wordCloudDiv.className = 'word-cloud d-flex flex-wrap justify-content-center align-items-center';
        wordCloudDiv.style.height = '300px';
        wordCloudDiv.style.padding = '20px';
        
        Object.entries(frequencies).forEach(([word, freq]) => {
            const span = document.createElement('span');
            span.textContent = word;
            span.className = 'word-cloud-item me-2 mb-2';
            
            // Calculate font size based on frequency
            const fontSize = Math.max(12, Math.min(36, (freq / maxFreq) * 24 + 12));
            span.style.fontSize = `${fontSize}px`;
            span.style.fontWeight = 'bold';
            span.style.color = this.getRandomColor();
            span.style.cursor = 'pointer';
            span.title = `Frequency: ${freq}`;
            
            // Add hover effect
            span.addEventListener('mouseenter', () => {
                span.style.opacity = '0.7';
                span.style.transform = 'scale(1.1)';
            });
            
            span.addEventListener('mouseleave', () => {
                span.style.opacity = '1';
                span.style.transform = 'scale(1)';
            });
            
            wordCloudDiv.appendChild(span);
        });
        
        container.appendChild(wordCloudDiv);
    }
    
    // Create network graph for word relationships
    createNetworkGraph(containerId, nodes, links) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        const width = container.offsetWidth || 600;
        const height = 400;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('class', 'network-graph');
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2));
        
        // Create links
        const link = svg.append('g')
            .selectAll('line')
            .data(links)
            .enter()
            .append('line')
            .style('stroke', 'var(--bs-secondary)')
            .style('stroke-width', '2px');
        
        // Create nodes
        const node = svg.append('g')
            .selectAll('g')
            .data(nodes)
            .enter()
            .append('g')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // Add circles to nodes
        node.append('circle')
            .attr('r', d => Math.max(8, d.weight * 3))
            .style('fill', 'var(--bs-primary)')
            .style('stroke', 'var(--bs-light)')
            .style('stroke-width', '2px');
        
        // Add labels
        node.append('text')
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', 'var(--bs-body-color)')
            .text(d => d.label);
        
        // Update positions on simulation tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
        });
        
        // Drag functions
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        this.svgElements[containerId] = svg;
    }
    
    // Create heatmap visualization
    createHeatmap(containerId, data, labels) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        const width = container.offsetWidth || 600;
        const height = 400;
        const margin = { top: 50, right: 50, bottom: 50, left: 50 };
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('class', 'heatmap');
        
        // Create scales
        const xScale = d3.scaleBand()
            .range([margin.left, width - margin.right])
            .domain(labels.x)
            .padding(0.1);
        
        const yScale = d3.scaleBand()
            .range([margin.top, height - margin.bottom])
            .domain(labels.y)
            .padding(0.1);
        
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, d3.max(data.flat())]);
        
        // Create cells
        svg.selectAll('rect')
            .data(data.flat())
            .enter()
            .append('rect')
            .attr('x', (d, i) => xScale(labels.x[i % labels.x.length]))
            .attr('y', (d, i) => yScale(labels.y[Math.floor(i / labels.x.length)]))
            .attr('width', xScale.bandwidth())
            .attr('height', yScale.bandwidth())
            .style('fill', d => colorScale(d))
            .style('stroke', 'var(--bs-border-color)')
            .style('stroke-width', '1px');
        
        // Add axes
        svg.append('g')
            .attr('transform', `translate(0, ${height - margin.bottom})`)
            .call(d3.axisBottom(xScale))
            .style('color', 'var(--bs-body-color)');
        
        svg.append('g')
            .attr('transform', `translate(${margin.left}, 0)`)
            .call(d3.axisLeft(yScale))
            .style('color', 'var(--bs-body-color)');
        
        this.svgElements[containerId] = svg;
    }
    
    // Create animated bar chart
    createAnimatedBarChart(containerId, data, labels) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Create canvas for Chart.js
        const canvas = document.createElement('canvas');
        canvas.id = `${containerId}_chart`;
        canvas.width = 400;
        canvas.height = 300;
        
        container.innerHTML = '';
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }
        
        this.charts[containerId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Values',
                    data: data,
                    backgroundColor: 'rgba(13, 110, 253, 0.8)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: 'var(--bs-body-color)'
                        },
                        grid: {
                            color: 'var(--bs-border-color)'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'var(--bs-body-color)'
                        },
                        grid: {
                            color: 'var(--bs-border-color)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: 'var(--bs-body-color)'
                        }
                    }
                }
            }
        });
    }
    
    // Create progress visualization
    createProgressVisualization(containerId, progress, total) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        const percentage = (progress / total) * 100;
        
        const progressDiv = document.createElement('div');
        progressDiv.className = 'progress-visualization text-center';
        progressDiv.innerHTML = `
            <div class="progress-circle mb-3">
                <svg width="120" height="120" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="50" fill="none" stroke="var(--bs-secondary)" stroke-width="8" opacity="0.3"/>
                    <circle cx="60" cy="60" r="50" fill="none" stroke="var(--bs-primary)" stroke-width="8" 
                            stroke-dasharray="314.16" stroke-dashoffset="${314.16 - (314.16 * percentage / 100)}"
                            stroke-linecap="round" transform="rotate(-90 60 60)"/>
                    <text x="60" y="60" text-anchor="middle" dy="0.35em" font-size="18" font-weight="bold" fill="var(--bs-body-color)">
                        ${Math.round(percentage)}%
                    </text>
                </svg>
            </div>
            <div class="progress-text">
                <strong>${progress}</strong> of <strong>${total}</strong> completed
            </div>
        `;
        
        container.appendChild(progressDiv);
    }
    
    // Create timeline visualization
    createTimeline(containerId, events) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        const timelineDiv = document.createElement('div');
        timelineDiv.className = 'timeline-visualization';
        
        events.forEach((event, index) => {
            const eventDiv = document.createElement('div');
            eventDiv.className = 'timeline-event d-flex align-items-center mb-3';
            eventDiv.innerHTML = `
                <div class="timeline-marker bg-primary rounded-circle me-3" style="width: 12px; height: 12px; min-width: 12px;"></div>
                <div class="timeline-content">
                    <h6 class="mb-1">${event.title}</h6>
                    <p class="mb-0 text-muted">${event.description}</p>
                    ${event.time ? `<small class="text-secondary">${event.time}</small>` : ''}
                </div>
            `;
            timelineDiv.appendChild(eventDiv);
        });
        
        container.appendChild(timelineDiv);
    }
    
    // Utility function to generate random colors
    getRandomColor() {
        const colors = [
            'var(--bs-primary)',
            'var(--bs-secondary)',
            'var(--bs-success)',
            'var(--bs-danger)',
            'var(--bs-warning)',
            'var(--bs-info)'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    // Clean up function
    cleanup() {
        // Destroy all Chart.js instances
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
        
        // Clean up SVG elements
        Object.values(this.svgElements).forEach(svg => {
            if (svg && svg.node && svg.node()) {
                svg.node().remove();
            }
        });
        this.svgElements = {};
    }
    
    // Responsive helper
    handleResize() {
        // Trigger resize for all charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
}

// Color scheme helpers for different themes
const ColorSchemes = {
    primary: {
        light: ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5'],
        dark: ['#1565c0', '#1976d2', '#1e88e5', '#2196f3', '#42a5f5']
    },
    success: {
        light: ['#e8f5e8', '#c8e6c8', '#a5d6a5', '#81c784', '#66bb6a'],
        dark: ['#2e7d32', '#388e3c', '#43a047', '#4caf50', '#66bb6a']
    },
    warning: {
        light: ['#fff8e1', '#ffecb3', '#ffe082', '#ffd54f', '#ffca28'],
        dark: ['#f57c00', '#fb8c00', '#ff9800', '#ffa726', '#ffb74d']
    },
    danger: {
        light: ['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#ef5350'],
        dark: ['#c62828', '#d32f2f', '#e53935', '#f44336', '#ef5350']
    }
};

// Export visualization utilities
window.VisualizationManager = VisualizationManager;
window.ColorSchemes = ColorSchemes;

// Global visualization manager instance
window.visualizationManager = new VisualizationManager();

// Handle window resize
window.addEventListener('resize', () => {
    window.visualizationManager.handleResize();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    window.visualizationManager.cleanup();
});

// Helper functions for creating specific NLP visualizations
window.nlpVisualizations = {
    // Create dependency tree visualization
    createDependencyTree: function(containerId, dependencies) {
        const nodes = [];
        const links = [];
        
        dependencies.forEach((dep, index) => {
            nodes.push({
                id: index,
                label: dep.word,
                pos: dep.pos,
                weight: 1
            });
            
            if (dep.head !== -1) {
                links.push({
                    source: dep.head,
                    target: index,
                    relation: dep.relation
                });
            }
        });
        
        window.visualizationManager.createNetworkGraph(containerId, nodes, links);
    },
    
    // Create POS distribution chart
    createPOSDistribution: function(containerId, posData) {
        const labels = Object.keys(posData);
        const data = Object.values(posData).map(arr => arr.length);
        
        window.visualizationManager.createAnimatedBarChart(containerId, data, labels);
    },
    
    // Create sentiment timeline
    createSentimentTimeline: function(containerId, sentimentData) {
        const events = sentimentData.map((item, index) => ({
            title: `Sentence ${index + 1}`,
            description: item.text,
            time: `Sentiment: ${item.sentiment}`,
            sentiment: item.sentiment
        }));
        
        window.visualizationManager.createTimeline(containerId, events);
    }
};
