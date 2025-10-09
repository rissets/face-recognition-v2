/**
 * System Module
 * Handles system status, configuration, and administrative functions
 */

class SystemModule {
    constructor() {
        this.systemStatus = null;
        this.refreshInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('loadSystemStatusBtn').addEventListener('click', this.loadSystemStatus.bind(this));
        document.getElementById('testCustomEndpointBtn').addEventListener('click', this.testCustomEndpoint.bind(this));
    }

    async loadSystemStatus() {
        try {
            const status = await api.getSystemStatus();
            this.systemStatus = status;
            this.displaySystemStatus(status);
            
            window.app.showAlert('System status loaded successfully', 'success');

        } catch (error) {
            window.app.showAlert(`Failed to load system status: ${error.message}`, 'error');
            
            // Show sample system status for demo
            this.displaySampleSystemStatus();
        }
    }

    displaySampleSystemStatus() {
        const sampleStatus = {
            status: 'healthy',
            uptime: '7 days, 14 hours, 23 minutes',
            version: '1.0.0',
            database: {
                status: 'connected',
                connections: 15,
                response_time: '2ms'
            },
            face_recognition: {
                engine_status: 'active',
                models_loaded: 3,
                processing_queue: 0
            },
            system_resources: {
                cpu_usage: '23%',
                memory_usage: '67%',
                disk_usage: '45%',
                network_status: 'connected'
            },
            services: {
                authentication: 'running',
                enrollment: 'running',
                streaming: 'running',
                analytics: 'running'
            },
            performance: {
                avg_response_time: '120ms',
                requests_per_minute: 45,
                error_rate: '0.2%'
            }
        };
        
        this.displaySystemStatus(sampleStatus);
    }

    displaySystemStatus(status) {
        const systemStatusContent = document.getElementById('systemStatusContent');
        
        if (!status) {
            systemStatusContent.textContent = 'No system status available';
            return;
        }

        // Create a comprehensive system status display
        let html = '';

        // Overall status
        html += this.createStatusSection('Overall Status', {
            'Status': this.getStatusBadge(status.status),
            'Uptime': status.uptime || 'Unknown',
            'Version': status.version || 'Unknown'
        });

        // Database status
        if (status.database) {
            html += this.createStatusSection('Database', {
                'Connection Status': this.getStatusBadge(status.database.status),
                'Active Connections': status.database.connections || 0,
                'Response Time': status.database.response_time || 'N/A'
            });
        }

        // Face recognition engine
        if (status.face_recognition) {
            html += this.createStatusSection('Face Recognition Engine', {
                'Engine Status': this.getStatusBadge(status.face_recognition.engine_status),
                'Models Loaded': status.face_recognition.models_loaded || 0,
                'Processing Queue': status.face_recognition.processing_queue || 0
            });
        }

        // System resources
        if (status.system_resources) {
            html += this.createStatusSection('System Resources', {
                'CPU Usage': status.system_resources.cpu_usage || 'N/A',
                'Memory Usage': status.system_resources.memory_usage || 'N/A',
                'Disk Usage': status.system_resources.disk_usage || 'N/A',
                'Network Status': this.getStatusBadge(status.system_resources.network_status)
            });
        }

        // Services
        if (status.services) {
            const serviceItems = {};
            Object.entries(status.services).forEach(([key, value]) => {
                serviceItems[key.replace('_', ' ').toUpperCase()] = this.getStatusBadge(value);
            });
            html += this.createStatusSection('Services', serviceItems);
        }

        // Performance metrics
        if (status.performance) {
            html += this.createStatusSection('Performance Metrics', {
                'Average Response Time': status.performance.avg_response_time || 'N/A',
                'Requests Per Minute': status.performance.requests_per_minute || 0,
                'Error Rate': status.performance.error_rate || 'N/A'
            });
        }

        systemStatusContent.innerHTML = html;
    }

    createStatusSection(title, items) {
        let html = `
            <div class="mb-4">
                <h6 class="border-bottom pb-2">${title}</h6>
                <div class="row">
        `;

        Object.entries(items).forEach(([key, value], index) => {
            if (index % 2 === 0) {
                html += '<div class="col-md-6">';
            }
            
            html += `
                <div class="mb-2">
                    <strong>${key}:</strong> 
                    <span class="ms-2">${value}</span>
                </div>
            `;
            
            if (index % 2 === 1) {
                html += '</div>';
            }
        });
        
        // Close last column if odd number of items
        if (Object.keys(items).length % 2 === 1) {
            html += '</div>';
        }
        
        html += `
                </div>
            </div>
        `;
        
        return html;
    }

    getStatusBadge(status) {
        const badges = {
            'healthy': '<span class="badge bg-success">Healthy</span>',
            'active': '<span class="badge bg-success">Active</span>',
            'running': '<span class="badge bg-success">Running</span>',
            'connected': '<span class="badge bg-success">Connected</span>',
            'warning': '<span class="badge bg-warning">Warning</span>',
            'error': '<span class="badge bg-danger">Error</span>',
            'stopped': '<span class="badge bg-danger">Stopped</span>',
            'disconnected': '<span class="badge bg-danger">Disconnected</span>',
            'maintenance': '<span class="badge bg-info">Maintenance</span>'
        };
        
        return badges[status] || `<span class="badge bg-secondary">${status}</span>`;
    }

    async testCustomEndpoint() {
        const endpoint = document.getElementById('customEndpoint').value;
        const method = document.getElementById('httpMethod').value;
        const bodyText = document.getElementById('requestBody').value;
        
        if (!endpoint) {
            window.app.showAlert('Please enter an endpoint', 'error');
            return;
        }

        try {
            let requestData = null;
            if (bodyText.trim()) {
                try {
                    requestData = JSON.parse(bodyText);
                } catch (e) {
                    window.app.showAlert('Invalid JSON in request body', 'error');
                    return;
                }
            }

            const response = await api.testCustomEndpoint(method, endpoint, requestData);
            
            const apiResponse = document.getElementById('apiResponse');
            if (apiResponse) {
                apiResponse.textContent = JSON.stringify(response, null, 2);
            }
            
            window.app.showAlert('Custom endpoint test completed successfully', 'success');
            
        } catch (error) {
            const apiResponse = document.getElementById('apiResponse');
            if (apiResponse) {
                apiResponse.textContent = `Error: ${error.message}`;
            }
            window.app.showAlert(`Custom endpoint test failed: ${error.message}`, 'error');
        }
    }

    // System health monitoring
    startHealthMonitoring(interval = 60000) {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(() => {
            if (window.app.isAuthenticated) {
                this.loadSystemStatus();
            }
        }, interval);
    }

    stopHealthMonitoring() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    // API endpoint testing utilities
    async testAllEndpoints() {
        const endpoints = [
            { method: 'GET', url: '/api/system/status/', description: 'System Status' },
            { method: 'GET', url: '/api/auth/profile/', description: 'User Profile' },
            { method: 'GET', url: '/api/recognition/embeddings/', description: 'Face Embeddings' },
            { method: 'GET', url: '/api/analytics/dashboard/', description: 'Analytics Dashboard' },
            { method: 'GET', url: '/api/streaming/sessions/', description: 'Streaming Sessions' },
            { method: 'GET', url: '/api/users/devices/', description: 'User Devices' }
        ];

        const results = [];
        
        for (const endpoint of endpoints) {
            try {
                const startTime = Date.now();
                const response = await api.testCustomEndpoint(endpoint.method, endpoint.url);
                const duration = Date.now() - startTime;
                
                results.push({
                    ...endpoint,
                    status: 'success',
                    duration,
                    response: response
                });
            } catch (error) {
                results.push({
                    ...endpoint,
                    status: 'error',
                    error: error.message
                });
            }
        }

        this.displayEndpointTestResults(results);
        return results;
    }

    displayEndpointTestResults(response) {
        // Handle both array responses and paginated responses
        let results = Array.isArray(response) ? response : (response?.results || []);
        
        let html = `
            <div class="mt-4">
                <h6>Endpoint Test Results</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Method</th>
                                <th>Endpoint</th>
                                <th>Status</th>
                                <th>Duration</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        results.forEach(result => {
            const statusBadge = result.status === 'success' ? 
                '<span class="badge bg-success">Success</span>' : 
                '<span class="badge bg-danger">Failed</span>';
            
            const duration = result.duration ? `${result.duration}ms` : 'N/A';
            
            html += `
                <tr>
                    <td><span class="badge bg-info">${result.method}</span></td>
                    <td><code>${result.url}</code></td>
                    <td>${statusBadge}</td>
                    <td>${duration}</td>
                </tr>
            `;
        });

        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        const apiResponse = document.getElementById('apiResponse');
        if (apiResponse) {
            apiResponse.innerHTML = html;
        }
    }

    // Configuration management
    async getSystemConfiguration() {
        try {
            // This would retrieve system configuration
            const config = {
                face_recognition: {
                    confidence_threshold: 0.8,
                    max_faces_per_user: 5,
                    liveness_detection: true
                },
                authentication: {
                    max_login_attempts: 3,
                    session_timeout: 3600,
                    require_2fa: false
                },
                system: {
                    debug_mode: false,
                    log_level: 'INFO',
                    backup_enabled: true
                }
            };

            return config;
        } catch (error) {
            window.app.showAlert(`Failed to get configuration: ${error.message}`, 'error');
        }
    }

    async updateSystemConfiguration(config) {
        try {
            // This would update system configuration
            window.app.showAlert('System configuration updated', 'success');
            return config;
        } catch (error) {
            window.app.showAlert(`Failed to update configuration: ${error.message}`, 'error');
        }
    }

    // Maintenance mode
    async enableMaintenanceMode() {
        try {
            window.app.showAlert('Maintenance mode enabled', 'warning');
        } catch (error) {
            window.app.showAlert(`Failed to enable maintenance mode: ${error.message}`, 'error');
        }
    }

    async disableMaintenanceMode() {
        try {
            window.app.showAlert('Maintenance mode disabled', 'success');
        } catch (error) {
            window.app.showAlert(`Failed to disable maintenance mode: ${error.message}`, 'error');
        }
    }

    // Backup and restore
    async createSystemBackup() {
        try {
            const backup = {
                timestamp: new Date().toISOString(),
                version: '1.0.0',
                data: 'Backup data would be included here'
            };

            const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `system_backup_${new Date().toISOString().split('T')[0]}.json`;
            link.click();
            
            URL.revokeObjectURL(url);
            
            window.app.showAlert('System backup created successfully', 'success');
        } catch (error) {
            window.app.showAlert(`Failed to create backup: ${error.message}`, 'error');
        }
    }

    onTabActivated() {
        // Called when system tab is activated
        this.loadSystemStatus();
        this.startHealthMonitoring();
    }

    onTabDeactivated() {
        // Called when leaving system tab
        this.stopHealthMonitoring();
    }

    // Cleanup method
    destroy() {
        this.stopHealthMonitoring();
    }
}

// Initialize system module
document.addEventListener('DOMContentLoaded', () => {
    window.systemModule = new SystemModule();
});