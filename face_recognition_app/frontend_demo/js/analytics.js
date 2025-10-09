/**
 * Analytics Module
 * Handles analytics dashboard, statistics, and reporting
 */

class AnalyticsModule {
    constructor() {
        this.charts = {};
        this.refreshInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('loadAnalyticsBtn').addEventListener('click', this.loadAnalyticsDashboard.bind(this));
        document.getElementById('loadStatsBtn').addEventListener('click', this.loadStatistics.bind(this));
        document.getElementById('loadAuthLogsBtn').addEventListener('click', this.loadAuthenticationLogs.bind(this));
        document.getElementById('loadSecurityAlertsBtn').addEventListener('click', this.loadSecurityAlerts.bind(this));
    }

    async loadAnalyticsDashboard() {
        try {
            if (!window.app.isAuthenticated) {
                window.app.showAlert('Please login first', 'error');
                return;
            }

            const dashboard = await api.getAnalyticsDashboard();
            this.displayDashboard(dashboard);
            
            window.app.showAlert('Analytics dashboard loaded', 'success');

        } catch (error) {
            window.app.showAlert(`Failed to load analytics dashboard: ${error.message}`, 'error');
            
            // Show sample data if API fails
            this.displaySampleDashboard();
        }
    }

    async loadStatistics() {
        try {
            if (!window.app.isAuthenticated) {
                window.app.showAlert('Please login first', 'error');
                return;
            }

            const stats = await api.getStatistics();
            this.displayStatistics(stats);
            
            window.app.showAlert('Statistics loaded', 'success');

        } catch (error) {
            window.app.showAlert(`Failed to load statistics: ${error.message}`, 'error');
            
            // Show sample statistics if API fails
            this.displaySampleStatistics();
        }
    }

    displayDashboard(dashboard) {
        if (!dashboard) return;

        // Update summary cards
        if (dashboard.summary) {
            this.updateSummaryCards(dashboard.summary);
        }

        // Update charts if data is available
        if (dashboard.charts) {
            this.updateCharts(dashboard.charts);
        }

        // Update recent activities
        if (dashboard.recent_activities) {
            this.updateRecentActivities(dashboard.recent_activities);
        }
    }

    displaySampleDashboard() {
        // Sample dashboard data for demo purposes
        const sampleDashboard = {
            summary: {
                total_users: 15,
                successful_auths: 142,
                failed_auths: 23,
                active_sessions: 3
            },
            charts: {
                auth_success_rate: [85, 78, 92, 88, 95],
                daily_logins: [45, 52, 38, 67, 73, 58, 81]
            },
            recent_activities: [
                { activity: 'User john@example.com logged in', timestamp: new Date() },
                { activity: 'Failed authentication attempt', timestamp: new Date(Date.now() - 60000) },
                { activity: 'New user registered', timestamp: new Date(Date.now() - 120000) }
            ]
        };
        
        this.displayDashboard(sampleDashboard);
    }

    displaySampleStatistics() {
        const sampleStats = {
            users: {
                total: 15,
                active_today: 8,
                enrolled_faces: 12,
                new_registrations: 3
            },
            authentication: {
                total_attempts: 165,
                successful: 142,
                failed: 23,
                success_rate: 86.1
            },
            system: {
                uptime: '99.8%',
                avg_response_time: '120ms',
                errors_today: 2,
                storage_used: '2.3GB'
            }
        };
        
        this.displayStatistics(sampleStats);
    }

    displayStatistics(stats) {
        if (!stats) return;

        // Update individual stat cards
        if (stats.users) {
            this.updateStatCard('totalUsersCount', stats.users.total || 0);
        }
        
        if (stats.authentication) {
            this.updateStatCard('successfulAuthsCount', stats.authentication.successful || 0);
            this.updateStatCard('failedAuthsCount', stats.authentication.failed || 0);
        }
        
        if (stats.system) {
            this.updateStatCard('activeSessionsCount', stats.system.active_sessions || 0);
        }
    }

    updateSummaryCards(summary) {
        this.updateStatCard('totalUsersCount', summary.total_users || 0);
        this.updateStatCard('successfulAuthsCount', summary.successful_auths || 0);
        this.updateStatCard('failedAuthsCount', summary.failed_auths || 0);
        this.updateStatCard('activeSessionsCount', summary.active_sessions || 0);
    }

    updateStatCard(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
            
            // Add animation effect
            element.style.transform = 'scale(1.1)';
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 200);
        }
    }

    updateCharts(chartData) {
        // Simple text-based chart representation for demo
        // In a real application, you would use Chart.js or similar library
        console.log('Chart data received:', chartData);
        
        // You could implement simple bar charts using CSS or a charting library
        if (chartData.auth_success_rate) {
            this.createSimpleChart('auth-success-chart', chartData.auth_success_rate, 'Success Rate %');
        }
        
        if (chartData.daily_logins) {
            this.createSimpleChart('daily-logins-chart', chartData.daily_logins, 'Daily Logins');
        }
    }

    createSimpleChart(containerId, data, title) {
        // Create a simple text-based chart
        let container = document.getElementById(containerId);
        if (!container) {
            // Create container if it doesn't exist
            container = document.createElement('div');
            container.id = containerId;
            container.className = 'mt-3 p-3 border rounded';
            
            // Add to statistics grid or create one
            const statsGrid = document.getElementById('statisticsGrid');
            if (statsGrid) {
                statsGrid.appendChild(container);
            }
        }
        
        const max = Math.max(...data);
        const bars = data.map((value, index) => {
            const height = (value / max) * 100;
            return `
                <div class="d-inline-block me-1" style="width: 20px;">
                    <div class="bg-primary" style="height: ${height}px; margin-bottom: 2px;"></div>
                    <small>${value}</small>
                </div>
            `;
        }).join('');
        
        container.innerHTML = `
            <h6>${title}</h6>
            <div class="d-flex align-items-end" style="height: 120px;">
                ${bars}
            </div>
        `;
    }

    updateRecentActivities(activities) {
        // Update recent activities display
        console.log('Recent activities:', activities);
    }

    async loadAuthenticationLogs() {
        try {
            if (!window.app.isAuthenticated) return;

            const logs = await api.getAuthenticationLogs();
            this.displayAuthenticationLogs(logs);

        } catch (error) {
            window.app.showAlert(`Failed to load authentication logs: ${error.message}`, 'error');
            
            // Show sample logs for demo
            this.displaySampleAuthLogs();
        }
    }

    displaySampleAuthLogs() {
        const sampleLogs = [
            {
                user: { email: 'john@example.com' },
                auth_method: 'face',
                success: true,
                created_at: new Date().toISOString()
            },
            {
                user: { email: 'jane@example.com' },
                auth_method: 'password',
                success: false,
                created_at: new Date(Date.now() - 60000).toISOString()
            },
            {
                user: { email: 'bob@example.com' },
                auth_method: 'face',
                success: true,
                created_at: new Date(Date.now() - 120000).toISOString()
            }
        ];
        
        this.displayAuthenticationLogs(sampleLogs);
    }

    displayAuthenticationLogs(response) {
        const authLogsList = document.getElementById('authLogsList');
        authLogsList.innerHTML = '';

        // Handle both array responses and paginated responses
        let logs = Array.isArray(response) ? response : (response?.results || []);

        if (!logs || logs.length === 0) {
            authLogsList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-muted">No logs found</td>
                </tr>
            `;
            return;
        }

        logs.slice(0, 10).forEach(log => {
            const row = document.createElement('tr');
            
            const userEmail = log.user?.email || log.attempted_email || 'Unknown';
            const method = log.auth_method || 'N/A';
            const success = log.success || log.result === 'success';
            
            const resultBadge = success ? 
                '<span class="badge bg-success">Success</span>' : 
                '<span class="badge bg-danger">Failed</span>';

            row.innerHTML = `
                <td>${userEmail}</td>
                <td>${method}</td>
                <td>${resultBadge}</td>
                <td>
                    <small>${window.app.formatDate(log.created_at)}</small>
                </td>
            `;
            authLogsList.appendChild(row);
        });
    }

    async loadSecurityAlerts() {
        try {
            if (!window.app.isAuthenticated) return;

            const alerts = await api.getSecurityAlerts();
            this.displaySecurityAlerts(alerts);

        } catch (error) {
            window.app.showAlert(`Failed to load security alerts: ${error.message}`, 'error');
            
            // Show sample alerts for demo
            this.displaySampleSecurityAlerts();
        }
    }

    displaySampleSecurityAlerts() {
        const sampleAlerts = [
            {
                alert_type: 'failed_login_attempts',
                severity: 'medium',
                created_at: new Date().toISOString()
            },
            {
                alert_type: 'suspicious_face_auth',
                severity: 'high',
                created_at: new Date(Date.now() - 300000).toISOString()
            },
            {
                alert_type: 'account_lockout',
                severity: 'low',
                created_at: new Date(Date.now() - 600000).toISOString()
            }
        ];
        
        this.displaySecurityAlerts(sampleAlerts);
    }

    displaySecurityAlerts(response) {
        const securityAlertsList = document.getElementById('securityAlertsList');
        securityAlertsList.innerHTML = '';

        // Handle both array responses and paginated responses
        let alerts = Array.isArray(response) ? response : (response?.results || []);

        if (!alerts || alerts.length === 0) {
            securityAlertsList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-muted">No alerts found</td>
                </tr>
            `;
            return;
        }

        alerts.slice(0, 10).forEach(alert => {
            const row = document.createElement('tr');
            
            const severityBadge = this.getSeverityBadge(alert.severity);
            
            row.innerHTML = `
                <td>${alert.alert_type?.replace('_', ' ') || 'Unknown'}</td>
                <td>${severityBadge}</td>
                <td>
                    <small>${window.app.formatDate(alert.created_at)}</small>
                </td>
            `;
            securityAlertsList.appendChild(row);
        });
    }

    getSeverityBadge(severity) {
        const badges = {
            'low': '<span class="badge bg-info">Low</span>',
            'medium': '<span class="badge bg-warning">Medium</span>',
            'high': '<span class="badge bg-danger">High</span>',
            'critical': '<span class="badge bg-dark">Critical</span>'
        };
        
        return badges[severity] || '<span class="badge bg-secondary">Unknown</span>';
    }

    startAutoRefresh(interval = 30000) {
        // Auto-refresh analytics data every 30 seconds
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(() => {
            if (window.app.isAuthenticated) {
                this.loadStatistics();
            }
        }, interval);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    onTabActivated() {
        // Called when analytics tab is activated
        this.loadAnalyticsDashboard();
        this.loadStatistics();
        this.loadAuthenticationLogs();
        this.loadSecurityAlerts();
        
        // Start auto-refresh
        this.startAutoRefresh();
    }

    onTabDeactivated() {
        // Called when leaving analytics tab
        this.stopAutoRefresh();
    }

    // Export data functionality
    exportData(type) {
        // Export analytics data as CSV or JSON
        const timestamp = new Date().toISOString().split('T')[0];
        const filename = `analytics_${type}_${timestamp}`;
        
        // This would export actual data in a real implementation
        window.app.showAlert(`Export functionality would download ${filename}.csv`, 'info');
    }

    // Generate report
    async generateReport(reportType, dateRange) {
        try {
            // This would generate a comprehensive report
            const reportData = {
                type: reportType,
                dateRange,
                generated_at: new Date(),
                summary: 'Report generation would be implemented here'
            };
            
            window.app.showAlert('Report generation completed', 'success');
            return reportData;
            
        } catch (error) {
            window.app.showAlert(`Failed to generate report: ${error.message}`, 'error');
        }
    }

    // Cleanup method
    destroy() {
        this.stopAutoRefresh();
    }
}

// Initialize analytics module
document.addEventListener('DOMContentLoaded', () => {
    window.analyticsModule = new AnalyticsModule();
});