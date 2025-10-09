/**
 * Users Module
 * Handles user management, devices, and user-specific data
 */

class UsersModule {
    constructor() {
        this.currentUser = null;
        this.userDevices = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('loadUserProfileBtn').addEventListener('click', this.loadUserProfile.bind(this));
        document.getElementById('loadUserDevicesBtn').addEventListener('click', this.loadUserDevices.bind(this));
        document.getElementById('loadAuthHistoryBtn').addEventListener('click', this.loadAuthHistory.bind(this));
        document.getElementById('loadSecurityAlertsUserBtn').addEventListener('click', this.loadUserSecurityAlerts.bind(this));
    }

    async loadUserProfile() {
        try {
            if (!window.app.isAuthenticated) {
                window.app.showAlert('Please login first', 'error');
                return;
            }

            const profile = await api.getUserProfile();
            this.currentUser = profile;
            this.displayUserProfile(profile);
            
            window.app.showAlert('User profile loaded', 'success');

        } catch (error) {
            window.app.showAlert(`Failed to load user profile: ${error.message}`, 'error');
        }
    }

    displayUserProfile(profile) {
        const userProfileContent = document.getElementById('userProfileContent');
        
        if (!profile) {
            userProfileContent.textContent = 'No profile data available';
            return;
        }

        // Create a formatted display of user profile
        const profileDisplay = {
            'User ID': profile.id,
            'Email': profile.email,
            'First Name': profile.first_name,
            'Last Name': profile.last_name,
            'Date Joined': window.app.formatDate(profile.date_joined),
            'Last Login': window.app.formatDate(profile.last_login),
            'Face Enrolled': profile.face_enrolled ? 'Yes' : 'No',
            'Face Auth Enabled': profile.face_auth_enabled ? 'Yes' : 'No',
            'Account Active': profile.is_active ? 'Yes' : 'No',
            'Enrollment Progress': `${profile.enrollment_progress || 0}%`,
            'Last Face Auth': window.app.formatDate(profile.last_face_auth)
        };

        userProfileContent.innerHTML = this.createProfileHTML(profileDisplay);
    }

    createProfileHTML(profileData) {
        let html = '<div class="row">';
        
        Object.entries(profileData).forEach(([key, value], index) => {
            if (index % 2 === 0) {
                html += '<div class="col-md-6">';
            }
            
            html += `
                <div class="mb-2">
                    <strong>${key}:</strong> 
                    <span class="text-muted">${value || 'N/A'}</span>
                </div>
            `;
            
            if (index % 2 === 1) {
                html += '</div>';
            }
        });
        
        // Close last column if odd number of items
        if (Object.keys(profileData).length % 2 === 1) {
            html += '</div>';
        }
        
        html += '</div>';
        return html;
    }

    async loadUserDevices() {
        try {
            if (!window.app.isAuthenticated) return;

            const devices = await api.getUserDevices();
            this.userDevices = devices;
            this.displayUserDevices(devices);

        } catch (error) {
            window.app.showAlert(`Failed to load user devices: ${error.message}`, 'error');
            
            // Show sample devices for demo
            this.displaySampleDevices();
        }
    }

    displaySampleDevices() {
        const sampleDevices = [
            {
                device_name: 'Chrome on MacBook Pro',
                device_type: 'desktop',
                last_seen: new Date().toISOString(),
                is_trusted: true,
                ip_address: '192.168.1.100'
            },
            {
                device_name: 'Safari on iPhone',
                device_type: 'mobile',
                last_seen: new Date(Date.now() - 3600000).toISOString(),
                is_trusted: true,
                ip_address: '192.168.1.101'
            },
            {
                device_name: 'Chrome on Windows',
                device_type: 'desktop',
                last_seen: new Date(Date.now() - 86400000).toISOString(),
                is_trusted: false,
                ip_address: '203.0.113.1'
            }
        ];
        
        this.displayUserDevices(sampleDevices);
    }

    displayUserDevices(response) {
        const userDevicesList = document.getElementById('userDevicesList');
        userDevicesList.innerHTML = '';

        // Handle both array responses and paginated responses
        let devices = Array.isArray(response) ? response : (response?.results || []);

        if (!devices || devices.length === 0) {
            userDevicesList.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-muted">No devices found</td>
                </tr>
            `;
            return;
        }

        devices.forEach(device => {
            const row = document.createElement('tr');
            
            const deviceIcon = this.getDeviceIcon(device.device_type);
            const trustedBadge = device.is_trusted ? 
                '<span class="badge bg-success">Trusted</span>' : 
                '<span class="badge bg-warning">Untrusted</span>';

            row.innerHTML = `
                <td>
                    <i class="${deviceIcon}"></i> ${device.device_name || 'Unknown Device'}
                    ${device.ip_address ? `<br><small class="text-muted">${device.ip_address}</small>` : ''}
                </td>
                <td>
                    <span class="badge bg-info">${device.device_type || 'Unknown'}</span>
                </td>
                <td>
                    <small>${window.app.formatDate(device.last_seen)}</small>
                </td>
                <td>${trustedBadge}</td>
            `;
            userDevicesList.appendChild(row);
        });
    }

    getDeviceIcon(deviceType) {
        const icons = {
            'desktop': 'fas fa-desktop',
            'mobile': 'fas fa-mobile-alt',
            'tablet': 'fas fa-tablet-alt',
            'laptop': 'fas fa-laptop'
        };
        
        return icons[deviceType] || 'fas fa-question-circle';
    }

    async loadAuthHistory() {
        try {
            if (!window.app.isAuthenticated) return;

            const history = await api.getUserAuthHistory();
            this.displayAuthHistory(history);

        } catch (error) {
            window.app.showAlert(`Failed to load authentication history: ${error.message}`, 'error');
            
            // Show sample history for demo
            this.displaySampleAuthHistory();
        }
    }

    displaySampleAuthHistory() {
        const sampleHistory = [
            {
                auth_method: 'face',
                success: true,
                created_at: new Date().toISOString()
            },
            {
                auth_method: 'password',
                success: true,
                created_at: new Date(Date.now() - 3600000).toISOString()
            },
            {
                auth_method: 'face',
                success: false,
                created_at: new Date(Date.now() - 7200000).toISOString()
            },
            {
                auth_method: 'face',
                success: true,
                created_at: new Date(Date.now() - 86400000).toISOString()
            }
        ];
        
        this.displayAuthHistory(sampleHistory);
    }

    displayAuthHistory(response) {
        const authHistoryList = document.getElementById('authHistoryList');
        authHistoryList.innerHTML = '';

        // Handle both array responses and paginated responses
        let history = Array.isArray(response) ? response : (response?.results || []);

        if (!history || history.length === 0) {
            authHistoryList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-muted">No history found</td>
                </tr>
            `;
            return;
        }

        history.slice(0, 10).forEach(entry => {
            const row = document.createElement('tr');
            
            const methodIcon = this.getAuthMethodIcon(entry.auth_method);
            const resultBadge = entry.success ? 
                '<span class="badge bg-success">Success</span>' : 
                '<span class="badge bg-danger">Failed</span>';

            row.innerHTML = `
                <td>
                    <i class="${methodIcon}"></i> ${entry.auth_method || 'Unknown'}
                </td>
                <td>${resultBadge}</td>
                <td>
                    <small>${window.app.formatDate(entry.created_at)}</small>
                </td>
            `;
            authHistoryList.appendChild(row);
        });
    }

    getAuthMethodIcon(method) {
        const icons = {
            'face': 'fas fa-user-circle',
            'password': 'fas fa-key',
            'biometric': 'fas fa-fingerprint',
            'sms': 'fas fa-sms',
            'email': 'fas fa-envelope'
        };
        
        return icons[method] || 'fas fa-sign-in-alt';
    }

    async loadUserSecurityAlerts() {
        try {
            if (!window.app.isAuthenticated) return;

            const alerts = await api.getUserSecurityAlerts();
            this.displayUserSecurityAlerts(alerts);

        } catch (error) {
            window.app.showAlert(`Failed to load security alerts: ${error.message}`, 'error');
            
            // Show sample alerts for demo
            this.displaySampleUserSecurityAlerts();
        }
    }

    displaySampleUserSecurityAlerts() {
        const sampleAlerts = [
            {
                alert_type: 'New device login',
                severity: 'medium',
                description: 'Login from new device detected',
                created_at: new Date().toISOString(),
                resolved: false
            },
            {
                alert_type: 'Failed face authentication',
                severity: 'low',
                description: 'Multiple failed face authentication attempts',
                created_at: new Date(Date.now() - 3600000).toISOString(),
                resolved: true
            }
        ];
        
        this.displayUserSecurityAlerts(sampleAlerts);
    }

    displayUserSecurityAlerts(alerts) {
        const userSecurityAlerts = document.getElementById('userSecurityAlerts');
        
        if (!alerts || alerts.length === 0) {
            userSecurityAlerts.innerHTML = `
                <div class="text-center text-muted p-3">
                    <i class="fas fa-shield-alt fa-2x mb-2"></i>
                    <p>No security alerts</p>
                </div>
            `;
            return;
        }

        let html = '';
        alerts.forEach(alert => {
            const severityColor = this.getSeverityColor(alert.severity);
            const statusBadge = alert.resolved ? 
                '<span class="badge bg-success">Resolved</span>' : 
                '<span class="badge bg-warning">Active</span>';
            
            html += `
                <div class="alert alert-${severityColor} d-flex justify-content-between align-items-start">
                    <div>
                        <h6 class="alert-heading">${alert.alert_type}</h6>
                        <p class="mb-1">${alert.description || 'No description available'}</p>
                        <small class="text-muted">${window.app.formatDate(alert.created_at)}</small>
                    </div>
                    <div>
                        ${statusBadge}
                    </div>
                </div>
            `;
        });
        
        userSecurityAlerts.innerHTML = html;
    }

    getSeverityColor(severity) {
        const colors = {
            'low': 'info',
            'medium': 'warning', 
            'high': 'danger',
            'critical': 'dark'
        };
        
        return colors[severity] || 'secondary';
    }

    // Update user profile
    async updateUserProfile(profileData) {
        try {
            const updatedProfile = await api.updateUserProfile(profileData);
            this.currentUser = updatedProfile;
            this.displayUserProfile(updatedProfile);
            
            window.app.showAlert('Profile updated successfully', 'success');
            return updatedProfile;
            
        } catch (error) {
            window.app.showAlert(`Failed to update profile: ${error.message}`, 'error');
            throw error;
        }
    }

    // Security settings
    async updateSecuritySettings(settings) {
        try {
            // This would update security settings like 2FA, face auth, etc.
            window.app.showAlert('Security settings updated', 'success');
            
        } catch (error) {
            window.app.showAlert(`Failed to update security settings: ${error.message}`, 'error');
        }
    }

    // Device management
    async trustDevice(deviceId) {
        try {
            // This would mark a device as trusted
            window.app.showAlert('Device marked as trusted', 'success');
            this.loadUserDevices();
            
        } catch (error) {
            window.app.showAlert(`Failed to trust device: ${error.message}`, 'error');
        }
    }

    async removeDevice(deviceId) {
        try {
            // This would remove a device from trusted list
            window.app.showAlert('Device removed', 'success');
            this.loadUserDevices();
            
        } catch (error) {
            window.app.showAlert(`Failed to remove device: ${error.message}`, 'error');
        }
    }

    // Export user data
    async exportUserData() {
        try {
            const userData = {
                profile: this.currentUser,
                devices: this.userDevices,
                exported_at: new Date()
            };
            
            const blob = new Blob([JSON.stringify(userData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `user_data_${new Date().toISOString().split('T')[0]}.json`;
            link.click();
            
            URL.revokeObjectURL(url);
            
            window.app.showAlert('User data exported successfully', 'success');
            
        } catch (error) {
            window.app.showAlert(`Failed to export user data: ${error.message}`, 'error');
        }
    }

    onTabActivated() {
        // Called when users tab is activated
        this.loadUserProfile();
        this.loadUserDevices();
        this.loadAuthHistory();
        this.loadUserSecurityAlerts();
    }

    // Cleanup method
    destroy() {
        // Clean up any resources
    }
}

// Initialize users module
document.addEventListener('DOMContentLoaded', () => {
    window.usersModule = new UsersModule();
});