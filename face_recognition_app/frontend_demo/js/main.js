/**
 * Main Application Logic
 * Handles UI initialization, authentication state, and general app functions
 */

class FaceRecognitionApp {
    constructor() {
        this.isAuthenticated = false;
        this.currentUser = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAuthState();
        this.initializeUI();
    }

    setupEventListeners() {
        // Authentication forms
        document.getElementById('registrationForm').addEventListener('submit', this.handleRegistration.bind(this));
        document.getElementById('loginForm').addEventListener('submit', this.handleLogin.bind(this));
        document.getElementById('logoutBtn').addEventListener('click', this.handleLogout.bind(this));

        // Clear log button
        document.getElementById('clearLogBtn').addEventListener('click', this.clearLog.bind(this));

        // Tab change events
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', this.handleTabChange.bind(this));
        });

        // Load user profile button
        document.getElementById('loadUserProfileBtn').addEventListener('click', this.loadUserProfile.bind(this));

        // System status
        document.getElementById('loadSystemStatusBtn').addEventListener('click', this.loadSystemStatus.bind(this));

        // Custom API testing
        document.getElementById('testCustomEndpointBtn').addEventListener('click', this.testCustomEndpoint.bind(this));

        // Auto-refresh token
        this.setupTokenRefresh();
    }

    setupTokenRefresh() {
        // Refresh token every 10 minutes if authenticated
        setInterval(() => {
            if (this.isAuthenticated) {
                this.refreshAuthToken();
            }
        }, 10 * 60 * 1000);
    }

    async refreshAuthToken() {
        try {
            if (api.refreshToken) {
                const response = await api.request('POST', '/api/auth/token/refresh/', {
                    refresh: api.refreshToken
                }, { includeAuth: false });

                if (response.access) {
                    api.setAuthToken(response.access);
                    this.showAlert('Token refreshed successfully', 'success');
                }
            }
        } catch (error) {
            console.warn('Token refresh failed:', error);
            // If refresh fails, logout user
            this.handleLogout();
        }
    }

    checkAuthState() {
        const token = localStorage.getItem('access_token');
        if (token) {
            api.setAuthToken(token);
            this.isAuthenticated = true;
            this.updateAuthUI();
            this.loadUserProfile();
        }
    }

    initializeUI() {
        this.updateAuthUI();
        this.showAlert('Face Recognition System Demo loaded. Please login to access all features.', 'info');
    }

    updateAuthUI() {
        const authStatus = document.getElementById('authStatus');
        const logoutBtn = document.getElementById('logoutBtn');
        
        if (this.isAuthenticated) {
            authStatus.textContent = `Authenticated${this.currentUser ? ` as ${this.currentUser.email}` : ''}`;
            logoutBtn.style.display = 'inline-block';
            
            // Enable authentication-required tabs
            this.enableProtectedTabs();
        } else {
            authStatus.textContent = 'Not Authenticated';
            logoutBtn.style.display = 'none';
            
            // Disable authentication-required tabs
            this.disableProtectedTabs();
        }
    }

    enableProtectedTabs() {
        const protectedTabs = ['enrollment-tab', 'recognition-tab', 'streaming-tab', 'analytics-tab', 'users-tab'];
        protectedTabs.forEach(tabId => {
            const tab = document.getElementById(tabId);
            if (tab) {
                tab.classList.remove('disabled');
                tab.removeAttribute('disabled');
            }
        });
    }

    disableProtectedTabs() {
        const protectedTabs = ['enrollment-tab', 'recognition-tab', 'streaming-tab', 'analytics-tab', 'users-tab'];
        protectedTabs.forEach(tabId => {
            const tab = document.getElementById(tabId);
            if (tab) {
                tab.classList.add('disabled');
                tab.setAttribute('disabled', 'true');
            }
        });
        
        // Switch to auth tab
        const authTab = document.getElementById('auth-tab');
        if (authTab) {
            authTab.click();
        }
    }

    async handleRegistration(event) {
        event.preventDefault();
        
        const email = document.getElementById('regEmail').value;
        const firstName = document.getElementById('regFirstName').value;
        const lastName = document.getElementById('regLastName').value;
        const password = document.getElementById('regPassword').value;
        const passwordConfirm = document.getElementById('regPasswordConfirm').value;

        if (password !== passwordConfirm) {
            this.showAlert('Passwords do not match', 'error');
            return;
        }

        try {
            const userData = {
                email,
                first_name: firstName,
                last_name: lastName,
                password,
                password_confirm: passwordConfirm
            };

            const response = await api.register(userData);
            this.showAlert('Registration successful! You can now login.', 'success');
            
            // Clear form
            document.getElementById('registrationForm').reset();
            
        } catch (error) {
            this.showAlert(`Registration failed: ${error.message}`, 'error');
        }
    }

    async handleLogin(event) {
        event.preventDefault();
        
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;

        try {
            const response = await api.login(email, password);
            
            this.isAuthenticated = true;
            this.updateAuthUI();
            this.showAlert('Login successful!', 'success');
            
            // Clear form
            document.getElementById('loginForm').reset();
            
            // Load user profile
            await this.loadUserProfile();
            
        } catch (error) {
            this.showAlert(`Login failed: ${error.message}`, 'error');
        }
    }

    async handleLogout() {
        try {
            await api.logout();
            this.isAuthenticated = false;
            this.currentUser = null;
            this.updateAuthUI();
            this.showAlert('Logged out successfully', 'success');
            
            // Clear user profile display
            const userProfile = document.getElementById('userProfile');
            if (userProfile) {
                userProfile.style.display = 'none';
            }
            
        } catch (error) {
            this.showAlert(`Logout error: ${error.message}`, 'error');
        }
    }

    async loadUserProfile() {
        if (!this.isAuthenticated) return;

        try {
            const profile = await api.getUserProfile();
            this.currentUser = profile;
            
            // Display profile
            const userProfile = document.getElementById('userProfile');
            const profileData = document.getElementById('profileData');
            
            if (userProfile && profileData) {
                profileData.textContent = JSON.stringify(profile, null, 2);
                userProfile.style.display = 'block';
            }
            
            // Update user profile content in users tab
            const userProfileContent = document.getElementById('userProfileContent');
            if (userProfileContent) {
                userProfileContent.textContent = JSON.stringify(profile, null, 2);
            }
            
            this.updateAuthUI();
            
        } catch (error) {
            this.showAlert(`Failed to load user profile: ${error.message}`, 'error');
        }
    }

    async loadSystemStatus() {
        try {
            const status = await api.getSystemStatus();
            
            const systemStatusContent = document.getElementById('systemStatusContent');
            if (systemStatusContent) {
                systemStatusContent.textContent = JSON.stringify(status, null, 2);
            }
            
            this.showAlert('System status loaded successfully', 'success');
            
        } catch (error) {
            this.showAlert(`Failed to load system status: ${error.message}`, 'error');
        }
    }

    async testCustomEndpoint() {
        const endpoint = document.getElementById('customEndpoint').value;
        const method = document.getElementById('httpMethod').value;
        const bodyText = document.getElementById('requestBody').value;
        
        if (!endpoint) {
            this.showAlert('Please enter an endpoint', 'error');
            return;
        }

        try {
            let requestData = null;
            if (bodyText.trim()) {
                try {
                    requestData = JSON.parse(bodyText);
                } catch (e) {
                    this.showAlert('Invalid JSON in request body', 'error');
                    return;
                }
            }

            const response = await api.testCustomEndpoint(method, endpoint, requestData);
            
            const apiResponse = document.getElementById('apiResponse');
            if (apiResponse) {
                apiResponse.textContent = JSON.stringify(response, null, 2);
            }
            
            this.showAlert('Custom endpoint test completed', 'success');
            
        } catch (error) {
            const apiResponse = document.getElementById('apiResponse');
            if (apiResponse) {
                apiResponse.textContent = error.message;
            }
            this.showAlert(`Custom endpoint test failed: ${error.message}`, 'error');
        }
    }

    handleTabChange(event) {
        const targetTab = event.target.getAttribute('data-bs-target');
        
        // Check if user needs to be authenticated for this tab
        const protectedTabs = ['#enrollment', '#recognition', '#streaming', '#analytics', '#users'];
        
        if (protectedTabs.includes(targetTab) && !this.isAuthenticated) {
            event.preventDefault();
            this.showAlert('Please login to access this feature', 'warning');
            
            // Switch back to auth tab
            const authTab = document.getElementById('auth-tab');
            if (authTab) {
                setTimeout(() => authTab.click(), 100);
            }
            return;
        }

        // Initialize tab-specific functionality
        switch (targetTab) {
            case '#enrollment':
                if (window.enrollmentModule) {
                    window.enrollmentModule.onTabActivated();
                }
                break;
            case '#recognition':
                if (window.authenticationModule) {
                    window.authenticationModule.onTabActivated();
                }
                break;
            case '#streaming':
                if (window.streamingModule) {
                    window.streamingModule.onTabActivated();
                }
                break;
            case '#analytics':
                if (window.analyticsModule) {
                    window.analyticsModule.onTabActivated();
                }
                break;
            case '#users':
                if (window.usersModule) {
                    window.usersModule.onTabActivated();
                }
                break;
        }
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;

        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${this.getBootstrapAlertClass(type)} alert-dismissible fade show`;
        alertElement.setAttribute('role', 'alert');
        
        const icon = this.getAlertIcon(type);
        
        alertElement.innerHTML = `
            <i class="${icon}"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        alertContainer.appendChild(alertElement);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertElement.parentNode) {
                alertElement.remove();
            }
        }, 5000);
    }

    getBootstrapAlertClass(type) {
        const alertClasses = {
            'success': 'success',
            'error': 'danger',
            'warning': 'warning',
            'info': 'info'
        };
        return alertClasses[type] || 'info';
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };
        return icons[type] || 'fas fa-info-circle';
    }

    clearLog() {
        const globalLog = document.getElementById('globalLog');
        if (globalLog) {
            globalLog.innerHTML = '';
        }
    }

    // Utility methods
    formatDate(dateString) {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleString();
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDuration(seconds) {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hrs > 0) {
            return `${hrs}h ${mins}m ${secs}s`;
        } else if (mins > 0) {
            return `${mins}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    generateColor(index) {
        const colors = [
            '#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8',
            '#6610f2', '#e83e8c', '#fd7e14', '#20c997', '#6f42c1'
        ];
        return colors[index % colors.length];
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FaceRecognitionApp();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceRecognitionApp;
}