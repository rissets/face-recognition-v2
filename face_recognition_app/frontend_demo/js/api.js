/**
 * API Client for Face Recognition System
 * Base URL: 127.0.0.1:8000
 */

class APIClient {
    constructor() {
        this.baseURL = 'http://127.0.0.1:8000';
        this.token = localStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token');
    }

    // Get authorization headers
    getHeaders(includeAuth = true, contentType = 'application/json') {
        const headers = {
            'Content-Type': contentType,
        };
        
        if (includeAuth && this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        
        return headers;
    }

    // Log API calls
    logAPI(method, url, request, response, status) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = {
            timestamp,
            method,
            url,
            request,
            response,
            status,
            success: status >= 200 && status < 300
        };
        
        this.displayLog(logEntry);
        console.log('API Call:', logEntry);
    }

    // Display log in UI
    displayLog(entry) {
        const logContainer = document.getElementById('globalLog');
        if (!logContainer) return;

        const logElement = document.createElement('div');
        logElement.className = `mb-2 p-2 border-start border-3 ${entry.success ? 'border-success' : 'border-danger'}`;
        
        let requestStr = '';
        let responseStr = '';
        
        try {
            requestStr = entry.request ? JSON.stringify(entry.request, null, 2) : 'No body';
            responseStr = entry.response ? JSON.stringify(entry.response, null, 2) : 'No response';
        } catch (e) {
            requestStr = String(entry.request) || 'No body';
            responseStr = String(entry.response) || 'No response';
        }

        logElement.innerHTML = `
            <div class="d-flex justify-content-between">
                <strong>${entry.method} ${entry.url}</strong>
                <small class="text-muted">${entry.timestamp}</small>
            </div>
            <div class="mt-1">
                <span class="badge bg-${entry.success ? 'success' : 'danger'}">${entry.status}</span>
            </div>
            <details class="mt-2">
                <summary>Details</summary>
                <div class="mt-2">
                    <strong>Request:</strong>
                    <pre class="mt-1 text-muted" style="font-size: 0.8em; white-space: pre-wrap;">${requestStr}</pre>
                    <strong>Response:</strong>
                    <pre class="mt-1 text-muted" style="font-size: 0.8em; white-space: pre-wrap;">${responseStr}</pre>
                </div>
            </details>
        `;

        logContainer.insertBefore(logElement, logContainer.firstChild);

        // Keep only last 50 entries
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }

    // Generic API request method
    async request(method, endpoint, data = null, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        const config = {
            method: method.toUpperCase(),
            headers: this.getHeaders(options.includeAuth !== false, options.contentType),
            ...options.fetchOptions
        };

        if (data && ['POST', 'PUT', 'PATCH'].includes(config.method)) {
            if (config.headers['Content-Type'] === 'application/json') {
                config.body = JSON.stringify(data);
            } else {
                config.body = data;
            }
        }

        try {
            const response = await fetch(url, config);
            let responseData = null;

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                responseData = await response.json();
            } else {
                responseData = await response.text();
            }

            this.logAPI(method.toUpperCase(), endpoint, data, responseData, response.status);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${JSON.stringify(responseData)}`);
            }

            return responseData;
        } catch (error) {
            this.logAPI(method.toUpperCase(), endpoint, data, error.message, 0);
            throw error;
        }
    }

    // Authentication methods
    async register(userData) {
        return await this.request('POST', '/api/auth/register/', userData, { includeAuth: false });
    }

    async login(email, password) {
        const response = await this.request('POST', '/api/auth/token/', { 
            email: email,  // Our USERNAME_FIELD is email
            password: password 
        }, { includeAuth: false });
        
        if (response.access) {
            this.token = response.access;
            this.refreshToken = response.refresh;
            localStorage.setItem('access_token', this.token);
            localStorage.setItem('refresh_token', this.refreshToken);
        }
        
        return response;
    }

    async logout() {
        this.token = null;
        this.refreshToken = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
    }

    async getUserProfile() {
        return await this.request('GET', '/api/auth/profile/');
    }

    async updateUserProfile(profileData) {
        return await this.request('PATCH', '/api/auth/profile/', profileData);
    }

    // Enrollment methods
    async createEnrollmentSession(sessionData = {}) {
        return await this.request('POST', '/api/enrollment/create/', sessionData);
    }

    async processEnrollmentFrame(sessionToken, frameData) {
        return await this.request('POST', '/api/enrollment/process-frame/', {
            session_token: sessionToken,
            frame_data: frameData
        });
    }

    // Authentication methods
    async createAuthenticationSession(sessionData = {}) {
        return await this.request('POST', '/api/auth/face/create/', sessionData);
    }

    async processAuthenticationFrame(sessionToken, frameData) {
        return await this.request('POST', '/api/auth/face/process-frame/', {
            session_token: sessionToken,
            frame_data: frameData
        });
    }

    // Recognition methods
    async getFaceEmbeddings() {
        return await this.request('GET', '/api/recognition/embeddings/');
    }

    async getFaceEmbedding(id) {
        return await this.request('GET', `/api/recognition/embeddings/${id}/`);
    }

    async getEnrollmentSessions() {
        return await this.request('GET', '/api/recognition/sessions/');
    }

    async getEnrollmentSession(id) {
        return await this.request('GET', `/api/recognition/sessions/${id}/`);
    }

    async getAuthenticationAttempts() {
        return await this.request('GET', '/api/recognition/attempts/');
    }

    async getAuthenticationAttempt(id) {
        return await this.request('GET', `/api/recognition/attempts/${id}/`);
    }

    // Streaming methods
    async getStreamingSessions() {
        return await this.request('GET', '/api/streaming/sessions/');
    }

    async createStreamingSession(sessionData) {
        return await this.request('POST', '/api/streaming/sessions/create/', sessionData);
    }

    async getStreamingSession(id) {
        return await this.request('GET', `/api/streaming/sessions/${id}/`);
    }

    async sendWebRTCSignal(signalData) {
        return await this.request('POST', '/api/streaming/signaling/', signalData);
    }

    // Analytics methods
    async getAuthenticationLogs() {
        return await this.request('GET', '/api/analytics/auth-logs/');
    }

    async getAuthenticationLog(id) {
        return await this.request('GET', `/api/analytics/auth-logs/${id}/`);
    }

    async getSecurityAlerts() {
        return await this.request('GET', '/api/analytics/security-alerts/');
    }

    async getSecurityAlert(id) {
        return await this.request('GET', `/api/analytics/security-alerts/${id}/`);
    }

    async getAnalyticsDashboard() {
        return await this.request('GET', '/api/analytics/dashboard/');
    }

    async getStatistics() {
        return await this.request('GET', '/api/analytics/statistics/');
    }

    // User methods
    async getUserDevices() {
        return await this.request('GET', '/api/users/devices/');
    }

    async getUserDevice(id) {
        return await this.request('GET', `/api/users/devices/${id}/`);
    }

    async getUserAuthHistory() {
        return await this.request('GET', '/api/user/auth-history/');
    }

    async getUserSecurityAlerts() {
        return await this.request('GET', '/api/user/security-alerts/');
    }

    // System methods
    async getSystemStatus() {
        return await this.request('GET', '/api/system/status/');
    }

    async sendWebRTCOffer(offer) {
        return await this.request('POST', '/api/webrtc/signal/', {
            type: 'offer',
            signal: offer
        });
    }

    async sendWebRTCAnswer(answer) {
        return await this.request('POST', '/api/webrtc/signal/', {
            type: 'answer',  
            signal: answer
        });
    }

    async sendWebRTCIceCandidate(candidate) {
        return await this.request('POST', '/api/webrtc/signal/', {
            type: 'ice-candidate',
            signal: candidate
        });
    }

    // Custom endpoint testing
    async testCustomEndpoint(method, endpoint, data = null) {
        return await this.request(method, endpoint, data);
    }

    // Utility methods
    isAuthenticated() {
        return !!this.token;
    }

    setAuthToken(token) {
        this.token = token;
        localStorage.setItem('access_token', token);
    }

    clearAuthToken() {
        this.token = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
    }
}

// Create global API client instance
const api = new APIClient();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APIClient, api };
}