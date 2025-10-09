/**
 * Face Authentication Module
 * Handles face authentication process
 */

class AuthenticationModule {
    constructor() {
        this.cameraManager = null;
        this.isAuthenticating = false;
        this.sessionToken = null;
        this.frameInterval = null;
        this.authStats = {
            framesProcessed: 0,
            attempts: 0,
            lastResult: null,
            errors: 0
        };
        this.sessionRetries = 0;
        this.maxRetries = 3;
        this.lastFailureReason = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.cameraManager = new CameraManager();
    }

    setupEventListeners() {
        document.getElementById('startAuthBtn').addEventListener('click', this.startAuthentication.bind(this));
        document.getElementById('stopAuthBtn').addEventListener('click', this.stopAuthentication.bind(this));
        document.getElementById('loadAttemptsBtn').addEventListener('click', this.loadAuthenticationAttempts.bind(this));
    }

    async startAuthentication() {
        try {
            if (!window.app.isAuthenticated) {
                window.app.showAlert('Please login first', 'error');
                return;
            }

            // Check camera availability
            const hasCamera = await CameraManager.checkCameraAvailability();
            if (!hasCamera) {
                window.app.showAlert('No camera available', 'error');
                return;
            }

            // Initialize camera
            const video = document.getElementById('authVideo');
            const canvas = document.getElementById('authCanvas');
            
            await this.cameraManager.initializeCamera(video, canvas);

            // Create authentication session
            const sessionResponse = await api.createAuthenticationSession({
                session_type: 'identification',
                device_info: {
                    user_agent: navigator.userAgent,
                    platform: navigator.platform,
                    device_type: 'web',
                    browser: this.getBrowserName()
                }
            });

            this.sessionToken = sessionResponse.session_token;
            this.isAuthenticating = true;

            // Reset retry counter on successful session start
            this.sessionRetries = 0;
            this.lastFailureReason = null;

            // Update UI
            this.updateAuthUI(true);
            this.resetAuthStats();

            // Hide overlay
            const overlay = document.getElementById('authOverlay');
            overlay.style.display = 'none';

            // Start processing frames
            this.startFrameProcessing();

            window.app.showAlert('Authentication started! Look at the camera.', 'success');

        } catch (error) {
            window.app.showAlert(`Failed to start authentication: ${error.message}`, 'error');
            this.stopAuthentication();
        }
    }

    async stopAuthentication() {
        try {
            this.isAuthenticating = false;

            // Stop intervals
            if (this.frameInterval) {
                clearInterval(this.frameInterval);
                this.frameInterval = null;
            }

            // Stop camera
            await this.cameraManager.stopCamera();

            // Update UI
            this.updateAuthUI(false);

            // Show overlay
            const overlay = document.getElementById('authOverlay');
            overlay.style.display = 'flex';

            window.app.showAlert('Authentication stopped', 'info');

        } catch (error) {
            window.app.showAlert(`Error stopping authentication: ${error.message}`, 'error');
        }
    }

    startFrameProcessing() {
        // Process frames every 1 second
        this.frameInterval = setInterval(async () => {
            if (!this.isAuthenticating || !this.sessionToken) {
                return;
            }

            try {
                // Capture frame
                const frameData = this.cameraManager.captureFrame();
                
                // Validate frame
                const validation = FrameProcessor.validateFrame(frameData);
                if (!validation.valid) {
                    console.warn('Invalid frame:', validation.error);
                    return;
                }

                // Process frame
                const response = await FrameProcessor.processFrameForAuthentication(frameData, this.sessionToken);
                
                this.authStats.framesProcessed++;
                this.authStats.attempts++;
                this.authStats.lastResult = response;

                // Display result
                this.displayAuthenticationResult(response);

                // If authentication succeeded or failed definitively, stop
                if (response.authenticated !== undefined) {
                    if (response.authenticated) {
                        window.app.showAlert('Authentication successful!', 'success');
                    } else if (response.max_attempts_reached) {
                        window.app.showAlert('Authentication failed - max attempts reached', 'error');
                    }
                    
                    // Load attempts list
                    this.loadAuthenticationAttempts();
                    
                    // Stop after successful auth or max attempts
                    if (response.authenticated || response.max_attempts_reached) {
                        setTimeout(() => this.stopAuthentication(), 2000);
                    }
                }

            } catch (error) {
                console.error('Authentication frame processing error:', error);
                
                // Handle session-related errors
                if (error.message.includes('Invalid or expired session')) {
                    this.lastFailureReason = 'Session expired';
                    this.restartAuthenticationSession();
                    return;
                } else if (error.message.includes('session')) {
                    // Parse error message for better feedback
                    let errorResponse = null;
                    let errorMessage = 'Session error occurred';
                    
                    if (error.message.includes('HTTP 400:')) {
                        try {
                            const jsonPart = error.message.substring(error.message.indexOf('{'));
                            errorResponse = JSON.parse(jsonPart);
                            errorMessage = errorResponse.error || errorMessage;
                        } catch (parseError) {
                            console.error('Failed to parse error response:', parseError);
                        }
                    }
                    
                    // Store failure reason
                    this.lastFailureReason = errorMessage;
                    
                    // Check for errors that require session restart
                    const requiresRestart = 
                        (errorResponse && errorResponse.requires_new_session === true) ||
                        errorMessage.includes('Session has ended') || 
                        errorMessage.includes('failed') || 
                        errorMessage.includes('completed') ||
                        errorMessage.includes('status:');
                    
                    if (requiresRestart) {
                        // Special handling for liveness check failures
                        if (errorMessage.includes('Liveness check failed')) {
                            // Show specific liveness guidance
                            if (this.sessionRetries < 2) {
                                console.log(`Auto-restarting due to liveness failure (attempt ${this.sessionRetries + 1})`);
                                window.app.showAlert(
                                    `Liveness check failed. Please blink naturally and look directly at the camera. Retrying... (${this.sessionRetries + 1}/3)`, 
                                    'warning'
                                );
                                // Longer delay for liveness issues
                                setTimeout(() => this.restartAuthenticationSession(), 3000);
                                return;
                            } else {
                                // Ask user for liveness failures after 2 attempts
                                const shouldRestart = confirm(
                                    `Liveness Detection Issue (${this.sessionRetries + 1}/3):\n\n` +
                                    `The system couldn't detect natural blinking. This helps verify you're a real person.\n\n` +
                                    `Tips for better detection:\n` +
                                    `• Look directly at the camera\n` +
                                    `• Blink naturally (not forced)\n` +
                                    `• Ensure good lighting on your face\n` +
                                    `• Keep your face clearly visible\n` +
                                    `• Stay still and centered in frame\n\n` +
                                    `Would you like to try again?`
                                );
                                if (shouldRestart) {
                                    this.restartAuthenticationSession();
                                } else {
                                    this.stopAuthentication();
                                }
                                return;
                            }
                        }
                        
                        // Handle other session restart errors
                        if (this.sessionRetries < 2) {
                            console.log(`Auto-restarting session due to: ${errorMessage} (attempt ${this.sessionRetries + 1})`);
                            window.app.showAlert(`Session issue detected: ${errorMessage}. Restarting... (${this.sessionRetries + 1}/3)`, 'warning');
                            // Standard delay for other errors
                            setTimeout(() => this.restartAuthenticationSession(), 2000);
                        } else {
                            // Ask user after multiple failures
                            const shouldRestart = confirm(
                                `Authentication session error (attempt ${this.sessionRetries + 1}/3):\n\n` +
                                `${errorMessage}\n\n` +
                                `Would you like to start a new session?`
                            );
                            if (shouldRestart) {
                                this.restartAuthenticationSession();
                            } else {
                                this.stopAuthentication();
                            }
                        }
                        return;
                    }
                    
                    // For other session errors, show warning but continue
                    window.app.showAlert(`Session warning: ${errorMessage}`, 'warning');
                    return;
                }
                
                // For other errors, continue processing but show warning
                this.authStats.errors++;
                if (this.authStats.errors > 5) {
                    window.app.showAlert('Too many processing errors. Please try again.', 'error');
                    this.stopAuthentication();
                }
            }
        }, 1000);
    }

    displayAuthenticationResult(result) {
        const authResults = document.getElementById('authResults');
        const authResultContent = document.getElementById('authResultContent');

        authResults.style.display = 'block';

        let statusBadge = '';
        let resultText = '';
        let detailsText = '';

        if (result.authenticated === true) {
            statusBadge = '<span class="badge bg-success">AUTHENTICATED</span>';
            resultText = `
                <h6 class="text-success">
                    <i class="fas fa-check-circle"></i> Authentication Successful
                </h6>
            `;
            
            if (result.user_id) {
                detailsText += `<p><strong>User ID:</strong> ${result.user_id}</p>`;
            }
            
            if (result.confidence_score) {
                detailsText += `<p><strong>Confidence:</strong> ${(result.confidence_score * 100).toFixed(1)}%</p>`;
            }
            
        } else if (result.authenticated === false) {
            statusBadge = '<span class="badge bg-danger">FAILED</span>';
            resultText = `
                <h6 class="text-danger">
                    <i class="fas fa-times-circle"></i> Authentication Failed
                </h6>
            `;
            
            if (result.error) {
                detailsText += `<p><strong>Reason:</strong> ${result.error}</p>`;
            }
            
        } else {
            statusBadge = '<span class="badge bg-warning">PROCESSING</span>';
            resultText = `
                <h6 class="text-warning">
                    <i class="fas fa-spinner fa-spin"></i> Processing...
                </h6>
            `;
        }

        // Add common details
        if (result.liveness_data) {
            const liveness = result.liveness_data;
            detailsText += `
                <p><strong>Liveness Check:</strong> ${liveness.liveness_passed ? 
                    '<span class="text-success">Passed</span>' : 
                    '<span class="text-danger">Failed</span>'}</p>
            `;
        }

        if (result.quality_score) {
            detailsText += `<p><strong>Image Quality:</strong> ${(result.quality_score * 100).toFixed(1)}%</p>`;
        }

        if (result.processing_time) {
            detailsText += `<p><strong>Processing Time:</strong> ${result.processing_time}ms</p>`;
        }

        // Attempt count
        detailsText += `<p><strong>Attempt:</strong> ${this.authStats.attempts}</p>`;

        authResultContent.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <div>${resultText}</div>
                <div>${statusBadge}</div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    ${detailsText}
                </div>
                <div class="col-md-6">
                    <h6>Session Statistics</h6>
                    <ul class="list-unstyled">
                        <li><strong>Frames Processed:</strong> ${this.authStats.framesProcessed}</li>
                        <li><strong>Total Attempts:</strong> ${this.authStats.attempts}</li>
                    </ul>
                </div>
            </div>
        `;
    }

    updateAuthUI(authenticating) {
        const startBtn = document.getElementById('startAuthBtn');
        const stopBtn = document.getElementById('stopAuthBtn');

        if (authenticating) {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
            
            // Update button text with retry info if applicable
            if (this.sessionRetries > 0) {
                stopBtn.innerHTML = `<i class="fas fa-stop"></i> Stop (Retry ${this.sessionRetries}/${this.maxRetries + 1})`;
            } else {
                stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Authentication';
            }
        } else {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
            stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Authentication';
            
            // Hide results after delay
            setTimeout(() => {
                const authResults = document.getElementById('authResults');
                authResults.style.display = 'none';
            }, 5000);
        }
    }

    resetAuthStats() {
        this.authStats = {
            framesProcessed: 0,
            attempts: 0,
            lastResult: null,
            errors: 0
        };
    }

    async loadAuthenticationAttempts() {
        try {
            if (!window.app.isAuthenticated) return;

            const attempts = await api.getAuthenticationAttempts();
            this.displayAuthenticationAttempts(attempts);
            
        } catch (error) {
            window.app.showAlert(`Failed to load authentication attempts: ${error.message}`, 'error');
        }
    }

    displayAuthenticationAttempts(response) {
        const attemptsList = document.getElementById('attemptsList');
        attemptsList.innerHTML = '';

        // Handle both array responses and paginated responses
        let attempts = Array.isArray(response) ? response : (response?.results || []);

        if (!attempts || attempts.length === 0) {
            attemptsList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-muted">No attempts found</td>
                </tr>
            `;
            return;
        }

        attempts.slice(0, 10).forEach(attempt => {
            const row = document.createElement('tr');
            const resultBadge = attempt.success ? 
                '<span class="badge bg-success">Success</span>' : 
                '<span class="badge bg-danger">Failed</span>';
            
            const score = attempt.confidence_score ? 
                (attempt.confidence_score * 100).toFixed(1) + '%' : 
                'N/A';

            row.innerHTML = `
                <td>${resultBadge}</td>
                <td>${score}</td>
                <td>
                    <small>${window.app.formatDate(attempt.created_at)}</small>
                </td>
            `;
            attemptsList.appendChild(row);
        });
    }

    onTabActivated() {
        // Called when authentication tab is activated
        this.loadAuthenticationAttempts();
    }

    async restartAuthenticationSession() {
        // Check retry limit
        if (this.sessionRetries >= this.maxRetries) {
            const userWantsRetry = confirm(
                `Authentication session has failed ${this.maxRetries} times.\n\n` +
                `Last failure: ${this.lastFailureReason || 'Unknown error'}\n\n` +
                `Would you like to reset and try again?`
            );
            
            if (userWantsRetry) {
                this.sessionRetries = 0;
                this.lastFailureReason = null;
            } else {
                this.stopAuthentication();
                window.app.showAlert('Authentication cancelled after multiple failures', 'warning');
                return;
            }
        }
        
        this.sessionRetries++;
        
        try {
            console.log(`Restarting authentication session (attempt ${this.sessionRetries}/${this.maxRetries + 1})`);
            
            // Stop current session completely and clear all intervals
            await this.stopAuthentication();
            
            // Clear any existing session token
            this.sessionToken = null;
            
            // Wait longer between retries with incremental backoff
            const baseWait = this.lastFailureReason?.includes('Liveness') ? 3000 : 2000;
            const waitTime = baseWait + (1000 * (this.sessionRetries - 1));
            console.log(`Waiting ${waitTime}ms before restart...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            
            // Reset stats but keep retry counter
            this.resetAuthStats();
            
            // Start new session
            await this.startAuthentication();
            
            window.app.showAlert(
                `New authentication session started (attempt ${this.sessionRetries}/${this.maxRetries + 1})`, 
                'info'
            );
            
        } catch (error) {
            console.error('Failed to restart authentication session:', error);
            this.lastFailureReason = error.message;
            
            if (this.sessionRetries < this.maxRetries) {
                window.app.showAlert(
                    `Session restart failed. Will retry... (${this.sessionRetries}/${this.maxRetries + 1})`, 
                    'warning'
                );
                // Auto-retry with longer delay
                setTimeout(() => this.restartAuthenticationSession(), 5000);
            } else {
                window.app.showAlert(
                    'Failed to restart authentication session after multiple attempts', 
                    'error'
                );
                this.stopAuthentication();
            }
        }
    }

    getBrowserName() {
        const userAgent = navigator.userAgent;
        if (userAgent.includes('Chrome')) return 'Chrome';
        if (userAgent.includes('Firefox')) return 'Firefox';
        if (userAgent.includes('Safari')) return 'Safari';
        if (userAgent.includes('Edge')) return 'Edge';
        return 'Unknown';
    }

    showLivenessGuidance() {
        // Show modal or alert with detailed liveness guidance
        const guidance = `
        LIVENESS DETECTION TIPS:
        
        ✓ Look directly at the camera
        ✓ Blink naturally (don't force it)
        ✓ Keep your face clearly visible
        ✓ Ensure good lighting on your face
        ✓ Stay still but relaxed
        ✓ Don't wear sunglasses or caps
        
        The system needs to see natural blinking to verify you're real.
        `;
        
        window.app.showAlert(guidance, 'info');
    }

    // Cleanup method
    destroy() {
        this.stopAuthentication();
        if (this.cameraManager) {
            this.cameraManager.stopCamera();
        }
    }
}

// Initialize authentication module
document.addEventListener('DOMContentLoaded', () => {
    window.authenticationModule = new AuthenticationModule();
});