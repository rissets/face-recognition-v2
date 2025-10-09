/**
 * Face Enrollment Module
 * Handles face enrollment process with camera integration
 */

class EnrollmentModule {
    constructor() {
        this.cameraManager = null;
        this.blinkDetector = null;
        this.isEnrolling = false;
        this.sessionToken = null;
        this.progressInterval = null;
        this.frameInterval = null;
        this.enrollmentStats = {
            framesProcessed: 0,
            successfulFrames: 0,
            qualityScores: []
        };
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.cameraManager = new CameraManager();
        this.blinkDetector = new BlinkDetector();
    }

    setupEventListeners() {
        document.getElementById('startEnrollmentBtn').addEventListener('click', this.startEnrollment.bind(this));
        document.getElementById('stopEnrollmentBtn').addEventListener('click', this.stopEnrollment.bind(this));
        document.getElementById('loadEnrollmentSessionsBtn').addEventListener('click', this.loadEnrollmentSessions.bind(this));
        document.getElementById('loadEmbeddingsBtn').addEventListener('click', this.loadFaceEmbeddings.bind(this));
    }

    async startEnrollment() {
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
            const video = document.getElementById('enrollmentVideo');
            const canvas = document.getElementById('enrollmentCanvas');
            
            await this.cameraManager.initializeCamera(video, canvas);

            // Create enrollment session
            const sessionResponse = await api.createEnrollmentSession({
                device_info: {
                    user_agent: navigator.userAgent,
                    platform: navigator.platform,
                    device_type: 'web',
                    browser: this.getBrowserName()
                },
                target_samples: 5
            });

            this.sessionToken = sessionResponse.session_token;
            this.isEnrolling = true;

            // Update UI
            this.updateEnrollmentUI(true);
            this.resetEnrollmentStats();

            // Hide overlay
            const overlay = document.getElementById('enrollmentOverlay');
            overlay.style.display = 'none';

            // Start processing frames
            this.startFrameProcessing();

            window.app.showAlert('Enrollment started! Look at the camera and blink naturally.', 'success');

        } catch (error) {
            console.error('Enrollment start error:', error);
            
            // Parse error message to get response data
            let errorResponse = null;
            if (error.message && error.message.includes('HTTP 400:')) {
                try {
                    const jsonPart = error.message.substring(error.message.indexOf('{'));
                    errorResponse = JSON.parse(jsonPart);
                } catch (parseError) {
                    console.error('Failed to parse error response:', parseError);
                }
            }
            
            // Handle case where there's already an active enrollment session
            if (errorResponse && errorResponse.error && errorResponse.error.includes('Active enrollment session already exists')) {
                const shouldContinue = confirm(
                    'You have an active enrollment session. Do you want to continue with the existing session? ' +
                    'Click OK to continue or Cancel to start fresh (this will end the previous session).'
                );
                
                if (shouldContinue && errorResponse.session_token) {
                    // Use the existing session token
                    this.sessionToken = errorResponse.session_token;
                    this.isEnrolling = true;
                    
                    // Update UI
                    this.updateEnrollmentUI(true);
                    this.resetEnrollmentStats();
                    
                    // Hide overlay
                    const overlay = document.getElementById('enrollmentOverlay');
                    overlay.style.display = 'none';
                    
                    // Start processing frames
                    this.startFrameProcessing();
                    
                    window.app.showAlert('Continuing with existing enrollment session.', 'info');
                    return;
                } else {
                    // User wants to start fresh - we'll need to implement session cancellation
                    window.app.showAlert('Please complete or cancel your existing enrollment session first.', 'warning');
                }
            } else {
                window.app.showAlert(`Failed to start enrollment: ${error.message}`, 'error');
            }
            
            this.stopEnrollment();
        }
    }

    async stopEnrollment() {
        try {
            this.isEnrolling = false;

            // Stop intervals
            if (this.frameInterval) {
                clearInterval(this.frameInterval);
                this.frameInterval = null;
            }

            if (this.progressInterval) {
                clearInterval(this.progressInterval);
                this.progressInterval = null;
            }

            // Stop camera
            await this.cameraManager.stopCamera();

            // Update UI
            this.updateEnrollmentUI(false);

            // Show overlay
            const overlay = document.getElementById('enrollmentOverlay');
            overlay.style.display = 'flex';

            // Reset blink detector
            this.blinkDetector.reset();

            window.app.showAlert('Enrollment stopped', 'info');

        } catch (error) {
            window.app.showAlert(`Error stopping enrollment: ${error.message}`, 'error');
        }
    }

    startFrameProcessing() {
        // Process frames every 500ms
        this.frameInterval = setInterval(async () => {
            if (!this.isEnrolling || !this.sessionToken) {
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
                const response = await FrameProcessor.processFrameForEnrollment(frameData, this.sessionToken);
                
                this.enrollmentStats.framesProcessed++;
                
                if (response.success) {
                    this.enrollmentStats.successfulFrames++;
                    
                    if (response.quality_score) {
                        this.enrollmentStats.qualityScores.push(response.quality_score);
                    }

                    // Update progress
                    this.updateProgress(response);

                    // Check if enrollment is complete
                    if (response.session_status === 'completed') {
                        window.app.showAlert('Enrollment completed successfully!', 'success');
                        this.stopEnrollment();
                        this.loadEnrollmentSessions();
                        this.loadFaceEmbeddings();
                        return;
                    }
                }

                // Update blink detection
                this.updateBlinkDetection(response);

            } catch (error) {
                console.error('Frame processing error:', error);
                
                // Don't stop enrollment for individual frame errors
                if (error.message.includes('session') || error.message.includes('expired')) {
                    window.app.showAlert('Enrollment session expired', 'error');
                    this.stopEnrollment();
                }
            }
        }, 500);
    }

    updateProgress(response) {
        const progressContainer = document.getElementById('enrollmentProgress');
        const progressBar = document.getElementById('enrollmentProgressBar');
        const statusText = document.getElementById('enrollmentStatus');

        progressContainer.style.display = 'block';

        if (response.progress_percentage !== undefined) {
            const percentage = Math.round(response.progress_percentage);
            progressBar.style.width = `${percentage}%`;
            progressBar.textContent = `${percentage}%`;
            
            if (percentage >= 100) {
                progressBar.classList.add('bg-success');
            } else if (percentage >= 50) {
                progressBar.classList.add('bg-warning');
            }
        }

        if (response.completed_samples !== undefined && response.target_samples !== undefined) {
            statusText.textContent = `Sample ${response.completed_samples} of ${response.target_samples} collected`;
        }

        // Update quality feedback
        if (response.quality_score !== undefined) {
            const qualityText = this.getQualityFeedback(response.quality_score);
            statusText.textContent += ` - ${qualityText}`;
        }
    }

    updateBlinkDetection(response) {
        const blinkIndicator = document.getElementById('blinkIndicator');
        blinkIndicator.style.display = 'block';

        // Check liveness data from server response
        if (response.liveness_data && response.liveness_data.blink_detected) {
            blinkIndicator.innerHTML = `
                <div class="blink-detected">
                    <i class="fas fa-eye"></i> Blink detected! Great job!
                </div>
            `;
            this.blinkDetector.blinkCount++;
        } else {
            // Use local blink simulation
            const blinkResult = this.blinkDetector.detectBlink();
            
            if (blinkResult.blink) {
                blinkIndicator.innerHTML = `
                    <div class="blink-detected">
                        <i class="fas fa-eye"></i> Blink detected! (${blinkResult.blinkCount} total)
                    </div>
                `;
            } else {
                blinkIndicator.innerHTML = `
                    <div class="blink-waiting">
                        <i class="fas fa-eye"></i> Please blink naturally (${blinkResult.blinkCount} detected)
                    </div>
                `;
            }
        }
    }

    getQualityFeedback(score) {
        if (score >= 0.9) return 'Excellent quality';
        if (score >= 0.7) return 'Good quality';
        if (score >= 0.5) return 'Fair quality';
        return 'Please adjust lighting or position';
    }

    updateEnrollmentUI(enrolling) {
        const startBtn = document.getElementById('startEnrollmentBtn');
        const stopBtn = document.getElementById('stopEnrollmentBtn');

        if (enrolling) {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
        } else {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
            
            // Hide progress
            const progressContainer = document.getElementById('enrollmentProgress');
            progressContainer.style.display = 'none';
            
            // Hide blink indicator
            const blinkIndicator = document.getElementById('blinkIndicator');
            blinkIndicator.style.display = 'none';
        }
    }

    resetEnrollmentStats() {
        this.enrollmentStats = {
            framesProcessed: 0,
            successfulFrames: 0,
            qualityScores: []
        };
    }

    async loadEnrollmentSessions() {
        try {
            if (!window.app.isAuthenticated) return;

            const sessions = await api.getEnrollmentSessions();
            this.displayEnrollmentSessions(sessions);
            
        } catch (error) {
            window.app.showAlert(`Failed to load enrollment sessions: ${error.message}`, 'error');
        }
    }

    displayEnrollmentSessions(sessions) {
        const sessionsList = document.getElementById('enrollmentSessionsList');
        sessionsList.innerHTML = '';

        // Handle different response formats
        let sessionArray = sessions;
        if (sessions && sessions.results) {
            // Django REST framework pagination
            sessionArray = sessions.results;
        } else if (!Array.isArray(sessions)) {
            console.error('Sessions response is not an array:', sessions);
            sessionsList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-muted">Invalid response format</td>
                </tr>
            `;
            return;
        }

        if (!sessionArray || sessionArray.length === 0) {
            sessionsList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-muted">No sessions found</td>
                </tr>
            `;
            return;
        }

        sessionArray.forEach(session => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <span class="badge bg-${this.getStatusColor(session.status)}">${session.status}</span>
                </td>
                <td>
                    <div class="progress" style="height: 10px;">
                        <div class="progress-bar" style="width: ${session.progress_percentage || 0}%"></div>
                    </div>
                    <small>${session.completed_samples || 0}/${session.target_samples || 5}</small>
                </td>
                <td>
                    <small>${window.app.formatDate(session.created_at)}</small>
                </td>
            `;
            sessionsList.appendChild(row);
        });
    }

    async loadFaceEmbeddings() {
        try {
            if (!window.app.isAuthenticated) return;

            const embeddings = await api.getFaceEmbeddings();
            this.displayFaceEmbeddings(embeddings);
            
        } catch (error) {
            window.app.showAlert(`Failed to load face embeddings: ${error.message}`, 'error');
        }
    }

    displayFaceEmbeddings(embeddings) {
        const embeddingsList = document.getElementById('embeddingsList');
        embeddingsList.innerHTML = '';

        if (!embeddings || embeddings.length === 0) {
            embeddingsList.innerHTML = `<p class="text-muted">No embeddings found</p>`;
            return;
        }

        embeddings.forEach((embedding, index) => {
            const embeddingElement = document.createElement('div');
            embeddingElement.className = 'mb-2 p-2 border rounded';
            embeddingElement.innerHTML = `
                <div class="d-flex justify-content-between">
                    <strong>Embedding #${index + 1}</strong>
                    <span class="badge bg-${embedding.is_active ? 'success' : 'secondary'}">
                        ${embedding.is_active ? 'Active' : 'Inactive'}
                    </span>
                </div>
                <small class="text-muted">
                    Quality: ${embedding.quality_score || 'N/A'} | 
                    Created: ${window.app.formatDate(embedding.created_at)}
                </small>
            `;
            embeddingsList.appendChild(embeddingElement);
        });
    }

    getStatusColor(status) {
        const colors = {
            'pending': 'warning',
            'in_progress': 'info',
            'completed': 'success',
            'failed': 'danger',
            'expired': 'secondary'
        };
        return colors[status] || 'secondary';
    }

    onTabActivated() {
        // Called when enrollment tab is activated
        this.loadEnrollmentSessions();
        this.loadFaceEmbeddings();
    }

    getBrowserName() {
        const userAgent = navigator.userAgent;
        if (userAgent.includes('Chrome')) return 'Chrome';
        if (userAgent.includes('Firefox')) return 'Firefox';
        if (userAgent.includes('Safari')) return 'Safari';
        if (userAgent.includes('Edge')) return 'Edge';
        return 'Unknown';
    }

    // Cleanup method
    destroy() {
        this.stopEnrollment();
        if (this.cameraManager) {
            this.cameraManager.stopCamera();
        }
    }
}

// Initialize enrollment module
document.addEventListener('DOMContentLoaded', () => {
    window.enrollmentModule = new EnrollmentModule();
});