/**
 * Streaming Module
 * Handles WebRTC streaming and signaling
 */

class StreamingModule {
    constructor() {
        this.peerConnection = null;
        this.localStream = null;
        this.isStreaming = false;
        this.signalingState = 'closed';
        this.streamingSession = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupWebRTC();
    }

    setupEventListeners() {
        document.getElementById('startStreamBtn').addEventListener('click', this.startStreaming.bind(this));
        document.getElementById('stopStreamBtn').addEventListener('click', this.stopStreaming.bind(this));
        document.getElementById('createOfferBtn').addEventListener('click', this.createOffer.bind(this));
        document.getElementById('createAnswerBtn').addEventListener('click', this.createAnswer.bind(this));
        document.getElementById('addIceCandidateBtn').addEventListener('click', this.addIceCandidate.bind(this));
        document.getElementById('loadStreamingSessionsBtn').addEventListener('click', this.loadStreamingSessions.bind(this));
    }

    setupWebRTC() {
        // WebRTC configuration
        this.rtcConfig = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };
    }

    async startStreaming() {
        try {
            if (!window.app.isAuthenticated) {
                window.app.showAlert('Please login first', 'error');
                return;
            }

            // Create streaming session
            const sessionData = {
                session_type: 'authentication',
                video_quality: 'high',
                frame_rate: 30,
                bitrate: 1000000, // 1Mbps
                constraints: {
                    video: { width: 640, height: 480 },
                    audio: false
                }
            };

            this.streamingSession = await api.createStreamingSession(sessionData);

            // Get user media
            const video = document.getElementById('streamingVideo');
            
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: false
            });

            video.srcObject = this.localStream;
            this.isStreaming = true;

            // Update UI
            this.updateStreamingUI(true);

            // Initialize WebRTC
            await this.initializePeerConnection();

            this.logSignaling('Streaming started successfully');
            window.app.showAlert('Streaming started!', 'success');

        } catch (error) {
            window.app.showAlert(`Failed to start streaming: ${error.message}`, 'error');
            this.stopStreaming();
        }
    }

    async stopStreaming() {
        try {
            this.isStreaming = false;

            // Stop media tracks
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => {
                    track.stop();
                });
                this.localStream = null;
            }

            // Close peer connection
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }

            // Clear video
            const video = document.getElementById('streamingVideo');
            video.srcObject = null;

            // Update UI
            this.updateStreamingUI(false);

            this.logSignaling('Streaming stopped');
            window.app.showAlert('Streaming stopped', 'info');

        } catch (error) {
            window.app.showAlert(`Error stopping streaming: ${error.message}`, 'error');
        }
    }

    async initializePeerConnection() {
        try {
            this.peerConnection = new RTCPeerConnection(this.rtcConfig);

            // Add local stream
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => {
                    this.peerConnection.addTrack(track, this.localStream);
                });
            }

            // Set up event handlers
            this.peerConnection.onicecandidate = (event) => {
                if (event.candidate) {
                    this.logSignaling('ICE Candidate generated', event.candidate);
                }
            };

            this.peerConnection.onconnectionstatechange = () => {
                this.logSignaling(`Connection state: ${this.peerConnection.connectionState}`);
            };

            this.peerConnection.oniceconnectionstatechange = () => {
                this.logSignaling(`ICE Connection state: ${this.peerConnection.iceConnectionState}`);
            };

            this.peerConnection.onsignalingstatechange = () => {
                this.signalingState = this.peerConnection.signalingState;
                this.logSignaling(`Signaling state: ${this.signalingState}`);
            };

            this.logSignaling('Peer connection initialized');

        } catch (error) {
            this.logSignaling('Failed to initialize peer connection', error.message);
            throw error;
        }
    }

    async createOffer() {
        try {
            if (!this.peerConnection) {
                window.app.showAlert('No peer connection available', 'error');
                return;
            }

            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            // Send offer to server
            const response = await api.sendWebRTCOffer(offer);
            
            this.logSignaling('Offer created and sent', offer);
            window.app.showAlert('WebRTC offer created and sent', 'success');

        } catch (error) {
            this.logSignaling('Failed to create offer', error.message);
            window.app.showAlert(`Failed to create offer: ${error.message}`, 'error');
        }
    }

    async createAnswer() {
        try {
            if (!this.peerConnection) {
                window.app.showAlert('No peer connection available', 'error');
                return;
            }

            // Simulate receiving an offer (in real implementation, this would come from signaling server)
            const simulatedOffer = {
                type: 'offer',
                sdp: 'v=0\r\no=- 123456789 2 IN IP4 127.0.0.1\r\n...' // Simulated SDP
            };

            await this.peerConnection.setRemoteDescription(simulatedOffer);
            
            const answer = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(answer);

            // Send answer to server
            const response = await api.sendWebRTCAnswer(answer);

            this.logSignaling('Answer created and sent', answer);
            window.app.showAlert('WebRTC answer created and sent', 'success');

        } catch (error) {
            this.logSignaling('Failed to create answer', error.message);
            window.app.showAlert(`Failed to create answer: ${error.message}`, 'error');
        }
    }

    async addIceCandidate() {
        try {
            if (!this.peerConnection) {
                window.app.showAlert('No peer connection available', 'error');
                return;
            }

            // Simulate ICE candidate (in real implementation, this would come from signaling)
            const simulatedCandidate = {
                candidate: 'candidate:1 1 UDP 2122252543 192.168.1.100 54400 typ host',
                sdpMid: '0',
                sdpMLineIndex: 0
            };

            await this.peerConnection.addIceCandidate(new RTCIceCandidate(simulatedCandidate));

            // Send candidate to server
            const response = await api.sendWebRTCIceCandidate(simulatedCandidate);

            this.logSignaling('ICE candidate added', simulatedCandidate);
            window.app.showAlert('ICE candidate added', 'success');

        } catch (error) {
            this.logSignaling('Failed to add ICE candidate', error.message);
            window.app.showAlert(`Failed to add ICE candidate: ${error.message}`, 'error');
        }
    }

    async loadStreamingSessions() {
        try {
            if (!window.app.isAuthenticated) return;

            const sessions = await api.getStreamingSessions();
            this.displayStreamingSessions(sessions);

        } catch (error) {
            window.app.showAlert(`Failed to load streaming sessions: ${error.message}`, 'error');
        }
    }

    displayStreamingSessions(response) {
        const sessionsList = document.getElementById('streamingSessionsList');
        sessionsList.innerHTML = '';

        // Handle both array responses and paginated responses
        let sessions = Array.isArray(response) ? response : (response?.results || []);

        if (!sessions || sessions.length === 0) {
            sessionsList.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center text-muted">No sessions found</td>
                </tr>
            `;
            return;
        }

        sessions.slice(0, 10).forEach(session => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${session.session_type || 'N/A'}</td>
                <td>
                    <span class="badge bg-${this.getSessionStatusColor(session.status)}">${session.status}</span>
                </td>
                <td>
                    <small>${window.app.formatDate(session.created_at)}</small>
                </td>
            `;
            sessionsList.appendChild(row);
        });
    }

    getSessionStatusColor(status) {
        const colors = {
            'active': 'success',
            'inactive': 'secondary',
            'ended': 'warning',
            'error': 'danger'
        };
        return colors[status] || 'info';
    }

    updateStreamingUI(streaming) {
        const startBtn = document.getElementById('startStreamBtn');
        const stopBtn = document.getElementById('stopStreamBtn');

        if (streaming) {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
        } else {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        }
    }

    logSignaling(message, data = null) {
        const timestamp = new Date().toLocaleTimeString();
        const signalingLog = document.getElementById('signalingLog');
        
        if (!signalingLog) return;

        const logEntry = document.createElement('div');
        logEntry.className = 'mb-2 p-2 border-start border-2 border-info';
        
        let content = `<strong>${timestamp}</strong> - ${message}`;
        
        if (data) {
            content += `\n<pre class="mt-1 text-muted" style="font-size: 0.8em;">${JSON.stringify(data, null, 2)}</pre>`;
        }
        
        logEntry.innerHTML = content;
        signalingLog.insertBefore(logEntry, signalingLog.firstChild);

        // Keep only last 20 entries
        while (signalingLog.children.length > 20) {
            signalingLog.removeChild(signalingLog.lastChild);
        }
    }

    // WebRTC Statistics
    async getWebRTCStats() {
        if (!this.peerConnection) {
            return null;
        }

        try {
            const stats = await this.peerConnection.getStats();
            const statsObj = {};

            stats.forEach(report => {
                if (report.type === 'inbound-rtp' || report.type === 'outbound-rtp') {
                    statsObj[report.type] = {
                        bytesReceived: report.bytesReceived,
                        bytesSent: report.bytesSent,
                        packetsReceived: report.packetsReceived,
                        packetsLost: report.packetsLost,
                        jitter: report.jitter
                    };
                }
            });

            return statsObj;
        } catch (error) {
            console.error('Failed to get WebRTC stats:', error);
            return null;
        }
    }

    onTabActivated() {
        // Called when streaming tab is activated
        this.loadStreamingSessions();
    }

    // Cleanup method
    destroy() {
        this.stopStreaming();
    }
}

// Initialize streaming module
document.addEventListener('DOMContentLoaded', () => {
    window.streamingModule = new StreamingModule();
});