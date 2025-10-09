/**
 * Camera Utilities
 * Handles camera access, video streaming, and frame capture
 */

class CameraManager {
    constructor() {
        this.stream = null;
        this.video = null;
        this.canvas = null;
        this.context = null;
        this.isStreaming = false;
        this.constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        };
    }

    // Initialize camera for a specific video element
    async initializeCamera(videoElement, canvasElement = null) {
        this.video = videoElement;
        this.canvas = canvasElement;
        
        if (this.canvas) {
            this.context = this.canvas.getContext('2d');
        }

        try {
            // Stop any existing stream
            await this.stopCamera();
            
            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia(this.constraints);
            
            // Set video source
            this.video.srcObject = this.stream;
            this.isStreaming = true;
            
            // Wait for video to load
            await new Promise((resolve) => {
                this.video.addEventListener('loadedmetadata', resolve, { once: true });
            });
            
            return true;
        } catch (error) {
            console.error('Camera initialization failed:', error);
            throw new Error(`Camera access failed: ${error.message}`);
        }
    }

    // Stop camera stream
    async stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.stop();
            });
            this.stream = null;
        }
        
        if (this.video) {
            this.video.srcObject = null;
        }
        
        this.isStreaming = false;
    }

    // Capture frame from video
    captureFrame() {
        if (!this.video || !this.canvas || !this.context || !this.isStreaming) {
            throw new Error('Camera not properly initialized');
        }

        // Set canvas size to match video
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;

        // Draw current video frame to canvas
        this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to base64 JPEG
        const frameData = this.canvas.toDataURL('image/jpeg', 0.8);
        
        // Remove data URL prefix
        return frameData.split(',')[1];
    }

    // Get video stats
    getVideoStats() {
        if (!this.video || !this.isStreaming) {
            return null;
        }

        return {
            width: this.video.videoWidth,
            height: this.video.videoHeight,
            readyState: this.video.readyState,
            currentTime: this.video.currentTime,
            duration: this.video.duration || 0
        };
    }

    // Check camera availability
    static async checkCameraAvailability() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter(device => device.kind === 'videoinput');
            return cameras.length > 0;
        } catch (error) {
            console.error('Failed to check camera availability:', error);
            return false;
        }
    }

    // Get available cameras
    static async getAvailableCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Failed to get available cameras:', error);
            return [];
        }
    }

    // Switch camera
    async switchCamera(deviceId) {
        if (deviceId) {
            this.constraints.video.deviceId = { exact: deviceId };
        } else {
            delete this.constraints.video.deviceId;
        }

        if (this.video) {
            await this.initializeCamera(this.video, this.canvas);
        }
    }

    // Update video constraints
    updateConstraints(newConstraints) {
        this.constraints = { ...this.constraints, ...newConstraints };
    }

    // Take photo and download
    takePhoto(filename = 'photo.jpg') {
        if (!this.video || !this.canvas || !this.context || !this.isStreaming) {
            throw new Error('Camera not properly initialized');
        }

        // Capture frame
        this.captureFrame();
        
        // Create download link
        this.canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            link.click();
            URL.revokeObjectURL(url);
        }, 'image/jpeg', 0.8);
    }

    // Get media stream for WebRTC
    getMediaStream() {
        return this.stream;
    }

    // Check if camera is active
    isActive() {
        return this.isStreaming && this.stream && this.stream.active;
    }
}

// Frame processing utilities
class FrameProcessor {
    static async processFrameForEnrollment(frameData, sessionToken) {
        try {
            const response = await api.processEnrollmentFrame(sessionToken, frameData);
            return response;
        } catch (error) {
            console.error('Enrollment frame processing failed:', error);
            throw error;
        }
    }

    static async processFrameForAuthentication(frameData, sessionToken) {
        try {
            const response = await api.processAuthenticationFrame(sessionToken, frameData);
            return response;
        } catch (error) {
            console.error('Authentication frame processing failed:', error);
            throw error;
        }
    }

    // Basic frame validation
    static validateFrame(frameData) {
        if (!frameData || typeof frameData !== 'string') {
            return { valid: false, error: 'Invalid frame data format' };
        }

        if (frameData.length < 1000) {
            return { valid: false, error: 'Frame data too small' };
        }

        // Basic base64 validation
        try {
            atob(frameData);
            return { valid: true };
        } catch (error) {
            return { valid: false, error: 'Invalid base64 encoding' };
        }
    }

    // Frame size estimation
    static estimateFrameSize(frameData) {
        if (!frameData) return 0;
        
        // Base64 encoded size approximation
        const base64Size = frameData.length;
        const actualSize = (base64Size * 3) / 4;
        return Math.round(actualSize);
    }

    // Frame quality metrics
    static analyzeFrameQuality(canvas, context) {
        if (!canvas || !context) {
            return { quality: 0, brightness: 0, contrast: 0 };
        }

        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        let totalBrightness = 0;
        let totalPixels = data.length / 4;
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Calculate brightness using luminance formula
            const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
            totalBrightness += brightness;
        }
        
        const avgBrightness = totalBrightness / totalPixels;
        
        // Simple quality estimation based on brightness
        let quality = 1.0;
        if (avgBrightness < 50 || avgBrightness > 200) {
            quality = Math.max(0.3, 1.0 - Math.abs(avgBrightness - 125) / 125);
        }
        
        return {
            quality: Math.round(quality * 100),
            brightness: Math.round(avgBrightness),
            contrast: Math.round(this.calculateContrast(data)),
            totalPixels: totalPixels
        };
    }

    static calculateContrast(imageData) {
        let min = 255;
        let max = 0;
        
        for (let i = 0; i < imageData.length; i += 4) {
            const r = imageData[i];
            const g = imageData[i + 1];
            const b = imageData[i + 2];
            
            const gray = Math.round(r * 0.299 + g * 0.587 + b * 0.114);
            
            min = Math.min(min, gray);
            max = Math.max(max, gray);
        }
        
        return max - min;
    }
}

// Blink detection utility
class BlinkDetector {
    constructor() {
        this.eyeOpenness = [];
        this.blinkThreshold = 0.5;
        this.blinkCount = 0;
        this.isBlinking = false;
        this.lastBlinkTime = 0;
    }

    // Simulate blink detection (in real implementation, this would use face detection)
    detectBlink() {
        const now = Date.now();
        
        // Simulate random blink detection
        if (now - this.lastBlinkTime > 3000) { // At least 3 seconds between blinks
            const blinkProbability = Math.random();
            
            if (blinkProbability < 0.1 && !this.isBlinking) { // 10% chance of starting a blink
                this.isBlinking = true;
                this.blinkCount++;
                this.lastBlinkTime = now;
                
                // Reset blink after short duration
                setTimeout(() => {
                    this.isBlinking = false;
                }, 200);
                
                return {
                    blink: true,
                    blinkCount: this.blinkCount,
                    confidence: 0.95
                };
            }
        }
        
        return {
            blink: false,
            blinkCount: this.blinkCount,
            confidence: 0.8
        };
    }

    reset() {
        this.eyeOpenness = [];
        this.blinkCount = 0;
        this.isBlinking = false;
        this.lastBlinkTime = 0;
    }

    getBlinkStats() {
        return {
            totalBlinks: this.blinkCount,
            isCurrentlyBlinking: this.isBlinking,
            timeSinceLastBlink: Date.now() - this.lastBlinkTime
        };
    }
}

// Export for global use
window.CameraManager = CameraManager;
window.FrameProcessor = FrameProcessor;
window.BlinkDetector = BlinkDetector;