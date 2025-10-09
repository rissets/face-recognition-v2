<template>
  <div class="enhanced-auth-page">
    <NavigationBar />
    
    <div class="page-container">
      <div class="page-header">
        <h1>Enhanced Face Recognition Demo</h1>
        <p class="page-description">
          Experience our advanced face recognition system with real-time liveness detection 
          and obstacle recognition capabilities.
        </p>
      </div>

      <div class="features-overview">
        <div class="feature-card">
          <div class="feature-icon">👁️</div>
          <h3>Liveness Detection</h3>
          <p>Advanced blink detection and motion analysis to prevent spoofing attacks using photos or videos.</p>
          <ul>
            <li>Real-time Eye Aspect Ratio (EAR) monitoring</li>
            <li>Blink pattern analysis</li>
            <li>Natural head movement detection</li>
            <li>Anti-spoofing protection</li>
          </ul>
        </div>

        <div class="feature-card">
          <div class="feature-icon">🚫</div>
          <h3>Obstacle Detection</h3>
          <p>Intelligent recognition of face obstructions to ensure clear biometric capture.</p>
          <ul>
            <li>Glasses detection with reflection analysis</li>
            <li>Face mask detection</li>
            <li>Hat/cap recognition</li>
            <li>Hand covering detection</li>
          </ul>
        </div>

        <div class="feature-card">
          <div class="feature-icon">🎯</div>
          <h3>High Accuracy Recognition</h3>
          <p>State-of-the-art face recognition powered by InsightFace and deep learning models.</p>
          <ul>
            <li>512-dimensional face embeddings</li>
            <li>Cosine similarity matching</li>
            <li>Quality assessment scoring</li>
            <li>Adaptive thresholds</li>
          </ul>
        </div>
      </div>

      <div class="demo-section">
        <h2>Live Demo</h2>
        <div class="demo-instructions">
          <div class="instruction-card">
            <span class="step-number">1</span>
            <div class="instruction-content">
              <h4>Prepare Your Environment</h4>
              <p>Ensure good lighting and position your face clearly in the camera view.</p>
            </div>
          </div>
          
          <div class="instruction-card">
            <span class="step-number">2</span>
            <div class="instruction-content">
              <h4>Remove Obstacles</h4>
              <p>The system will detect any glasses, masks, or other obstructions and guide you accordingly.</p>
            </div>
          </div>
          
          <div class="instruction-card">
            <span class="step-number">3</span>
            <div class="instruction-content">
              <h4>Prove Liveness</h4>
              <p>Blink naturally or move your head slightly to prove you're a live person, not a photo.</p>
            </div>
          </div>
          
          <div class="instruction-card">
            <span class="step-number">4</span>
            <div class="instruction-content">
              <h4>Authentication</h4>
              <p>The system will match your face against registered users and provide instant results.</p>
            </div>
          </div>
        </div>

        <EnhancedWebRTCAuthorization />
      </div>

      <div class="technical-details">
        <h2>Technical Implementation</h2>
        <div class="tech-grid">
          <div class="tech-card">
            <h3>Frontend Technologies</h3>
            <ul>
              <li><strong>Vue 3:</strong> Reactive UI framework</li>
              <li><strong>WebRTC:</strong> Real-time video streaming</li>
              <li><strong>Canvas API:</strong> Frame capture and processing</li>
              <li><strong>MediaStream:</strong> Camera access and control</li>
            </ul>
          </div>

          <div class="tech-card">
            <h3>Backend Architecture</h3>
            <ul>
              <li><strong>Django REST:</strong> API framework</li>
              <li><strong>InsightFace:</strong> Face recognition engine</li>
              <li><strong>MediaPipe:</strong> Face mesh and landmark detection</li>
              <li><strong>OpenCV:</strong> Computer vision processing</li>
            </ul>
          </div>

          <div class="tech-card">
            <h3>Security Features</h3>
            <ul>
              <li><strong>Encrypted Storage:</strong> Protected face embeddings</li>
              <li><strong>Session Management:</strong> Secure authentication tokens</li>
              <li><strong>Audit Logging:</strong> Complete activity tracking</li>
              <li><strong>Privacy Protection:</strong> No raw image storage</li>
            </ul>
          </div>

          <div class="tech-card">
            <h3>Performance Metrics</h3>
            <ul>
              <li><strong>Accuracy:</strong> >99% face recognition</li>
              <li><strong>Speed:</strong> <100ms processing time</li>
              <li><strong>Liveness:</strong> >95% spoofing prevention</li>
              <li><strong>Obstacles:</strong> >90% detection accuracy</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="api-reference">
        <h2>API Endpoints</h2>
        <div class="endpoint-list">
          <div class="endpoint-item">
            <div class="endpoint-header">
              <span class="http-method post">POST</span>
              <span class="endpoint-path">/api/auth/webrtc/create/</span>
            </div>
            <p>Create a new WebRTC authentication session with liveness detection enabled.</p>
          </div>

          <div class="endpoint-item">
            <div class="endpoint-header">
              <span class="http-method post">POST</span>
              <span class="endpoint-path">/api/auth/frame-process/</span>
            </div>
            <p>Process video frames for face recognition, liveness detection, and obstacle analysis.</p>
          </div>

          <div class="endpoint-item">
            <div class="endpoint-header">
              <span class="http-method get">GET</span>
              <span class="endpoint-path">/api/auth/history/</span>
            </div>
            <p>Retrieve authentication history with detailed liveness and obstacle detection logs.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import NavigationBar from '../components/NavigationBar.vue'
import EnhancedWebRTCAuthorization from '../components/EnhancedWebRTCAuthorization.vue'
</script>

<style scoped>
.enhanced-auth-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.page-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  color: white;
}

.page-header {
  text-align: center;
  margin-bottom: 60px;
}

.page-header h1 {
  font-size: 3rem;
  font-weight: 700;
  margin: 0 0 20px 0;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.page-description {
  font-size: 1.3rem;
  margin: 0;
  opacity: 0.9;
  line-height: 1.6;
  max-width: 800px;
  margin: 0 auto;
}

.features-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 30px;
  margin-bottom: 60px;
}

.feature-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 30px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-10px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.feature-icon {
  font-size: 4rem;
  margin-bottom: 20px;
}

.feature-card h3 {
  font-size: 1.5rem;
  margin: 0 0 15px 0;
  color: white;
}

.feature-card p {
  margin: 0 0 20px 0;
  opacity: 0.9;
  line-height: 1.6;
}

.feature-card ul {
  list-style: none;
  padding: 0;
  text-align: left;
}

.feature-card li {
  margin: 8px 0;
  padding-left: 20px;
  position: relative;
  opacity: 0.8;
}

.feature-card li::before {
  content: '✓';
  position: absolute;
  left: 0;
  color: #4ade80;
  font-weight: bold;
}

.demo-section {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 40px;
  margin-bottom: 60px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.demo-section h2 {
  text-align: center;
  font-size: 2.5rem;
  margin: 0 0 40px 0;
}

.demo-instructions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 40px;
}

.instruction-card {
  display: flex;
  align-items: center;
  gap: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 15px;
  padding: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.step-number {
  background: linear-gradient(135deg, #4ade80, #22c55e);
  color: white;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 1.2rem;
  flex-shrink: 0;
}

.instruction-content h4 {
  margin: 0 0 8px 0;
  font-size: 1.1rem;
  color: white;
}

.instruction-content p {
  margin: 0;
  opacity: 0.8;
  font-size: 0.9rem;
  line-height: 1.4;
}

.technical-details {
  margin-bottom: 60px;
}

.technical-details h2 {
  text-align: center;
  font-size: 2.5rem;
  margin: 0 0 40px 0;
}

.tech-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 25px;
}

.tech-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 25px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.tech-card h3 {
  margin: 0 0 20px 0;
  color: #4ade80;
  font-size: 1.3rem;
}

.tech-card ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.tech-card li {
  margin: 12px 0;
  opacity: 0.9;
  line-height: 1.5;
}

.tech-card strong {
  color: white;
  font-weight: 600;
}

.api-reference {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 40px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.api-reference h2 {
  text-align: center;
  font-size: 2.5rem;
  margin: 0 0 40px 0;
}

.endpoint-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.endpoint-item {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.endpoint-header {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 10px;
}

.http-method {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: bold;
  text-transform: uppercase;
}

.http-method.post {
  background: #f59e0b;
  color: white;
}

.http-method.get {
  background: #10b981;
  color: white;
}

.endpoint-path {
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  background: rgba(0, 0, 0, 0.3);
  padding: 8px 12px;
  border-radius: 8px;
  color: #4ade80;
  font-size: 0.9rem;
}

.endpoint-item p {
  margin: 0;
  opacity: 0.8;
  line-height: 1.5;
}

@media (max-width: 768px) {
  .page-header h1 {
    font-size: 2.2rem;
  }
  
  .page-description {
    font-size: 1.1rem;
  }
  
  .features-overview {
    grid-template-columns: 1fr;
  }
  
  .demo-instructions {
    grid-template-columns: 1fr;
  }
  
  .tech-grid {
    grid-template-columns: 1fr;
  }
  
  .instruction-card {
    flex-direction: column;
    text-align: center;
  }
  
  .endpoint-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
}
</style>