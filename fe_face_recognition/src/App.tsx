
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { SocketProvider } from './contexts/SocketContext';
import Layout from './components/Layout/Layout';
import Login from './pages/Auth/Login';
import Register from './pages/Auth/Register';
import Dashboard from './pages/Dashboard/Dashboard';
import FaceEnrollment from './pages/FaceEnrollment/FaceEnrollment';
import RecognitionHistory from './pages/RecognitionHistory/RecognitionHistory';
import Analytics from './pages/Analytics/Analytics';
import LiveStream from './pages/LiveStream/LiveStream';
import ProtectedRoute from './components/ProtectedRoute/ProtectedRoute';

function App() {
  return (
    <div className="min-h-screen bg-dark-bg">
      <AuthProvider>
        <SocketProvider>
          <Router>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/" element={
                <ProtectedRoute>
                  <Layout>
                    <Dashboard />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/face-enrollment" element={
                <ProtectedRoute>
                  <Layout>
                    <FaceEnrollment />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/recognition-history" element={
                <ProtectedRoute>
                  <Layout>
                    <RecognitionHistory />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/analytics" element={
                <ProtectedRoute>
                  <Layout>
                    <Analytics />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/live-stream" element={
                <ProtectedRoute>
                  <Layout>
                    <LiveStream />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Router>
        </SocketProvider>
      </AuthProvider>
    </div>
  );
}

export default App;
