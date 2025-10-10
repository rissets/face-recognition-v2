import React, { useState, useRef, useEffect, useContext } from 'react';
import Webcam from 'react-webcam';
import { SocketContext } from '../../contexts/SocketContext';

interface DetectedFace {
  person_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  timestamp: string;
}

const LiveStream: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [error, setError] = useState('');
  const [recordRecognitions, setRecordRecognitions] = useState(true);
  const webcamRef = useRef<Webcam>(null);
  const intervalRef = useRef<number | null>(null);
  const socketContext = useContext(SocketContext);

  useEffect(() => {
    if (socketContext?.socket) {
      socketContext.socket.on('face_detected', handleFaceDetection);
      
      return () => {
        socketContext.socket?.off('face_detected', handleFaceDetection);
      };
    }
  }, [socketContext]);

  const handleFaceDetection = (data: DetectedFace) => {
    setDetectedFaces(prev => [data, ...prev.slice(0, 9)]); // Keep last 10 detections
  };

  const startStreaming = async () => {
    try {
      setError('');
      setIsStreaming(true);
      
      // Start capturing frames and sending them for recognition
      intervalRef.current = setInterval(() => {
        captureAndAnalyzeFrame();
      }, 1000); // Analyze frame every second
      
    } catch (err) {
      setError('Failed to start streaming');
      setIsStreaming(false);
      console.error('Streaming error:', err);
    }
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setDetectedFaces([]);
  };

  const captureAndAnalyzeFrame = async () => {
    if (webcamRef.current && isStreaming) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        try {
          // Convert base64 to blob
          const base64Data = imageSrc.split(',')[1];
          const blob = base64ToBlob(base64Data, 'image/jpeg');
          
          const formData = new FormData();
          formData.append('image', blob, 'frame.jpg');
          formData.append('record', recordRecognitions.toString());
          
          // Send to recognition endpoint
          const response = await fetch('http://localhost:8000/api/recognition/live-recognize/', {
            method: 'POST',
            headers: {
              'Authorization': `Token ${localStorage.getItem('token')}`,
            },
            body: formData,
          });
          
          if (response.ok) {
            const data = await response.json();
            if (data.faces && data.faces.length > 0) {
              data.faces.forEach((face: DetectedFace) => {
                handleFaceDetection({
                  ...face,
                  timestamp: new Date().toISOString(),
                });
              });
            }
          }
        } catch (err) {
          console.error('Frame analysis error:', err);
        }
      }
    }
  };

  const base64ToBlob = (base64: string, mimeType: string) => {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-900/50 text-green-300 border-green-500/50';
    if (confidence >= 0.6) return 'bg-yellow-900/50 text-yellow-300 border-yellow-500/50';
    return 'bg-red-900/50 text-red-300 border-red-500/50';
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold glow-text mb-2">Live Face Recognition</h1>
        <p className="text-gray-400">Real-time face detection and recognition</p>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500/50 text-red-300 p-4 rounded-lg mb-6">
          <div className="flex items-center">
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            {error}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Camera Feed */}
        <div className="lg:col-span-2">
          <div className="cyber-card">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-cyan-400 flex items-center">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Camera Feed
                </h2>
                <div className="flex gap-4 items-center">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={recordRecognitions}
                      onChange={(e) => setRecordRecognitions(e.target.checked)}
                      disabled={isStreaming}
                      className="sr-only"
                    />
                    <div className={`relative w-10 h-6 transition duration-200 ease-linear rounded-full ${recordRecognitions ? 'bg-cyan-400' : 'bg-gray-600'} ${isStreaming ? 'opacity-50' : ''}`}>
                      <label className={`absolute left-0 bg-white border-2 mb-2 w-6 h-6 rounded-full transition transform duration-100 ease-linear cursor-pointer ${recordRecognitions ? 'translate-x-full border-cyan-400' : 'translate-x-0 border-gray-300'}`}></label>
                    </div>
                    <span className="ml-3 text-gray-300 text-sm">Record Recognitions</span>
                  </label>
                  <button
                    className={`font-semibold py-2 px-6 rounded-lg transition-all duration-300 flex items-center ${
                      isStreaming 
                        ? 'bg-red-600 hover:bg-red-700 text-white border border-red-500' 
                        : 'cyber-button'
                    }`}
                    onClick={isStreaming ? stopStreaming : startStreaming}
                  >
                    {isStreaming ? (
                      <>
                        <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                        </svg>
                        Stop Stream
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                        </svg>
                        Start Stream
                      </>
                    )}
                  </button>
                </div>
              </div>
              
              <div className="relative">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  width="100%"
                  className={`w-full rounded-lg ${isStreaming ? 'webcam-frame border-green-400' : 'border-2 border-gray-600'}`}
                />
                
                {isStreaming && (
                  <div className="absolute top-4 left-4 bg-green-600/90 text-white px-3 py-1 rounded-full text-sm font-semibold flex items-center">
                    <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></div>
                    LIVE
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Detection Results */}
        <div className="lg:col-span-1">
          <div className="cyber-card">
            <div className="p-6">
              <h2 className="text-xl font-semibold text-cyan-400 mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                </svg>
                Recent Detections
              </h2>
              
              {detectedFaces.length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-gray-500 mb-4">
                    <svg className="mx-auto h-12 w-12 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </div>
                  <p className="text-gray-400 text-sm">
                    {isStreaming ? 'No faces detected yet' : 'Start streaming to see detections'}
                  </p>
                </div>
              ) : (
                <div className="max-h-96 overflow-y-auto space-y-3">
                  {detectedFaces.map((face, index) => (
                    <div
                      key={index}
                      className={`detection-item ${index === 0 ? 'border-cyan-400/50' : ''}`}
                    >
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-semibold text-white">
                          {face.person_name}
                        </h3>
                        <span className="text-xs text-gray-400">
                          {formatTimestamp(face.timestamp)}
                        </span>
                      </div>
                      
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getConfidenceColor(face.confidence)}`}>
                        {(face.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="cyber-card mt-6">
        <div className="p-6">
          <h2 className="text-xl font-semibold text-cyan-400 mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Instructions
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-cyan-400/20 text-cyan-400 rounded-full flex items-center justify-center text-sm font-semibold">
                1
              </div>
              <p className="text-gray-300 text-sm">
                Click "Start Stream" to begin live face recognition
              </p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-cyan-400/20 text-cyan-400 rounded-full flex items-center justify-center text-sm font-semibold">
                2
              </div>
              <p className="text-gray-300 text-sm">
                Position faces clearly in front of the camera
              </p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-cyan-400/20 text-cyan-400 rounded-full flex items-center justify-center text-sm font-semibold">
                3
              </div>
              <p className="text-gray-300 text-sm">
                Toggle "Record Recognitions" to save detection results to history
              </p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-6 h-6 bg-cyan-400/20 text-cyan-400 rounded-full flex items-center justify-center text-sm font-semibold">
                4
              </div>
              <p className="text-gray-300 text-sm">
                Recent detections will appear in the panel on the right
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveStream;