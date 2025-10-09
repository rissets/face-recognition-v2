import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const FaceEnrollment: React.FC = () => {
  const [personName, setPersonName] = useState('');
  const [enrollmentStatus, setEnrollmentStatus] = useState<'idle' | 'capturing' | 'processing' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const webcamRef = useRef<Webcam>(null);

  const captureImage = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setCapturedImages(prev => [...prev, imageSrc]);
        setMessage(`Captured ${capturedImages.length + 1} image(s). Capture at least 3 images for better accuracy.`);
      }
    }
  };

  const clearImages = () => {
    setCapturedImages([]);
    setMessage('');
  };

  const handleEnrollment = async () => {
    if (!personName.trim()) {
      setMessage('Please enter a person name');
      return;
    }

    if (capturedImages.length === 0) {
      setMessage('Please capture at least one image');
      return;
    }

    setEnrollmentStatus('processing');
    setMessage('Processing enrollment...');

    try {
      const formData = new FormData();
      formData.append('name', personName);

      // Convert base64 images to blobs and append to form data
      capturedImages.forEach((imageData, index) => {
        const base64Data = imageData.split(',')[1];
        const blob = base64ToBlob(base64Data, 'image/jpeg');
        formData.append(`image_${index}`, blob, `face_${index}.jpg`);
      });

      const response = await axios.post('/core/enroll-face/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setEnrollmentStatus('success');
        setMessage(`Successfully enrolled ${personName} with ${capturedImages.length} images!`);
        setPersonName('');
        setCapturedImages([]);
      } else {
        setEnrollmentStatus('error');
        setMessage(response.data.message || 'Enrollment failed');
      }
    } catch (error) {
      setEnrollmentStatus('error');
      setMessage('Enrollment failed. Please try again.');
      console.error('Enrollment error:', error);
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

  const resetEnrollment = () => {
    setEnrollmentStatus('idle');
    setMessage('');
    setPersonName('');
    setCapturedImages([]);
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold glow-text mb-2">Face Enrollment</h1>
        <p className="text-gray-400">Capture multiple angles for better accuracy</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Camera Section */}
        <div className="cyber-card">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-cyan-400 mb-4 flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586l-.707-.707A1 1 0 0012.293 4H7.707a1 1 0 00-.707.293L6.293 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
              </svg>
              Camera Feed
            </h2>
            <div className="relative w-full mb-4">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                width="100%"
                className="webcam-frame w-full"
              />
              <div className="absolute top-2 right-2 bg-black/50 text-cyan-400 px-2 py-1 rounded text-sm">
                {capturedImages.length} captured
              </div>
            </div>
            <div className="flex gap-3 flex-wrap">
              <button 
                className="cyber-button flex-1 min-w-[140px]"
                onClick={captureImage}
                disabled={enrollmentStatus === 'processing'}
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Capture ({capturedImages.length})
              </button>
              <button 
                className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={clearImages}
                disabled={capturedImages.length === 0 || enrollmentStatus === 'processing'}
              >
                <svg className="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* Enrollment Form */}
        <div className="cyber-card">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-cyan-400 mb-4 flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
              </svg>
              Enrollment Details
            </h2>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Person Name
              </label>
              <input
                type="text"
                value={personName}
                onChange={(e) => setPersonName(e.target.value)}
                className="cyber-input w-full"
                placeholder="Enter person's name..."
                disabled={enrollmentStatus === 'processing'}
              />
            </div>

            {message && (
              <div className={`p-4 rounded-lg mb-4 ${
                enrollmentStatus === 'success' 
                  ? 'bg-green-900/50 border border-green-500/50 text-green-300' 
                  : enrollmentStatus === 'error' 
                  ? 'bg-red-900/50 border border-red-500/50 text-red-300'
                  : 'bg-blue-900/50 border border-cyan-500/50 text-cyan-300'
              }`}>
                <div className="flex items-center">
                  {enrollmentStatus === 'success' ? (
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  ) : enrollmentStatus === 'error' ? (
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                  )}
                  {message}
                </div>
              </div>
            )}

            <div className="flex gap-3 flex-wrap">
              <button
                className="cyber-button flex-1 min-w-[140px] flex items-center justify-center"
                onClick={handleEnrollment}
                disabled={enrollmentStatus === 'processing' || !personName.trim() || capturedImages.length === 0}
              >
                {enrollmentStatus === 'processing' ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                    </svg>
                    Enroll Face
                  </>
                )}
              </button>
              
              {(enrollmentStatus === 'success' || enrollmentStatus === 'error') && (
                <button
                  className="bg-gray-600 hover:bg-gray-700 border border-gray-500 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300"
                  onClick={resetEnrollment}
                >
                  Enroll Another
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Captured Images Preview */}
      {capturedImages.length > 0 && (
        <div className="cyber-card">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-cyan-400 mb-4 flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
              </svg>
              Captured Images ({capturedImages.length})
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {capturedImages.map((image, index) => (
                <div key={index} className="relative group">
                  <img
                    src={image}
                    alt={`Captured ${index + 1}`}
                    className="w-full h-20 object-cover rounded-lg border border-cyan-400/30 group-hover:border-cyan-400 transition-all duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="absolute bottom-1 left-1 text-xs text-white font-semibold">
                      #{index + 1}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FaceEnrollment;