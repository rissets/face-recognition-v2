import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const success = await login(username, password);
    
    if (success) {
      navigate('/');
    } else {
      setError('Invalid username or password');
    }
    
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-dark-bg via-blue-900/20 to-purple-900/20">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 25% 25%, rgba(0, 245, 255, 0.1) 0%, transparent 50%),
                           radial-gradient(circle at 75% 75%, rgba(124, 58, 237, 0.1) 0%, transparent 50%)`
        }}></div>
      </div>
      
      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(50)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyber-blue/30 rounded-full animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 3}s`
            }}
          ></div>
        ))}
      </div>

      <div className="relative z-10 w-full max-w-md mx-4 sm:mx-6">
        <div className="cyber-card p-6 sm:p-8 backdrop-blur-xl border-2 border-cyber-blue/30 shadow-glow-lg">
          {/* Logo/Title */}
          <div className="text-center mb-6 sm:mb-8">
            <h1 className="font-cyber text-2xl sm:text-3xl font-bold glow-text mb-2">
              NEURAL VISION
            </h1>
            <div className="h-px bg-gradient-to-r from-transparent via-cyber-blue to-transparent mb-3 sm:mb-4"></div>
            <h2 className="text-lg sm:text-xl text-gray-300 font-light tracking-wide">
              AUTHENTICATION PROTOCOL
            </h2>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 animate-fade-in">
              <p className="text-red-400 text-sm font-medium">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
            <div className="space-y-2">
              <label className="block text-xs sm:text-sm font-medium text-cyber-blue uppercase tracking-wider">
                USERNAME
              </label>
              <input
                type="text"
                required
                className="cyber-input w-full text-sm sm:text-base"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
              />
            </div>

            <div className="space-y-2">
              <label className="block text-xs sm:text-sm font-medium text-cyber-blue uppercase tracking-wider">
                PASSWORD
              </label>
              <input
                type="password"
                required
                className="cyber-input w-full text-sm sm:text-base"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="cyber-button w-full py-3 sm:py-4 text-base sm:text-lg font-bold tracking-wider uppercase disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-cyber-blue mr-3"></div>
                  AUTHENTICATING...
                </div>
              ) : (
                'ACCESS SYSTEM'
              )}
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => navigate('/register')}
                className="text-cyber-purple hover:text-cyber-blue transition-colors duration-200 text-xs sm:text-sm font-medium tracking-wide"
              >
                CREATE NEW ACCOUNT
              </button>
            </div>
          </form>

          {/* Bottom decoration */}
          <div className="flex justify-center mt-6 sm:mt-8">
            <div className="flex space-x-1">
              {[...Array(3)].map((_, i) => (
                <div
                  key={i}
                  className="w-2 h-2 bg-cyber-blue rounded-full animate-pulse"
                  style={{ animationDelay: `${i * 0.2}s` }}
                ></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;