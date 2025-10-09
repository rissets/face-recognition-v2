import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

const Register: React.FC = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    first_name: '',
    last_name: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { register } = useAuth();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    const registerData = {
      username: formData.username,
      email: formData.email,
      password: formData.password,
      first_name: formData.first_name,
      last_name: formData.last_name,
    };
    const success = await register(registerData);
    
    if (success) {
      navigate('/');
    } else {
      setError('Registration failed. Please try again.');
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

      <div className="relative z-10 w-full max-w-md mx-4">
        <div className="cyber-card p-8 backdrop-blur-xl border-2 border-cyber-blue/30 shadow-glow-lg">
          {/* Logo/Title */}
          <div className="text-center mb-8">
            <h1 className="font-cyber text-3xl font-bold glow-text mb-2">
              NEURAL VISION
            </h1>
            <div className="h-px bg-gradient-to-r from-transparent via-cyber-blue to-transparent mb-4"></div>
            <h2 className="text-xl text-gray-300 font-light tracking-wide">
              CREATE NEW ACCOUNT
            </h2>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 animate-fade-in">
              <p className="text-red-400 text-sm font-medium">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-cyber-blue uppercase tracking-wider">
                  First Name
                </label>
                <input
                  type="text"
                  name="first_name"
                  className="cyber-input w-full"
                  placeholder="First name"
                  value={formData.first_name}
                  onChange={handleChange}
                />
              </div>
              <div className="space-y-2">
                <label className="block text-sm font-medium text-cyber-blue uppercase tracking-wider">
                  Last Name
                </label>
                <input
                  type="text"
                  name="last_name"
                  className="cyber-input w-full"
                  placeholder="Last name"
                  value={formData.last_name}
                  onChange={handleChange}
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-cyber-blue uppercase tracking-wider">
                Username
              </label>
              <input
                type="text"
                name="username"
                required
                className="cyber-input w-full"
                placeholder="Choose a username"
                value={formData.username}
                onChange={handleChange}
                autoFocus
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-cyber-blue uppercase tracking-wider">
                Email
              </label>
              <input
                type="email"
                name="email"
                required
                className="cyber-input w-full"
                placeholder="Enter your email"
                value={formData.email}
                onChange={handleChange}
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-cyber-blue uppercase tracking-wider">
                Password
              </label>
              <input
                type="password"
                name="password"
                required
                className="cyber-input w-full"
                placeholder="Create a password"
                value={formData.password}
                onChange={handleChange}
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-cyber-blue uppercase tracking-wider">
                Confirm Password
              </label>
              <input
                type="password"
                name="confirmPassword"
                required
                className="cyber-input w-full"
                placeholder="Confirm your password"
                value={formData.confirmPassword}
                onChange={handleChange}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="cyber-button w-full py-4 text-lg font-bold tracking-wider uppercase disabled:opacity-50 disabled:cursor-not-allowed mt-6"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-cyber-blue mr-3"></div>
                  CREATING ACCOUNT...
                </div>
              ) : (
                'CREATE ACCOUNT'
              )}
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => navigate('/login')}
                className="text-cyber-purple hover:text-cyber-blue transition-colors duration-200 text-sm font-medium tracking-wide"
              >
                ALREADY HAVE AN ACCOUNT?
              </button>
            </div>
          </form>

          {/* Bottom decoration */}
          <div className="flex justify-center mt-8">
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

export default Register;