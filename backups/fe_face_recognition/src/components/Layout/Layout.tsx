import React, { useState } from 'react';
import type { ReactNode } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false); // Default closed on mobile
  const navigate = useNavigate();
  const location = useLocation();
  const { logout, user } = useAuth();

  const menuItems = [
    { text: 'Dashboard', icon: '📊', path: '/', gradient: 'from-cyber-blue to-cyber-purple' },
    { text: 'Face Enrollment', icon: '👤', path: '/face-enrollment', gradient: 'from-cyber-green to-cyber-blue' },
    { text: 'Recognition History', icon: '📋', path: '/recognition-history', gradient: 'from-cyber-purple to-cyber-pink' },
    { text: 'Analytics', icon: '📈', path: '/analytics', gradient: 'from-cyber-orange to-cyber-pink' },
    { text: 'Live Stream', icon: '📹', path: '/live-stream', gradient: 'from-cyber-pink to-cyber-purple' },
  ];

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-dark-bg flex relative">
      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        ${sidebarOpen ? 'w-72' : 'lg:w-20'}
        fixed lg:relative z-50 lg:z-auto
        transition-all duration-300 cyber-nav flex flex-col
        h-full lg:h-auto
      `}>
        {/* Logo/Brand */}
        <div className="p-6 border-b border-cyber-blue/20">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-cyber-blue to-cyber-purple rounded-lg flex items-center justify-center shadow-glow">
              <span className="text-xl">🧠</span>
            </div>
            {sidebarOpen && (
              <div>
                <h1 className="font-cyber text-lg font-bold glow-text">NEURAL</h1>
                <p className="text-xs text-gray-400 tracking-wider">VISION SYSTEM</p>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            {menuItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <button
                  key={item.text}
                  onClick={() => handleNavigation(item.path)}
                  className={`
                    w-full flex items-center space-x-3 p-4 rounded-lg transition-all duration-200
                    ${isActive 
                      ? 'cyber-nav-item active' 
                      : 'cyber-nav-item'
                    }
                    group relative overflow-hidden
                  `}
                >
                  <div className={`
                    w-8 h-8 rounded-lg flex items-center justify-center text-lg
                    bg-gradient-to-r ${item.gradient} ${isActive ? 'shadow-glow' : ''}
                  `}>
                    {item.icon}
                  </div>
                  {sidebarOpen && (
                    <span className="font-medium tracking-wide">{item.text}</span>
                  )}
                  {isActive && (
                    <div className="absolute right-0 top-0 bottom-0 w-1 bg-gradient-to-b from-cyber-blue to-cyber-purple rounded-l-full"></div>
                  )}
                </button>
              );
            })}
          </div>
        </nav>

        {/* Sidebar Toggle */}
        <div className="p-4 border-t border-cyber-blue/20">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="w-full p-3 rounded-lg cyber-nav-item text-center"
          >
            <span className="text-lg">{sidebarOpen ? '←' : '→'}</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="cyber-header px-4 lg:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {/* Mobile Menu Button */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 rounded-lg cyber-nav-item"
            >
              <span className="text-lg">☰</span>
            </button>
            
            <div>
              <h2 className="text-lg lg:text-xl font-semibold text-white truncate">Welcome back, {user?.first_name || user?.username}</h2>
              <p className="text-xs lg:text-sm text-gray-400 hidden sm:block">Neural Vision Control Center</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2 lg:space-x-4">
            {/* Status indicator - hidden on small screens */}
            <div className="hidden md:flex items-center space-x-2">
              <div className="w-2 h-2 bg-cyber-green rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-400">System Online</span>
            </div>
            
            {/* User menu */}
            <div className="flex items-center space-x-2 lg:space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-cyber-blue to-cyber-purple rounded-full flex items-center justify-center shadow-glow">
                <span className="text-sm font-bold">{(user?.first_name?.[0] || user?.username?.[0] || 'U').toUpperCase()}</span>
              </div>
              <button
                onClick={handleLogout}
                className="px-2 py-1 lg:px-4 lg:py-2 bg-red-500/20 text-red-400 rounded-lg border border-red-500/30 hover:bg-red-500/30 transition-colors duration-200 text-xs lg:text-sm font-medium"
              >
                <span className="hidden sm:inline">Logout</span>
                <span className="sm:hidden">⏻</span>
              </button>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 p-4 lg:p-6 overflow-auto">
          <div className="animate-fade-in max-w-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;