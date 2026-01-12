import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

interface DashboardStats {
  total_users: number;
  total_recognitions: number;
  enrolled_faces: number;
  recent_recognitions: unknown[];
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchDashboardStats();
  }, []);

  const fetchDashboardStats = async () => {
    try {
      const response = await axios.get('/analytics/dashboard-stats/');
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const dashboardCards = [
    {
      title: 'Face Enrollment',
      description: 'Enroll new faces for recognition',
      icon: '👤',
      action: () => navigate('/face-enrollment'),
      gradient: 'from-cyber-green to-cyber-blue',
    },
    {
      title: 'Recognition History',
      description: 'View past recognition results',
      icon: '📋',
      action: () => navigate('/recognition-history'),
      gradient: 'from-cyber-purple to-cyber-pink',
    },
    {
      title: 'Analytics',
      description: 'Analyze recognition patterns',
      icon: '�',
      action: () => navigate('/analytics'),
      gradient: 'from-cyber-orange to-cyber-pink',
    },
    {
      title: 'Live Stream',
      description: 'Real-time face recognition',
      icon: '📹',
      action: () => navigate('/live-stream'),
      gradient: 'from-cyber-pink to-cyber-purple',
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-cyber-blue/30 rounded-full animate-spin"></div>
          <div className="absolute top-0 left-0 w-16 h-16 border-4 border-transparent border-t-cyber-blue rounded-full animate-spin"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="font-cyber text-2xl sm:text-3xl lg:text-4xl font-bold glow-text mb-4">CONTROL CENTER</h1>
        <div className="h-px bg-gradient-to-r from-transparent via-cyber-blue to-transparent"></div>
      </div>
      
      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
          <div className="stats-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs sm:text-sm uppercase tracking-wider mb-2">Total Users</p>
                <p className="text-2xl sm:text-3xl font-bold text-cyber-blue">{stats.total_users}</p>
              </div>
              <div className="text-3xl sm:text-4xl opacity-20">👤</div>
            </div>
          </div>
          
          <div className="stats-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs sm:text-sm uppercase tracking-wider mb-2">Enrolled Faces</p>
                <p className="text-2xl sm:text-3xl font-bold text-cyber-green">{stats.enrolled_faces}</p>
              </div>
              <div className="text-3xl sm:text-4xl opacity-20">🧠</div>
            </div>
          </div>
          
          <div className="stats-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs sm:text-sm uppercase tracking-wider mb-2">Total Recognitions</p>
                <p className="text-2xl sm:text-3xl font-bold text-cyber-purple">{stats.total_recognitions}</p>
              </div>
              <div className="text-3xl sm:text-4xl opacity-20">📊</div>
            </div>
          </div>
          
          <div className="stats-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs sm:text-sm uppercase tracking-wider mb-2">Recent Activity</p>
                <p className="text-2xl sm:text-3xl font-bold text-cyber-orange">{stats.recent_recognitions.length}</p>
              </div>
              <div className="text-3xl sm:text-4xl opacity-20">⚡</div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-white mb-4 lg:mb-6 text-center">
          <span className="bg-gradient-to-r from-cyber-blue to-cyber-purple bg-clip-text text-transparent">
            QUICK ACCESS MODULES
          </span>
        </h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
          {dashboardCards.map((card, index) => (
            <div key={index} className="feature-card group cursor-pointer" onClick={card.action}>
              <div className="text-center p-2">
                <div className={`w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-3 sm:mb-4 rounded-full bg-gradient-to-r ${card.gradient} flex items-center justify-center text-xl sm:text-2xl shadow-glow group-hover:shadow-glow-lg transition-all duration-300`}>
                  {card.icon}
                </div>
                <h3 className="text-base sm:text-lg font-bold text-white mb-2 group-hover:text-cyber-blue transition-colors duration-200">
                  {card.title}
                </h3>
                <p className="text-gray-400 text-xs sm:text-sm mb-3 sm:mb-4">{card.description}</p>
                <div className="w-full h-px bg-gradient-to-r from-transparent via-cyber-blue/30 to-transparent group-hover:via-cyber-blue transition-all duration-300"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* System Status */}
      <div className="cyber-card p-4 lg:p-6">
        <h3 className="text-lg sm:text-xl font-bold text-cyber-blue mb-4">SYSTEM STATUS</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-cyber-green rounded-full animate-pulse"></div>
            <span className="text-gray-300 text-sm sm:text-base">Neural Network: Online</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-cyber-blue rounded-full animate-pulse"></div>
            <span className="text-gray-300 text-sm sm:text-base">Vision Modules: Active</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-cyber-purple rounded-full animate-pulse"></div>
            <span className="text-gray-300 text-sm sm:text-base">Database: Connected</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;