# 🤖 Face Recognition Frontend - Futuristic React App

Aplikasi frontend yang futuristik untuk sistem face recognition dengan design cyberpunk dan teknologi modern.

## ✨ Features

### 🎯 **Core Features**
- 🔐 **Authentication System** - Login & Register dengan JWT
- 📊 **Dashboard** - Overview dan statistik real-time
- 👤 **Face Enrollment** - Pendaftaran wajah dengan webcam
- 📈 **Recognition History** - Riwayat pengenalan wajah
- 📊 **Analytics** - Data visualization dan insights
- 🎥 **Live Stream** - Real-time face recognition

### 🎨 **Futuristic UI Design**
- 🌈 **Cyberpunk Theme** - Design futuristik dengan warna neon
- ✨ **Glow Effects** - Efek cahaya dan shadow yang menarik
- 🔮 **Glass Morphism** - Efek transparan dan blur modern
- 🎭 **Advanced Animations** - Glitch, hologram, dan pulse effects
- 📱 **Responsive Design** - Layout adaptif untuk semua device

## 🚀 Tech Stack

### **Frontend Framework**
- ⚛️ **React 18** - Modern React dengan TypeScript
- 🏗️ **Vite** - Fast build tool dan development server
- 📘 **TypeScript** - Type safety dan better DX

### **Styling & UI**
- 🎨 **Tailwind CSS v4** - Utility-first CSS framework
- 🔤 **Orbitron Font** - Futuristic typography
- ✨ **Custom CSS Animations** - Keyframe animations dan effects

### **State Management & API**
- 🔄 **React Context** - Authentication dan global state
- 📡 **Axios** - HTTP client dengan JWT authentication
- 🔌 **Socket.IO** - Real-time communication

### **Media & Charts**
- 📹 **React Webcam** - Camera integration
- 📊 **Recharts** - Data visualization
- 🎯 **React Router** - Navigation dan routing

## 🛠️ Installation & Setup

### **Prerequisites**

```bash
# Node.js (recommended v18+)
node --version

# npm atau yarn
npm --version
```

### **Install Dependencies**

```bash
# Clone dan masuk ke direktori
cd fe_face_recognition

# Install dependencies
npm install
```

### **Development Server**

```bash
# Start development server
npm run dev

# Server akan berjalan di http://localhost:5173
```

### **Build for Production**

```bash
# Build aplikasi
npm run build

# Preview build
npm run preview
```

## 🚦 Development Scripts

```bash
# Development
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build

# Code Quality
npm run lint         # Run ESLint
npm run type-check   # TypeScript checking
```

## 📁 Project Structure

```bash
src/
├── components/          # Reusable components
│   ├── Layout/         # Main layout dengan sidebar
│   └── ProtectedRoute/ # Route protection
├── contexts/           # React contexts
│   ├── AuthContext.tsx # Authentication state
│   └── SocketContext.tsx # WebSocket connection
├── hooks/              # Custom hooks
│   └── useAuth.ts      # Authentication hook
├── pages/              # Page components
│   ├── Auth/           # Login & Register
│   ├── Dashboard/      # Main dashboard
│   ├── FaceEnrollment/ # Face registration
│   ├── RecognitionHistory/ # History view
│   ├── Analytics/      # Data visualization
│   └── LiveStream/     # Real-time recognition
└── assets/             # Static assets
```

## 🎨 Styling Guide

### **Color Palette**

- 🔵 **Primary**: Cyan-400 (#00f5ff) - Neon cyan
- 🟣 **Secondary**: Purple-600 (#7c3aed) - Futuristic purple
- ⚫ **Background**: Gray-900 (#111827) - Dark background
- 🌈 **Gradients**: Linear combinations of cyan dan purple

### **Custom Components**

```css
/* Cyber Cards */
.cyber-card {
  background: linear-gradient(145deg, rgba(42, 42, 42, 0.8), rgba(26, 26, 26, 0.9));
  box-shadow: 0 0 10px rgba(124, 58, 237, 0.3);
}

/* Neon Buttons */
.cyber-button {
  background: linear-gradient(to right, #00f5ff, #7c3aed);
  box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
}

/* Glow Text */
.glow-text {
  text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
}
```

### **Animations**

- ⚡ **Glitch Effect** - Text dan element animations
- 🔮 **Hologram** - Opacity dan scale transitions
- 💫 **Pulse Glow** - Shadow pulsing effects
- 📈 **Slide Up** - Entry animations

## 🌐 Backend Integration

### **API Endpoints**

```javascript
// Base URL
const API_BASE_URL = 'http://localhost:8000/api'

// Authentication
POST /auth/login
POST /auth/register
POST /auth/refresh

// Face Recognition
POST /face/enroll
POST /face/recognize
GET /face/history
GET /face/analytics
```

### **WebSocket Connection**

```javascript
// Socket.IO untuk real-time features
const socket = io('http://localhost:8000')

socket.on('recognition_result', (data) => {
  // Handle real-time recognition results
})
```

## 🎯 Usage

### **1. Authentication**

- Navigate ke `/login` untuk masuk
- Daftar akun baru di `/register`
- JWT token disimpan di localStorage

### **2. Face Enrollment**

- Kunjungi `/face-enrollment`
- Allow camera permissions
- Capture multiple face angles
- Submit untuk training

### **3. Live Recognition**

- Buka `/live-stream`
- Camera akan mendeteksi wajah real-time
- Results ditampilkan dengan confidence score

### **4. Analytics**

- Dashboard menampilkan statistik
- Charts untuk recognition trends
- History table dengan filter options

## 🔐 Environment Variables

```bash
# .env.local
VITE_API_BASE_URL=http://localhost:8000/api
VITE_SOCKET_URL=http://localhost:8000
VITE_APP_NAME="Face Recognition System"
```

## 📚 Documentation

- 🎨 **[Futuristic Styling Guide](./FUTURISTIC_STYLING.md)** - Detail styling dan animations
- 🔧 **[Component Documentation](./docs/components.md)** - Component API reference
- 🌐 **[API Integration](./docs/api.md)** - Backend integration guide

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 Acknowledgments

- 🎨 **Design Inspiration**: Cyberpunk 2077, Tron Legacy
- 🔧 **Tools**: Vite, Tailwind CSS, React
- 👥 **Community**: React, TypeScript, dan Open Source community

---

**🚀 Built with modern web technologies and futuristic design principles**
