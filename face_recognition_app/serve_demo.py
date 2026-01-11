#!/usr/bin/env python3
"""
Simple HTTP server untuk serve HTML demo tanpa auto-reload
Lebih stabil daripada VS Code Live Server
"""
import http.server
import socketserver
import os
import sys

PORT = 8888

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler dengan CORS dan no-cache headers"""
    
    def end_headers(self):
        # CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        
        # Disable caching
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        # Custom log format
        print(f"[{self.log_date_time_string()}] {format % args}")

def main():
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    Handler = NoCacheHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("=" * 60)
            print("🚀 OIDC + Face Recognition Demo Server")
            print("=" * 60)
            print(f"📁 Serving files from: {os.getcwd()}")
            print(f"🌐 Server URL: http://localhost:{PORT}")
            print(f"📄 Demo Page: http://localhost:{PORT}/oidc_face_demo.html")
            print("=" * 60)
            print("Press Ctrl+C to stop server")
            print()
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {PORT} already in use")
            print(f"   Try: lsof -ti:{PORT} | xargs kill -9")
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
