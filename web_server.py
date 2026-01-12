#!/usr/bin/env python3
"""
Simple HTTP server for Face Recognition WebSocket Client
Serves the web interface and acts as a proxy for WebSocket connections
"""

import os
import sys
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import threading
import json

class FaceAuthHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for face auth web client"""
    
    def end_headers(self):
        """Add CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def do_GET(self):
        """Handle GET requests"""
        # Serve main HTML file for root path
        if self.path == '/' or self.path == '':
            self.path = '/web_face_auth.html'
        
        # Serve config file
        elif self.path == '/api/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Load config file
            config_path = Path(__file__).parent / 'web_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Remove sensitive data for security
                    safe_config = {
                        'default_profile': config.get('default_profile'),
                        'profiles': {}
                    }
                    for profile_name, profile_data in config.get('profiles', {}).items():
                        safe_config['profiles'][profile_name] = {
                            'name': profile_data.get('name'),
                            'base_url': profile_data.get('base_url'),
                            'description': profile_data.get('description'),
                            'has_credentials': bool(profile_data.get('api_key') and profile_data.get('secret_key'))
                        }
                    safe_config['settings'] = config.get('settings', {})
                    self.wfile.write(json.dumps(safe_config, indent=2).encode())
            else:
                self.wfile.write(json.dumps({'error': 'Config file not found'}).encode())
            return
        
        # Serve credentials for selected profile (only if explicitly requested)
        elif self.path.startswith('/api/credentials/'):
            profile_name = self.path.split('/')[-1]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            config_path = Path(__file__).parent / 'web_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    profile = config.get('profiles', {}).get(profile_name, {})
                    credentials = {
                        'base_url': profile.get('base_url', ''),
                        'api_key': profile.get('api_key', ''),
                        'secret_key': profile.get('secret_key', '')
                    }
                    self.wfile.write(json.dumps(credentials).encode())
            else:
                self.wfile.write(json.dumps({'error': 'Config file not found'}).encode())
            return
        
        # Try to serve the file
        try:
            super().do_GET()
        except Exception as e:
            self.send_error(404, f'File not found: {e}')

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

    def log_message(self, format, *args):
        """Custom logging"""
        # Format: [timestamp] message
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{timestamp}] {format % args}', file=sys.stderr)


def run_server(host='127.0.0.1', port=8080):
    """Run the HTTP server"""
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, FaceAuthHTTPHandler)
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║     Face Recognition WebSocket Client - Web Server             ║
╚════════════════════════════════════════════════════════════════╝

✅ Server Configuration:
   Host: {host}
   Port: {port}
   URL: http://{host}:{port}
   
📁 Serving from: {script_dir}

📋 Usage:
   1. Open your browser and go to: http://{host}:{port}
   2. Configure the server settings (Base URL, API Key, Secret Key)
   3. Select enrollment or authentication mode
   4. Click "Connect & Start" to begin

⚠️  Security Notes:
   - This is for development use only
   - Never expose this server to the internet without authentication
   - Keep your API keys and secret keys secure

📚 API Configuration:
   - Base URL: Your Face Recognition API server URL
   - API Key: Your API credentials
   - Secret Key: Your API secret credentials

🛑 Press Ctrl+C to stop the server

════════════════════════════════════════════════════════════════════
""")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n\n⏹️  Server stopped by user')
        httpd.server_close()
        sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Face Recognition WebSocket Client - Web Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on default localhost:8080
  python web_server.py

  # Run on specific host and port
  python web_server.py --host 0.0.0.0 --port 5000

  # Run on specific port only
  python web_server.py --port 3000
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (default: 8080)'
    )
    
    parser.add_argument(
        '--public',
        action='store_true',
        help='Bind to 0.0.0.0 to allow external connections'
    )
    
    args = parser.parse_args()
    
    host = '0.0.0.0' if args.public else args.host
    port = args.port
    
    run_server(host, port)


if __name__ == '__main__':
    main()
