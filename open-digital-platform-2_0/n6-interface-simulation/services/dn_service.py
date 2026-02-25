#!/usr/bin/env python3
"""
Data Network (DN) Service
Simulates an application server on the Data Network side of the N6 interface.
Listens on port 8001 for incoming HTTP requests.
"""

import http.server
import socketserver
import sys

PORT = 8001

class DNRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"[DN] Request received from {self.client_address[0]}:{self.client_address[1]}")
        print(f"[DN] Path requested: {self.path}")
        
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        response_message = "Hello from the Data Network! Your request successfully reached the DN service."
        self.wfile.write(response_message.encode())
        print("[DN] Response sent successfully")
    
    def log_message(self, format, *args):
        # Custom logging format
        print(f"[DN] {self.address_string()} - {format % args}")

def main():
    try:
        with socketserver.TCPServer(("", PORT), DNRequestHandler) as httpd:
            print(f"=" * 60)
            print(f"Data Network (DN) Service")
            print(f"=" * 60)
            print(f"[DN] Service listening on port {PORT}")
            print(f"[DN] Simulating application server on Data Network")
            print(f"[DN] Press Ctrl+C to stop the service")
            print(f"=" * 60)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[DN] Service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"[DN] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()