#!/usr/bin/env python3
"""
User Plane Function (UPF) Service
Simulates the UPF sending traffic across the N6 interface to the Data Network.
Sends HTTP requests to the DN service on port 8001.
"""

import requests
import time
import sys
import argparse

DN_HOST = "localhost"
DN_PORT = 8001
DN_URL = f"http://{DN_HOST}:{DN_PORT}"

def send_traffic(iterations=1, delay=2):
    """Send HTTP traffic to the Data Network service."""
    
    print(f"=" * 60)
    print(f"User Plane Function (UPF) Service")
    print(f"=" * 60)
    print(f"[UPF] Service starting...")
    print(f"[UPF] Target DN service: {DN_URL}")
    print(f"[UPF] Will send {iterations} request(s) with {delay}s delay")
    print(f"=" * 60)
    
    # Give the DN service a moment to start if running simultaneously
    time.sleep(2)
    
    successful_requests = 0
    failed_requests = 0
    
    for i in range(iterations):
        try:
            print(f"\n[UPF] Attempt {i+1}/{iterations}")
            print(f"[UPF] Sending HTTP GET request to {DN_URL}...")
            
            # Send the request with a timeout
            response = requests.get(DN_URL, timeout=5)
            
            if response.status_code == 200:
                print(f"[UPF] ✅ Success! Response received from DN:")
                print(f"[UPF] Status Code: {response.status_code}")
                print(f"[UPF] Response: {response.text}")
                successful_requests += 1
            else:
                print(f"[UPF] ⚠️ Unexpected status code: {response.status_code}")
                failed_requests += 1
                
        except requests.exceptions.Timeout:
            print(f"[UPF] ❌ Failed! Request timed out.")
            print(f"[UPF] The firewall might be blocking the traffic. 🔥")
            failed_requests += 1
            
        except requests.exceptions.ConnectionError as e:
            print(f"[UPF] ❌ Failed! Connection error.")
            print(f"[UPF] The firewall is likely blocking the connection or DN service is not running. 🔥")
            print(f"[UPF] Error details: {e}")
            failed_requests += 1
            
        except Exception as e:
            print(f"[UPF] ❌ Unexpected error: {e}")
            failed_requests += 1
        
        # Wait before next request (if not the last one)
        if i < iterations - 1:
            print(f"[UPF] Waiting {delay} seconds before next request...")
            time.sleep(delay)
    
    # Print summary
    print(f"\n" + "=" * 60)
    print(f"[UPF] Test Summary:")
    print(f"[UPF] Total requests: {iterations}")
    print(f"[UPF] Successful: {successful_requests}")
    print(f"[UPF] Failed: {failed_requests}")
    
    if failed_requests > 0 and successful_requests == 0:
        print(f"[UPF] 🔥 All requests blocked - Firewall is active!")
    elif failed_requests == 0:
        print(f"[UPF] ✅ All requests successful - No firewall blocking!")
    else:
        print(f"[UPF] ⚠️ Mixed results - Partial blocking detected")
    
    print(f"=" * 60)
    
    return successful_requests, failed_requests

def main():
    parser = argparse.ArgumentParser(description='UPF Service - Simulates N6 interface traffic')
    parser.add_argument('-n', '--iterations', type=int, default=1,
                        help='Number of requests to send (default: 1)')
    parser.add_argument('-d', '--delay', type=int, default=2,
                        help='Delay between requests in seconds (default: 2)')
    parser.add_argument('-t', '--target', type=str, default=DN_HOST,
                        help=f'Target host for DN service (default: {DN_HOST})')
    parser.add_argument('-p', '--port', type=int, default=DN_PORT,
                        help=f'Target port for DN service (default: {DN_PORT})')
    
    args = parser.parse_args()
    
    # Update global variables if custom target/port specified
    global DN_URL
    DN_URL = f"http://{args.target}:{args.port}"
    
    try:
        successful, failed = send_traffic(args.iterations, args.delay)
        sys.exit(0 if failed == 0 else 1)
    except KeyboardInterrupt:
        print("\n[UPF] Service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"[UPF] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()