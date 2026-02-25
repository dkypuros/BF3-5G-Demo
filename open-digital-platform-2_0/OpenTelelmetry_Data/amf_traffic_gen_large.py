# File: generate_heavy_traffic.py

import requests
import time
import concurrent.futures

AMF_URL = "http://telco_cloud.planolab.io:9000/amf_service"
TOTAL_REQUESTS = 20000
MAX_WORKERS = 50  # Adjust based on your system's capabilities

def send_request(session):
    try:
        response = session.get(AMF_URL)
        return response.status_code == 200
    except:
        return False

def generate_traffic():
    successful_requests = 0
    start_time = time.time()

    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(send_request, session) for _ in range(TOTAL_REQUESTS)]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful_requests += 1

    end_time = time.time()
    duration = end_time - start_time

    print(f"Sent {TOTAL_REQUESTS} requests in {duration:.2f} seconds")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {TOTAL_REQUESTS - successful_requests}")
    print(f"Requests per second: {TOTAL_REQUESTS / duration:.2f}")

if __name__ == "__main__":
    generate_traffic()