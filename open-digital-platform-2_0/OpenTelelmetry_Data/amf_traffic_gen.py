import requests
import time
import random

def send_request():
    try:
        response = requests.get("http://telco_cloud.planolab.io:9000/amf_service")
        return response.status_code == 200
    except:
        return False

def generate_traffic(duration_seconds, max_requests_per_second):
    start_time = time.time()
    requests_sent = 0
    successful_requests = 0

    while time.time() - start_time < duration_seconds:
        requests_this_second = random.randint(1, max_requests_per_second)
        for _ in range(requests_this_second):
            if send_request():
                successful_requests += 1
            requests_sent += 1
        time.sleep(1)

    print(f"Sent {requests_sent} requests, {successful_requests} were successful")

# Generate traffic for 5 minutes (300 seconds) with up to 10 requests per second
generate_traffic(300, 10)