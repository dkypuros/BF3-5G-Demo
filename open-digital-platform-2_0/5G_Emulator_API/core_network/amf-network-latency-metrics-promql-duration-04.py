import requests
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)

# Prometheus API URL
PROMETHEUS_API_URL = 'http://opentelemetry.planolab.io:9090/api/v1/query'

# Query to fetch ngap_handover_duration_seconds_sum
PROMETHEUS_QUERY = 'ngap_handover_duration_seconds_sum'

def query_prometheus(prometheus_url, query):
    try:
        # Construct the full URL
        full_url = f"{prometheus_url}?query={query}"
        # Send the GET request to the Prometheus API
        response = requests.get(full_url)
        # Check if the response status is OK (200)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Error querying Prometheus: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error querying Prometheus: {str(e)}")
        return None

def process_prometheus_data(data):
    if data and 'status' in data and data['status'] == 'success':
        result = data['data']['result']
        for metric in result:
            # Extract timestamp and duration sum value
            timestamp = metric['value'][0]
            duration_sum = float(metric['value'][1])
            # Convert the timestamp to human-readable format
            readable_timestamp = datetime.utcfromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
            # Print the processed result
            logging.info(f"Timestamp: {readable_timestamp}, Duration Sum: {duration_sum}")
    else:
        logging.error("No data returned from Prometheus.")

if __name__ == "__main__":
    # Query Prometheus for the handover duration sum
    logging.info(f"Querying Prometheus for {PROMETHEUS_QUERY}...")
    prometheus_data = query_prometheus(PROMETHEUS_API_URL, PROMETHEUS_QUERY)
    
    # Process the returned data
    process_prometheus_data(prometheus_data)
