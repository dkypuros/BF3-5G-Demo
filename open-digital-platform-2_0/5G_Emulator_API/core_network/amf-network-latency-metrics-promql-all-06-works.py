import requests
import csv
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prometheus server URL for the range query endpoint
prometheus_url = 'http://opentelemetry.planolab.io:9090/api/v1/query_range'

# Function to perform a range query to Prometheus
def range_query_prometheus(query, start, end, step):
    """Range query Prometheus and return the response."""
    params = {
        'query': query,
        'start': start,
        'end': end,
        'step': step
    }
    try:
        response = requests.get(prometheus_url, params=params)
        response.raise_for_status()
        json_response = response.json()
        logging.info(f"Prometheus response: {json_response}")
        return json_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Prometheus: {e}")
        return None

# Function to save the query results to a CSV file
def save_results_to_csv(results, output_file):
    """Save the raw time series data to a CSV file."""
    try:
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Cumulative_Sum_Value'])  # CSV headers

            if results and results.get('data') and results['data'].get('result'):
                for result in results['data']['result']:
                    for value in result['values']:
                        timestamp = datetime.utcfromtimestamp(value[0]).strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([timestamp, value[1]])

                logging.info(f"Results successfully saved to {output_file}")
            else:
                logging.warning(f"No data available in the query response")
    except Exception as e:
        logging.error(f"Error writing results to CSV file: {e}")

# Time range setup
end_time = datetime.utcnow()  # Current time as end time
start_time = end_time - timedelta(days=1)  # Start time 1 day ago

# Convert to Prometheus-compatible timestamps
start_timestamp = start_time.timestamp()
end_timestamp = end_time.timestamp()

# Prometheus query for cumulative sum
query = 'ngap_handover_duration_seconds_sum'

# Query step interval (e.g., 5-minute intervals)
step = 60  # Change this if needed for finer resolution

# Perform the range query and get the results
logging.info(f"Querying data from {start_time} to {end_time}")
prometheus_results = range_query_prometheus(query, start_timestamp, end_timestamp, step)

# Save raw results to a CSV file
output_csv = 'amf-handover-request-durations.csv'  # Desired output CSV file
save_results_to_csv(prometheus_results, output_csv)
