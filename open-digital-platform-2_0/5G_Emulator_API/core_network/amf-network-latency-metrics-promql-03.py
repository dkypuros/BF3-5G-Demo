import requests
import csv
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prometheus server URL
prometheus_url = 'http://opentelemetry.planolab.io:9090/api/v1/query_range'

# Function to query Prometheus
def query_prometheus(query, start, end, step='60s'):
    payload = {
        'query': query,
        'start': start,
        'end': end,
        'step': step
    }
    
    logging.info(f"Querying Prometheus with query: {query}")
    try:
        response = requests.get(prometheus_url, params=payload)
        response.raise_for_status()  # Raise an error for bad HTTP responses
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Prometheus: {e}")
        return []

    result = response.json()
    
    # Check if the response is as expected
    if 'status' in result and result['status'] == 'success' and 'data' in result and 'result' in result['data']:
        logging.info(f"Query successful: {query}")
        return result['data']['result']
    else:
        logging.warning(f"Unexpected response structure from Prometheus for query: {query}")
        return []

# Convert UNIX timestamp to readable format
def convert_timestamp(ts):
    try:
        readable_ts = datetime.utcfromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Converted timestamp {ts} to {readable_ts}")
        return readable_ts
    except ValueError as e:
        logging.error(f"Error converting timestamp: {e}")
        return None

# Time range: start and end times (in UNIX timestamps)
start_time = '1728267000'  # Example start time
end_time = '1728268200'    # Example end time

# Queries for raw data points
queries = {
    'handover_requests_total': 'ngap_handover_requests_total',
    'handover_duration_seconds_sum': 'ngap_handover_duration_seconds_sum',
    'handover_duration_seconds_count': 'ngap_handover_duration_seconds_count'
}

# Dictionary to store results
data = {}

# Query each metric
for metric_name, query in queries.items():
    logging.info(f"Starting query for metric: {metric_name}")
    
    results = query_prometheus(query, start_time, end_time)
    
    if not results:
        logging.warning(f"No data returned for query: {query}")
        continue
    
    for result in results:
        for value in result['values']:
            timestamp = convert_timestamp(value[0])
            
            if timestamp is None:
                logging.error(f"Skipping invalid timestamp for metric: {metric_name}")
                continue
            
            if timestamp not in data:
                data[timestamp] = {}
            
            data[timestamp][metric_name] = value[1]
            logging.info(f"Stored value {value[1]} for {metric_name} at {timestamp}")

# Display data in console and save to CSV
csv_filename = 'handover_metrics_timeseries.csv'

try:
    with open(csv_filename, mode='w') as csv_file:
        fieldnames = ['Timestamp'] + list(queries.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write headers to CSV
        writer.writeheader()
        logging.info(f"CSV headers written: {fieldnames}")

        # Write data rows
        for timestamp, metrics in data.items():
            row = {'Timestamp': timestamp}
            row.update(metrics)

            # Handle NaN values in case there are missing calculations
            for key in queries.keys():
                if key not in row or row[key] == 'NaN':
                    logging.warning(f"Missing value for {key} at {timestamp}, marking as 'N/A'")
                    row[key] = 'N/A'  # Handle NaN by marking it as 'N/A'

            # Print to console
            logging.info(f"Writing row to CSV: {row}")
            print(row)

            # Write row to CSV
            writer.writerow(row)

    logging.info(f"Data successfully saved to {csv_filename}")
except IOError as e:
    logging.error(f"Error writing to CSV file: {e}")
