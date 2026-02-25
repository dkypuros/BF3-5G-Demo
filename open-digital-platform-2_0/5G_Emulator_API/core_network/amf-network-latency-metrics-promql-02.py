import requests
import json
from datetime import datetime

# Prometheus server URL
prometheus_url = 'http://opentelemetry.planolab.io:9090/api/v1/query_range'

# PromQL query to pull time series data
query = 'rate(ngap_handover_requests_total[5m])'

# Define the specific active time window you identified
start_time = '2024-10-07T01:30:10Z'  # Adjusted start time
end_time = '2024-10-07T01:44:10Z'    # Adjusted end time

# Time step (1-minute intervals)
step = '60s'

# Query parameters for Prometheus /query_range API
params = {
    'query': query,
    'start': start_time,
    'end': end_time,
    'step': step
}

# Function to query Prometheus /query_range API
def query_prometheus_range(query_params):
    try:
        response = requests.get(prometheus_url, params=query_params)

        if response.status_code == 200:
            result = response.json()
            return result['data']['result']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Failed to query Prometheus: {str(e)}")
        return None

# Execute the query
result = query_prometheus_range(params)

# Process and print the time series data
if result:
    print("Time Series Data:")
    for metric in result:
        print(f"\nMetric: {metric['metric']}")
        for value in metric['values']:
            timestamp = datetime.utcfromtimestamp(float(value[0])).strftime('%Y-%m-%d %H:%M:%S')
            value_float = float(value[1])

            # Print each timestamp and value
            print(f"Time: {timestamp}, Value: {value_float}")
else:
    print("No data returned.")
