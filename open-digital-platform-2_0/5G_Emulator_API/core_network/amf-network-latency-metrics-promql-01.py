import requests
import json

# Prometheus URL
prometheus_url = 'http://opentelemetry.planolab.io:9090/api/v1/query'

# List of PromQL queries to test
queries = {
    "Total number of handover requests": 'ngap_handover_requests_total',
    "Rate of handover requests per second (last 5 minutes)": 'rate(ngap_handover_requests_total[5m])',
    "95th percentile of handover duration": 'histogram_quantile(0.95, rate(ngap_handover_duration_seconds_bucket[5m]))',
    "99th percentile of handover duration": 'histogram_quantile(0.99, rate(ngap_handover_duration_seconds_bucket[5m]))',
    "75th percentile of handover duration": 'histogram_quantile(0.75, sum(rate(ngap_handover_duration_seconds_bucket[5m])) by (le))',
    "50th percentile (median) of handover duration": 'histogram_quantile(0.5, rate(ngap_handover_duration_seconds_bucket[5m]))'
}

# Function to query Prometheus
def query_prometheus(promql_query):
    try:
        # Send the query to the Prometheus API
        response = requests.get(prometheus_url, params={'query': promql_query})

        # Print the raw response for debugging
        if response.status_code == 200:
            result = response.json()
            return result['data']['result']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Failed to query Prometheus: {str(e)}")
        return None

# Run each query and display the results
for description, query in queries.items():
    print(f"\nQuerying: {description}")
    result = query_prometheus(query)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("No data returned.")
