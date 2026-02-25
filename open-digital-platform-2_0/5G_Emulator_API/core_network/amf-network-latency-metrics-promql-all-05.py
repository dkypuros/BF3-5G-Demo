import requests
from tabulate import tabulate
from datetime import datetime, timedelta
import argparse

def query_prometheus(url, query, start, end, step):
    params = {
        'query': query,
        'start': start.timestamp(),
        'end': end.timestamp(),
        'step': step
    }
    full_url = f"{url}/api/v1/query_range"
    print(f"Querying: {full_url}")
    print(f"Params: {params}")
    try:
        response = requests.get(full_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Prometheus: {e}")
        return None

def format_value(value):
    try:
        return f"{float(value):.6f}"
    except ValueError:
        return "N/A"

def parse_time_string(time_string):
    units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    unit = time_string[-1]
    if unit not in units:
        raise ValueError(f"Invalid time unit: {unit}")
    try:
        value = int(time_string[:-1])
        return value * units[unit]
    except ValueError:
        raise ValueError(f"Invalid time string: {time_string}")

def main():
    parser = argparse.ArgumentParser(description="Query Prometheus for AMF network latency metrics")
    parser.add_argument("--url", default="http://opentelemetry.planolab.io:9090", help="Prometheus server URL")
    parser.add_argument("--range", default="15m", help="Time range (e.g., 15m, 1h, 24h)")
    parser.add_argument("--step", default="15s", help="Step interval (e.g., 15s, 1m, 5m)")
    args = parser.parse_args()

    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=parse_time_string(args.range))

    raw_queries = [
        'ngap_handover_duration_seconds_sum',
        'ngap_handover_duration_seconds_count'
    ]

    for query in raw_queries:
        print(f"\nQuerying raw metric: {query}")
        result = query_prometheus(args.url, query, start_time, end_time, args.step)
        if result and 'data' in result and 'result' in result['data']:
            for series in result['data']['result']:
                print(f"Metric: {series['metric']}")
                print("First few values:", series['values'][:5])
                print("Last few values:", series['values'][-5:])
        else:
            print("No data or unexpected response structure")

    # If raw metrics show data, then query the rate
    rate_queries = [
        'rate(ngap_handover_duration_seconds_sum[5m])',
        'rate(ngap_handover_duration_seconds_count[5m])'
    ]

    for query in rate_queries:
        print(f"\nQuerying rate: {query}")
        result = query_prometheus(args.url, query, start_time, end_time, args.step)
        if result and 'data' in result and 'result' in result['data']:
            for series in result['data']['result']:
                print(f"Metric: {series['metric']}")
                print("First few values:", series['values'][:5])
                print("Last few values:", series['values'][-5:])
        else:
            print("No data or unexpected response structure")

if __name__ == "__main__":
    main()