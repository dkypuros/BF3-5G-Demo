Prometheus works differently from traditional relational databases like MySQL. Prometheus is a time-series database designed specifically for monitoring and storing metrics. Here's a basic overview of how you interact with Prometheus and the kind of operations you can perform:

### Data Model

Prometheus stores data as time series, which are identified by a metric name and a set of key-value pairs (labels). Each time series consists of a sequence of timestamped values.

### Basic Operations

1. **Data Ingestion**:
   Prometheus scrapes metrics from configured targets at regular intervals. Metrics are usually exposed by applications via HTTP endpoints (often `/metrics`) in a format that Prometheus can understand.

2. **Configuration**:
   The main configuration file is `prometheus.yml`, where you define the scrape targets and various settings.

3. **Querying Data**:
   Prometheus provides its own query language called PromQL. You use PromQL to query and extract data from Prometheus.

### Example Interactions

1. **Scrape Configuration**:

```bash
sudo vi /etc/prometheus/prometheus.yml
sudo systemctl restart prometheus
```

   In the `prometheus.yml` file, you define where Prometheus should scrape metrics from. Here’s a simple example for AMF:

   ```yaml
   see file in tree ...
   ```

```bash
curl http://localhost:9100/metrics
```

2. **Metrics Endpoint**:
   Your application should expose metrics at an HTTP endpoint (e.g., `/metrics`). Prometheus will scrape this endpoint to collect metrics.

3. **Running Queries**:
   You interact with Prometheus primarily through its web UI or API. Here are some example PromQL queries:

   - Get the current value of a metric:
     ```
     my_metric_name
     ```

   - Compute the rate of increase per second for a counter over the last 5 minutes:
     ```
     rate(my_counter_metric[5m])
     ```

   - Get the average value of a metric over the past hour:
     ```
     avg_over_time(my_metric_name[1h])
     ```

### Example PromQL Queries

- **Querying all time series with a specific label**:
  ```
  up{job="myapp"}
  ```

- **Calculating the sum of a metric across all instances**:
  ```
  sum(http_requests_total)
  ```

- **Getting the average CPU usage over the last 10 minutes**:
  ```
  avg(rate(node_cpu_seconds_total[10m]))
  ```

### Practical Steps

1. **Expose Metrics in Your Application**:
   Ensure your applications expose metrics in a format Prometheus understands. Popular libraries for different languages are:
   - Go: `prometheus/client_golang`
   - Python: `prometheus_client`
   - Java: `prometheus/client_java`

2. **Configure Prometheus**:
   Update the `prometheus.yml` file with the endpoints where your metrics are exposed.

3. **Query Metrics**:
   Use the Prometheus web UI to explore your metrics with PromQL.

4. **Visualization**:
   For more advanced visualization, integrate Prometheus with Grafana. Grafana can connect to Prometheus and provide powerful dashboards and visualization tools.

### Example `prometheus.yml`

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'myapp'
    static_configs:
      - targets: ['localhost:8080']
```

### Example PromQL Queries

- **Total HTTP requests over the last 5 minutes**:
  ```
  sum(rate(http_requests_total[5m]))
  ```

- **Average memory usage over the last hour**:
  ```
  avg_over_time(node_memory_Active_bytes[1h])
  ```

By understanding these concepts and using the tools provided by Prometheus, you can effectively monitor and query your metrics data. If you have any specific questions or need further assistance, feel free to ask!