Example PromQL (Prometheus Query Language) queries you can use to test and explore your AMF metrics. These assume you're using the standard metrics we set up earlier, but you may need to adjust them based on your specific implementation:

1. Total number of AMF requests:
   ```
   amf_requests_total
   ```

2. Rate of AMF requests over the last 5 minutes:
   ```
   rate(amf_requests_total[5m])
   ```

3. 95th percentile of AMF request latency over the last hour:
   ```
   histogram_quantile(0.95, rate(amf_request_latency_seconds_bucket[1h]))
   ```

4. Average request latency over the last 5 minutes:
   ```
   rate(amf_request_latency_seconds_sum[5m]) / rate(amf_request_latency_seconds_count[5m])
   ```

5. Number of requests in the last hour:
   ```
   increase(amf_requests_total[1h])
   ```

6. CPU usage of the AMF process (if you're collecting node_exporter metrics):
   ```
   rate(process_cpu_seconds_total{job="amf"}[5m]) * 100
   ```

7. Memory usage of the AMF process (if you're collecting node_exporter metrics):
   ```
   process_resident_memory_bytes{job="amf"}
   ```

8. Number of open file descriptors (if you're collecting node_exporter metrics):
   ```
   process_open_fds{job="amf"}
   ```

9. Request rate by HTTP status code (if you've implemented this metric):
   ```
   rate(http_requests_total{job="amf"}[5m])
   ```

10. Error rate (if you've implemented an error counter):
    ```
    rate(amf_errors_total[5m])
    ```

To use these queries:

1. Go to the Prometheus web interface (typically http://your-prometheus-server:9090)
2. Click on the "Graph" tab
3. Enter the PromQL query in the query box
4. Click "Execute" to see the result

You can adjust the time range and resolution as needed using the controls below the graph.

Remember, some of these queries might not work if you haven't implemented the specific metrics they're querying. You may need to adapt them based on the exact metrics you've defined in your AMF.
