
Initial UE Context Setup.

```bash

curl -X POST http://localhost:9000/amf/ue/ue001 -H "Content-Type: application/json" -d '{"initial_gnb_id": "gnb001"}'

{"message":"UE context created"}

```

Quick Peek at Context

```bash
curl -X GET http://localhost:9000/amf/ue/ue001

```

Request or "trigger" Handover from UE

```bash

curl -X POST http://localhost:9000/amf/handover -H "Content-Type: application/json" -d '{"ue_id": "ue001", "source_gnb_id": "gnb001"}'

{"message":"Handover process completed","duration":0.09120726585388184}

```

ngap_handover_request_ack

- in the terminal (jaeger agent, collector, and query would persist data)

```bash
curl -X POST http://localhost:9000/amf/handover -H "Content-Type: application/json" -d '{"ue_id": "ue001", "source_gnb_id": "gnb001"}'

{"message":"Handover process completed","duration":0.09120726585388184}
```

Peek at New Context

```bash

curl -X GET http://localhost:9000/amf/ue/ue001

```


Prometheus PromQL

```bash
histogram_quantile(0.75, sum(rate(ngap_handover_duration_seconds_bucket[5m])) by (le))
```

```bash
# total number of handover requests
ngap_handover_requests_total

# rate of handover requests per second over the last 5 minutes
rate(ngap_handover_requests_total[5m])
ngap_handover_duration_seconds_bucket

# add multiple panels - compare different percentiles side by side
histogram_quantile(0.95, rate(ngap_handover_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(ngap_handover_duration_seconds_bucket[5m]))
histogram_quantile(0.75, sum(rate(ngap_handover_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.5, rate(ngap_handover_duration_seconds_bucket[5m]))

# average duration of handovers over the last 5 minutes
rate(ngap_handover_duration_seconds_sum[5m]) / rate(ngap_handover_duration_seconds_count[5m])
```


Anomolies Prometheus PromQL

```bash
# Sudden spikes in request rate
# This query will return non-zero values when the current 5-minute rate is more than double the average rate over the last hour
rate(ngap_handover_requests_total[5m]) > 2 * avg_over_time(rate(ngap_handover_requests_total[5m])[1h:5m])

# Unusual increase in 95th percentile latency
#This will detect when the current 95th percentile is 50% higher than its average over the last hour.
# Need to be done outside of Prometheus GUI but you can still use PromQL
histogram_quantile(0.95, rate(ngap_handover_duration_seconds_bucket[5m])) 
  > 
1.5 * avg_over_time(histogram_quantile(0.95, rate(ngap_handover_duration_seconds_bucket[5m]))[1h:5m])

# Detecting outliers using standard deviation
# This query identifies when the current 95th percentile deviates by more than 2 standard deviations from its 1-hour average.

(rate(ngap_handover_requests_total[5m]) - rate(ngap_handover_success_total[5m])) 
  / 
rate(ngap_handover_requests_total[5m]) > 0.1

# Detecting a shift in the ratio between 99th and 50th percentiles
# This detects when the spread between 99th and 50th percentiles increases significantly, which could indicate increased variability
histogram_quantile(0.99, rate(ngap_handover_duration_seconds_bucket[5m])) 
  / 
histogram_quantile(0.5, rate(ngap_handover_duration_seconds_bucket[5m])) 
  > 
1.5 * avg_over_time(
  histogram_quantile(0.99, rate(ngap_handover_duration_seconds_bucket[5m])) 
  / 
  histogram_quantile(0.5, rate(ngap_handover_duration_seconds_bucket[5m]))[1h:5m]
)

# Detecting a sudden increase in maximum latency
max_over_time(ngap_handover_duration_seconds_bucket{le="+Inf"}[5m]) 
  > 
2 * max_over_time(ngap_handover_duration_seconds_bucket{le="+Inf"}[1h])
```