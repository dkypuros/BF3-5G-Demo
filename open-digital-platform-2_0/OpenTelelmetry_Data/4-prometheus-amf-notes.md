
```bash
curl http://localhost:9100/metrics
```

```bash
[dkypuros@dave-thinkpad open-digital-platform-2_0]$ curl http://localhost:9100/metrics
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 1693.0
python_gc_objects_collected_total{generation="1"} 4097.0
python_gc_objects_collected_total{generation="2"} 613.0
# HELP python_gc_objects_uncollectable_total Uncollectable objects found during GC
# TYPE python_gc_objects_uncollectable_total counter
python_gc_objects_uncollectable_total{generation="0"} 0.0
python_gc_objects_uncollectable_total{generation="1"} 0.0
python_gc_objects_uncollectable_total{generation="2"} 0.0
# HELP python_gc_collections_total Number of times this generation was collected
# TYPE python_gc_collections_total counter
python_gc_collections_total{generation="0"} 164.0
python_gc_collections_total{generation="1"} 14.0
python_gc_collections_total{generation="2"} 1.0
# HELP python_info Python platform information
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="11",patchlevel="6",version="3.11.6"} 1.0
# HELP process_virtual_memory_bytes Virtual memory size in bytes.
# TYPE process_virtual_memory_bytes gauge
process_virtual_memory_bytes 5.46021376e+08
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 6.1812736e+07
# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.
# TYPE process_start_time_seconds gauge
process_start_time_seconds 1.72054262577e+09
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 0.9199999999999999
# HELP process_open_fds Number of open file descriptors.
# TYPE process_open_fds gauge
process_open_fds 19.0
# HELP process_max_fds Maximum number of open file descriptors.
# TYPE process_max_fds gauge
process_max_fds 524288.0
# HELP amf_requests_total Total AMF requests
# TYPE amf_requests_total counter
amf_requests_total 0.0
# HELP amf_requests_created Total AMF requests
# TYPE amf_requests_created gauge
amf_requests_created 1.720542627186933e+09
# HELP amf_request_latency_seconds AMF request latency
# TYPE amf_request_latency_seconds histogram
amf_request_latency_seconds_bucket{le="0.005"} 0.0
amf_request_latency_seconds_bucket{le="0.01"} 0.0
amf_request_latency_seconds_bucket{le="0.025"} 0.0
amf_request_latency_seconds_bucket{le="0.05"} 0.0
amf_request_latency_seconds_bucket{le="0.075"} 0.0
amf_request_latency_seconds_bucket{le="0.1"} 0.0
amf_request_latency_seconds_bucket{le="0.25"} 0.0
amf_request_latency_seconds_bucket{le="0.5"} 0.0
amf_request_latency_seconds_bucket{le="0.75"} 0.0
amf_request_latency_seconds_bucket{le="1.0"} 0.0
amf_request_latency_seconds_bucket{le="2.5"} 0.0
amf_request_latency_seconds_bucket{le="5.0"} 0.0
amf_request_latency_seconds_bucket{le="7.5"} 0.0
amf_request_latency_seconds_bucket{le="10.0"} 0.0
amf_request_latency_seconds_bucket{le="+Inf"} 0.0
amf_request_latency_seconds_count 0.0
amf_request_latency_seconds_sum 0.0
# HELP amf_request_latency_seconds_created AMF request latency
# TYPE amf_request_latency_seconds_created gauge
amf_request_latency_seconds_created 1.7205426271869643e+09
```


```bash
curl http://localhost:9000/amf_service
```