Detailed output with a 5-minute step interval over 24 hours. This gives us a much clearer picture of what's happening. Let's analyze the results:

1. Handover Duration Sum:
- Started at 0.09120726585388184
- Ended at 41.15698862075806
- Maximum difference between points: 252.02037000656128

2. Handover Count:
- Started at 1
- Ended at 444
- Maximum difference between points: 208.0

3. Handover Requests:
- Started at 1
- Ended at 444
- Maximum difference between points: 209.0

4. Average Handover Duration:
- Started at 0.091207 seconds
- Had a significant spike to 252.112 seconds from 2024-10-06 10:17:19 to 2024-10-06 12:37:19 (about 2 hours and 20 minutes)
- After the spike, settled to around 0.092696 seconds

5. Rate Calculation:
- Mostly 'NaN' values
- Maximum difference between consecutive rate values: 0.000047 (very small)

Analysis:

1. Normal Operation: For most of the day, the average handover duration was consistently around 0.091-0.092 seconds, which seems to be the normal operation.

2. Significant Anomaly: There was a major spike in handover duration from 10:17 AM to 12:37 PM, lasting about 2 hours and 20 minutes. During this time, the average handover duration increased dramatically to 252.112 seconds (over 4 minutes).

3. Recovery: After the anomaly, the system seems to have recovered, with handover durations returning to slightly higher than the initial values (0.092696 seconds vs 0.091207 seconds).

4. Handover Activity: The system processed 443 handovers during this 24-hour period, with the most significant activity likely occurring during the anomaly period.

5. Rate Calculation: The rate calculation is still producing mostly 'NaN' values, which is expected for counter metrics over longer periods. The small maximum difference (0.000047) suggests very little variation in the rate when it was calculable.

Recommendations:

1. Investigate the Anomaly: Focus on what happened between 10:17 AM and 12:37 PM on 2024-10-06. This significant spike in handover duration could indicate a serious network issue, system overload, or configuration problem.

2. Correlation with Other Metrics: Check other system and network metrics for the same time period to see if there were any correlated events (e.g., high CPU usage, network congestion, memory issues).

3. Set Up Alerts: Implement alerting for when the average handover duration exceeds a certain threshold (e.g., 1 second) for more than a few minutes.

4. Regular Monitoring: Continue monitoring these metrics closely, perhaps with a dashboard that shows the last 24 hours of data with 5-minute resolution.

5. Handover Process Analysis: Look into why some handovers are taking longer than others, even during normal operation. The slight increase in average duration after the anomaly (from 0.091 to 0.092 seconds) might be worth investigating.

6. Rate Calculation Adjustment: For more meaningful rate calculations, consider using a shorter time window in the Prometheus query, e.g., `rate(ngap_handover_duration_seconds_sum[1m]) / rate(ngap_handover_duration_seconds_count[1m])`.

7. Capacity Planning: Analyze if the number of handovers (444 in 24 hours) is within expected ranges. If it's lower or higher than expected, it might indicate issues with network coverage or user behavior.

8. Historical Data: Keep this data for long-term trend analysis. It will be valuable to compare future performance against this baseline.

This detailed view has revealed a significant anomaly that wasn't as clear in the previous, coarser-grained data. It emphasizes the importance of monitoring at appropriate time scales and investigating any unusual patterns promptly.


```bash
venv) [student@telcocloud core_network]$ python amf-network-latency-metrics-promql-all-06.py --range 1d --step 5m
Querying from 2024-10-05 22:59:35.957186 to 2024-10-06 22:59:35.957186

Querying: ngap_handover_duration_seconds_sum
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_duration_seconds_sum', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}
Metric: {'__name__': 'ngap_handover_duration_seconds_sum', 'instance': 'telco_cloud.planolab.io:9100', 'job': 'amf'}
First value: [1728187175.957, '0.09120726585388184']
Last value: [1728273575.957, '41.15698862075806']
Maximum difference between consecutive points: 252.02037000656128

Querying: ngap_handover_duration_seconds_count
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_duration_seconds_count', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}
Metric: {'__name__': 'ngap_handover_duration_seconds_count', 'instance': 'telco_cloud.planolab.io:9100', 'job': 'amf'}
First value: [1728187175.957, '1']
Last value: [1728273575.957, '444']
Maximum difference between consecutive points: 243.0

Querying: ngap_handover_requests_total
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_requests_total', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}
Metric: {'__name__': 'ngap_handover_requests_total', 'instance': 'telco_cloud.planolab.io:9100', 'job': 'amf'}
First value: [1728187175.957, '1']
Last value: [1728273575.957, '444']
Maximum difference between consecutive points: 243.0

Querying: ngap_handover_success_total
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_success_total', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}

Querying: ngap_handover_failure_total
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_failure_total', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}

Querying: ngap_handover_requested_total
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_requested_total', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}

Querying: ngap_handover_resource_allocation_attempted_total
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_resource_allocation_attempted_total', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}

Querying: ngap_handover_resource_allocation_successful_total
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_resource_allocation_successful_total', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_duration_seconds_sum', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'ngap_handover_duration_seconds_count', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}

Average Handover Duration:
Timestamp              Avg Handover Duration (s)
-------------------  ---------------------------
2024-10-05 22:59:35                     0.091207
2024-10-05 23:04:35                     0.091207
2024-10-05 23:09:35                     0.091207
2024-10-05 23:14:35                     0.091207
2024-10-05 23:19:35                     0.091207
2024-10-05 23:24:35                     0.091207
2024-10-05 23:29:35                     0.091207
2024-10-05 23:34:35                     0.091207
2024-10-05 23:39:35                     0.091207
2024-10-05 23:44:35                     0.091207
2024-10-05 23:49:35                     0.091207
2024-10-05 23:54:35                     0.091207
2024-10-05 23:59:35                     0.091207
2024-10-06 00:04:35                     0.091207
2024-10-06 00:09:35                     0.091207
2024-10-06 00:14:35                     0.091207
2024-10-06 00:19:35                     0.091207
2024-10-06 00:24:35                     0.091207
2024-10-06 00:29:35                     0.091207
2024-10-06 00:34:35                     0.091207
2024-10-06 00:39:35                     0.091207
2024-10-06 00:44:35                     0.091207
2024-10-06 00:49:35                     0.091207
2024-10-06 00:54:35                     0.091207
2024-10-06 00:59:35                     0.091207
2024-10-06 01:04:35                     0.091207
2024-10-06 01:09:35                     0.091207
2024-10-06 01:14:35                     0.091207
2024-10-06 01:19:35                     0.091207
2024-10-06 01:24:35                     0.091207
2024-10-06 01:29:35                     0.091207
2024-10-06 01:34:35                     0.091207
2024-10-06 01:39:35                     0.091207
2024-10-06 01:44:35                     0.091207
2024-10-06 01:49:35                     0.091207
2024-10-06 01:54:35                     0.091207
2024-10-06 01:59:35                     0.091207
2024-10-06 02:04:35                     0.091207
2024-10-06 02:09:35                     0.091207
2024-10-06 02:14:35                     0.091207
2024-10-06 02:19:35                     0.091207
2024-10-06 02:24:35                     0.091207
2024-10-06 02:29:35                     0.091207
2024-10-06 02:34:35                     0.091207
2024-10-06 02:39:35                     0.091207
2024-10-06 02:44:35                     0.091207
2024-10-06 02:49:35                     0.091207
2024-10-06 02:54:35                     0.091207
2024-10-06 02:59:35                     0.091207
2024-10-06 03:04:35                     0.091207
2024-10-06 03:09:35                     0.091207
2024-10-06 03:14:35                     0.091207
2024-10-06 03:19:35                     0.091207
2024-10-06 03:24:35                     0.091207
2024-10-06 03:29:35                     0.091207
2024-10-06 03:34:35                     0.091207
2024-10-06 03:39:35                     0.091207
2024-10-06 03:44:35                     0.091207
2024-10-06 03:49:35                     0.091207
2024-10-06 03:54:35                     0.091207
2024-10-06 03:59:35                     0.091207
2024-10-06 04:04:35                     0.091207
2024-10-06 04:09:35                     0.091207
2024-10-06 04:14:35                     0.091207
2024-10-06 04:19:35                     0.091207
2024-10-06 04:24:35                     0.091207
2024-10-06 04:29:35                     0.091207
2024-10-06 04:34:35                     0.091207
2024-10-06 04:39:35                     0.091207
2024-10-06 04:44:35                     0.091207
2024-10-06 04:49:35                     0.091207
2024-10-06 04:54:35                     0.091207
2024-10-06 04:59:35                     0.091207
2024-10-06 05:04:35                     0.091207
2024-10-06 05:09:35                     0.091207
2024-10-06 05:14:35                     0.091207
2024-10-06 05:19:35                     0.091207
2024-10-06 05:24:35                     0.091207
2024-10-06 05:29:35                     0.091207
2024-10-06 05:34:35                     0.091207
2024-10-06 05:39:35                     0.091207
2024-10-06 05:44:35                     0.091207
2024-10-06 05:49:35                     0.091207
2024-10-06 05:54:35                     0.091207
2024-10-06 05:59:35                     0.091207
2024-10-06 06:04:35                     0.091207
2024-10-06 06:09:35                     0.091207
2024-10-06 06:14:35                     0.091207
2024-10-06 06:19:35                     0.091207
2024-10-06 06:24:35                     0.091207
2024-10-06 06:29:35                     0.091207
2024-10-06 06:34:35                     0.091207
2024-10-06 06:39:35                     0.091207
2024-10-06 06:44:35                     0.091207
2024-10-06 06:49:35                     0.091207
2024-10-06 06:54:35                     0.091207
2024-10-06 06:59:35                     0.091207
2024-10-06 07:04:35                     0.091207
2024-10-06 07:09:35                     0.091207
2024-10-06 07:14:35                     0.091207
2024-10-06 07:19:35                     0.091207
2024-10-06 07:24:35                     0.091207
2024-10-06 07:29:35                     0.091207
2024-10-06 07:34:35                     0.091207
2024-10-06 07:39:35                     0.091207
2024-10-06 07:44:35                     0.091207
2024-10-06 07:49:35                     0.091207
2024-10-06 07:54:35                     0.091207
2024-10-06 07:59:35                     0.091207
2024-10-06 08:04:35                     0.091207
2024-10-06 08:09:35                     0.091207
2024-10-06 08:14:35                     0.091207
2024-10-06 08:19:35                     0.091207
2024-10-06 08:24:35                     0.091207
2024-10-06 08:29:35                     0.091207
2024-10-06 08:34:35                     0.091207
2024-10-06 08:39:35                     0.091207
2024-10-06 08:44:35                     0.091207
2024-10-06 08:49:35                     0.091207
2024-10-06 08:54:35                     0.091207
2024-10-06 08:59:35                     0.091207
2024-10-06 09:04:35                     0.091207
2024-10-06 09:09:35                     0.091207
2024-10-06 09:14:35                     0.091207
2024-10-06 09:19:35                     0.091207
2024-10-06 09:24:35                     0.091207
2024-10-06 09:29:35                     0.091207
2024-10-06 09:34:35                     0.091207
2024-10-06 09:39:35                     0.091207
2024-10-06 09:44:35                     0.091207
2024-10-06 09:49:35                     0.091207
2024-10-06 09:54:35                     0.091207
2024-10-06 09:59:35                     0.091207
2024-10-06 10:19:35                   252.112
2024-10-06 10:24:35                   252.112
2024-10-06 10:29:35                   252.112
2024-10-06 10:34:35                   252.112
2024-10-06 10:39:35                   252.112
2024-10-06 10:44:35                   252.112
2024-10-06 10:49:35                   252.112
2024-10-06 10:54:35                   252.112
2024-10-06 10:59:35                   252.112
2024-10-06 11:04:35                   252.112
2024-10-06 11:09:35                   252.112
2024-10-06 11:14:35                   252.112
2024-10-06 11:19:35                   252.112
2024-10-06 11:24:35                   252.112
2024-10-06 11:29:35                   252.112
2024-10-06 11:34:35                   252.112
2024-10-06 11:39:35                   252.112
2024-10-06 11:44:35                   252.112
2024-10-06 11:49:35                   252.112
2024-10-06 11:54:35                   252.112
2024-10-06 11:59:35                   252.112
2024-10-06 12:04:35                   252.112
2024-10-06 12:09:35                   252.112
2024-10-06 12:14:35                   252.112
2024-10-06 12:19:35                   252.112
2024-10-06 12:24:35                   252.112
2024-10-06 12:29:35                   252.112
2024-10-06 12:34:35                   252.112
2024-10-06 12:39:35                   252.112
2024-10-06 17:19:35                     0.092776
2024-10-06 17:24:35                     0.092776
2024-10-06 17:29:35                     0.092776
2024-10-06 17:34:35                     0.092776
2024-10-06 17:39:35                     0.092776
2024-10-06 17:44:35                     0.092776
2024-10-06 17:49:35                     0.092776
2024-10-06 17:54:35                     0.092776
2024-10-06 17:59:35                     0.092776
2024-10-06 18:04:35                     0.092776
2024-10-06 18:09:35                     0.092776
2024-10-06 18:14:35                     0.092776
2024-10-06 18:19:35                     0.092776
2024-10-06 18:24:35                     0.092776
2024-10-06 18:29:35                     0.092776
2024-10-06 18:34:35                     0.092776
2024-10-06 18:39:35                     0.092776
2024-10-06 18:44:35                     0.092776
2024-10-06 18:49:35                     0.092776
2024-10-06 18:54:35                     0.092776
2024-10-06 18:59:35                     0.092776
2024-10-06 19:04:35                     0.092776
2024-10-06 19:09:35                     0.092776
2024-10-06 19:14:35                     0.092776
2024-10-06 19:19:35                     0.092776
2024-10-06 19:24:35                     0.092776
2024-10-06 19:29:35                     0.092776
2024-10-06 19:34:35                     0.092776
2024-10-06 19:39:35                     0.092776
2024-10-06 19:44:35                     0.092776
2024-10-06 19:49:35                     0.092776
2024-10-06 19:54:35                     0.092776
2024-10-06 19:59:35                     0.092776
2024-10-06 20:04:35                     0.092776
2024-10-06 20:09:35                     0.092776
2024-10-06 20:34:35                     0.092638
2024-10-06 20:39:35                     0.092699
2024-10-06 20:44:35                     0.092696
2024-10-06 20:49:35                     0.092696
2024-10-06 20:54:35                     0.092696
2024-10-06 20:59:35                     0.092696
2024-10-06 21:04:35                     0.092696
2024-10-06 21:09:35                     0.092696
2024-10-06 21:14:35                     0.092696
2024-10-06 21:19:35                     0.092696
2024-10-06 21:24:35                     0.092696
2024-10-06 21:29:35                     0.092696
2024-10-06 21:34:35                     0.092696
2024-10-06 21:39:35                     0.092696
2024-10-06 21:44:35                     0.092696
2024-10-06 21:49:35                     0.092696
2024-10-06 21:54:35                     0.092696
2024-10-06 21:59:35                     0.092696
2024-10-06 22:04:35                     0.092696
2024-10-06 22:09:35                     0.092696
2024-10-06 22:14:35                     0.092696
2024-10-06 22:19:35                     0.092696
2024-10-06 22:24:35                     0.092696
2024-10-06 22:29:35                     0.092696
2024-10-06 22:34:35                     0.092696
2024-10-06 22:39:35                     0.092696
2024-10-06 22:44:35                     0.092696
2024-10-06 22:49:35                     0.092696
2024-10-06 22:54:35                     0.092696
2024-10-06 22:59:35                     0.092696

Maximum difference between consecutive average durations: 252.020370

Querying rate: rate(ngap_handover_duration_seconds_sum[5m]) / rate(ngap_handover_duration_seconds_count[5m])
Querying: http://opentelemetry.planolab.io:9090/api/v1/query_range
Params: {'query': 'rate(ngap_handover_duration_seconds_sum[5m]) / rate(ngap_handover_duration_seconds_count[5m])', 'start': 1728187175.957186, 'end': 1728273575.957186, 'step': '5m'}
Metric: {'instance': 'telco_cloud.planolab.io:9100', 'job': 'amf'}
First few values: [[1728187175.957, 'NaN'], [1728187475.957, 'NaN'], [1728187775.957, 'NaN'], [1728188075.957, 'NaN'], [1728188375.957, 'NaN']]
Last few values: [[1728272375.957, 'NaN'], [1728272675.957, 'NaN'], [1728272975.957, 'NaN'], [1728273275.957, 'NaN'], [1728273575.957, 'NaN']]
Maximum difference between consecutive rate values: 0.000189
(venv) [student@telcocloud core_network]$ 

```