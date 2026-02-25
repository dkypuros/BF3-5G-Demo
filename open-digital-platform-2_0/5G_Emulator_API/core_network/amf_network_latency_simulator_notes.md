Key changes to create clear anomalies:

generate_anomaly_periods(): Creates 2-3 random time periods where anomalies will occur.
is_in_anomaly_period(): Checks if the current time is within an anomaly period.
introduce_anomaly(): During anomaly periods, introduces high latency (1-3 seconds) 70% of the time, and very high latency (3-5 seconds) 20% of the time.

This script will now:

Print the anomaly periods at the start, so you know when to expect them.
Create clear spikes in latency during these periods.
Show noticeable changes in the 95th and 99th percentile latencies during and after anomaly periods.

These changes will result in more pronounced and observable anomalies in your traces and latency percentile metrics. You should see:

Clear latency spikes during the anomaly periods.
Significant increases in 95th and 99th percentile latencies during these periods.
Potential lingering effects on percentiles even after the anomaly periods end.

This approach should give you much more visible anomalies to observe and analyze in your monitoring systems.


## Output of Simulation Generation 

```bash
(venv) [student@telcocloud core_network]$ python amf-network_latency_simulator.py 
Anomaly periods: [('33:51', '35:13'), ('37:19', '38:19')]
Starting AMF API latency test...
UE Context Setup for ue055: Latency = 5.18 ms, Response: {'message': 'UE context created'}
Handover for ue055: Latency = 95.24 ms, Response: {'message': 'Handover process completed', 'duration': 0.0917811393737793}
UE Context Setup for ue059: Latency = 3.35 ms, Response: {'message': 'UE context created'}
Handover for ue059: Latency = 95.21 ms, Response: {'message': 'Handover process completed', 'duration': 0.0921630859375}
UE Context Setup for ue039: Latency = 4.18 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 94.86 ms, Response: {'message': 'Handover process completed', 'duration': 0.09187459945678711}
UE Context Setup for ue076: Latency = 4.16 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 96.21 ms, Response: {'message': 'Handover process completed', 'duration': 0.09219121932983398}
UE Context Setup for ue070: Latency = 4.06 ms, Response: {'message': 'UE context created'}
Handover for ue070: Latency = 97.25 ms, Response: {'message': 'Handover process completed', 'duration': 0.0937654972076416}
UE Context Setup for ue054: Latency = 4.37 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 95.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.09222769737243652}
UE Context Setup for ue059: Latency = 3.97 ms, Response: {'message': 'UE context created'}
Handover for ue059: Latency = 95.61 ms, Response: {'message': 'Handover process completed', 'duration': 0.09234929084777832}
UE Context Setup for ue034: Latency = 3.86 ms, Response: {'message': 'UE context created'}
Handover for ue034: Latency = 94.82 ms, Response: {'message': 'Handover process completed', 'duration': 0.09198594093322754}
UE Context Setup for ue091: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 95.10 ms, Response: {'message': 'Handover process completed', 'duration': 0.09173369407653809}
UE Context Setup for ue054: Latency = 3.97 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 95.70 ms, Response: {'message': 'Handover process completed', 'duration': 0.09268569946289062}
UE Context Setup for ue035: Latency = 2.85 ms, Response: {'message': 'UE context created'}
Handover for ue035: Latency = 112.56 ms, Response: {'message': 'Handover process completed', 'duration': 0.10945820808410645}
UE Context Setup for ue072: Latency = 2.56 ms, Response: {'message': 'UE context created'}
Handover for ue072: Latency = 95.34 ms, Response: {'message': 'Handover process completed', 'duration': 0.09213519096374512}
UE Context Setup for ue034: Latency = 3.59 ms, Response: {'message': 'UE context created'}
Handover for ue034: Latency = 95.51 ms, Response: {'message': 'Handover process completed', 'duration': 0.09206914901733398}
UE Context Setup for ue027: Latency = 4.32 ms, Response: {'message': 'UE context created'}
Handover for ue027: Latency = 96.70 ms, Response: {'message': 'Handover process completed', 'duration': 0.09307599067687988}
UE Context Setup for ue015: Latency = 2.44 ms, Response: {'message': 'UE context created'}
Handover for ue015: Latency = 93.65 ms, Response: {'message': 'Handover process completed', 'duration': 0.09160661697387695}
UE Context Setup for ue072: Latency = 3.66 ms, Response: {'message': 'UE context created'}
Handover for ue072: Latency = 94.90 ms, Response: {'message': 'Handover process completed', 'duration': 0.0918431282043457}
UE Context Setup for ue038: Latency = 3.19 ms, Response: {'message': 'UE context created'}
Handover for ue038: Latency = 94.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.09186506271362305}
UE Context Setup for ue050: Latency = 2.67 ms, Response: {'message': 'UE context created'}
Handover for ue050: Latency = 94.86 ms, Response: {'message': 'Handover process completed', 'duration': 0.09224343299865723}
UE Context Setup for ue040: Latency = 3.88 ms, Response: {'message': 'UE context created'}
Handover for ue040: Latency = 95.35 ms, Response: {'message': 'Handover process completed', 'duration': 0.09273338317871094}
UE Context Setup for ue069: Latency = 3.62 ms, Response: {'message': 'UE context created'}
Handover for ue069: Latency = 94.88 ms, Response: {'message': 'Handover process completed', 'duration': 0.0917966365814209}
UE Context Setup for ue078: Latency = 3.09 ms, Response: {'message': 'UE context created'}
Handover for ue078: Latency = 94.91 ms, Response: {'message': 'Handover process completed', 'duration': 0.09233689308166504}
UE Context Setup for ue077: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue077: Latency = 96.26 ms, Response: {'message': 'Handover process completed', 'duration': 0.09307503700256348}
UE Context Setup for ue085: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue085: Latency = 95.07 ms, Response: {'message': 'Handover process completed', 'duration': 0.09244966506958008}
UE Context Setup for ue076: Latency = 4.31 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 94.91 ms, Response: {'message': 'Handover process completed', 'duration': 0.09177279472351074}
UE Context Setup for ue020: Latency = 3.58 ms, Response: {'message': 'UE context created'}
Handover for ue020: Latency = 95.54 ms, Response: {'message': 'Handover process completed', 'duration': 0.09254312515258789}
UE Context Setup for ue015: Latency = 3.78 ms, Response: {'message': 'UE context created'}
Handover for ue015: Latency = 95.68 ms, Response: {'message': 'Handover process completed', 'duration': 0.09271717071533203}
UE Context Setup for ue045: Latency = 3.76 ms, Response: {'message': 'UE context created'}
Handover for ue045: Latency = 96.26 ms, Response: {'message': 'Handover process completed', 'duration': 0.09285664558410645}
UE Context Setup for ue082: Latency = 4.03 ms, Response: {'message': 'UE context created'}
Handover for ue082: Latency = 96.11 ms, Response: {'message': 'Handover process completed', 'duration': 0.09290385246276855}
UE Context Setup for ue033: Latency = 3.68 ms, Response: {'message': 'UE context created'}
Handover for ue033: Latency = 95.95 ms, Response: {'message': 'Handover process completed', 'duration': 0.09227776527404785}
UE Context Setup for ue002: Latency = 2.67 ms, Response: {'message': 'UE context created'}
Handover for ue002: Latency = 93.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.09189105033874512}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.68 ms
  95th Percentile: 4.73 ms
  99th Percentile: 5.73 ms
Handover:
  Average Latency: 95.95 ms
  95th Percentile: 104.14 ms
  99th Percentile: 123.12 ms
UE Context Setup for ue013: Latency = 3.55 ms, Response: {'message': 'UE context created'}
Handover for ue013: Latency = 95.72 ms, Response: {'message': 'Handover process completed', 'duration': 0.0921783447265625}
UE Context Setup for ue012: Latency = 4.08 ms, Response: {'message': 'UE context created'}
Handover for ue012: Latency = 94.22 ms, Response: {'message': 'Handover process completed', 'duration': 0.0918421745300293}
UE Context Setup for ue027: Latency = 3.48 ms, Response: {'message': 'UE context created'}
Handover for ue027: Latency = 94.75 ms, Response: {'message': 'Handover process completed', 'duration': 0.0917806625366211}
UE Context Setup for ue099: Latency = 3.10 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 95.25 ms, Response: {'message': 'Handover process completed', 'duration': 0.09253430366516113}
UE Context Setup for ue076: Latency = 3.86 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 95.80 ms, Response: {'message': 'Handover process completed', 'duration': 0.0924685001373291}
UE Context Setup for ue041: Latency = 3.14 ms, Response: {'message': 'UE context created'}
Handover for ue041: Latency = 95.31 ms, Response: {'message': 'Handover process completed', 'duration': 0.09259319305419922}
UE Context Setup for ue088: Latency = 3.10 ms, Response: {'message': 'UE context created'}
Handover for ue088: Latency = 94.94 ms, Response: {'message': 'Handover process completed', 'duration': 0.09214258193969727}
UE Context Setup for ue033: Latency = 3.82 ms, Response: {'message': 'UE context created'}
Handover for ue033: Latency = 95.96 ms, Response: {'message': 'Handover process completed', 'duration': 0.09273600578308105}
UE Context Setup for ue021: Latency = 4.22 ms, Response: {'message': 'UE context created'}
Handover for ue021: Latency = 95.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.0921626091003418}
UE Context Setup for ue088: Latency = 4.14 ms, Response: {'message': 'UE context created'}
Handover for ue088: Latency = 94.99 ms, Response: {'message': 'Handover process completed', 'duration': 0.09215378761291504}
UE Context Setup for ue074: Latency = 3.83 ms, Response: {'message': 'UE context created'}
Handover for ue074: Latency = 96.90 ms, Response: {'message': 'Handover process completed', 'duration': 0.09345269203186035}
UE Context Setup for ue061: Latency = 4.16 ms, Response: {'message': 'UE context created'}
Handover for ue061: Latency = 94.60 ms, Response: {'message': 'Handover process completed', 'duration': 0.09202909469604492}
UE Context Setup for ue056: Latency = 3.41 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 95.20 ms, Response: {'message': 'Handover process completed', 'duration': 0.09275960922241211}
UE Context Setup for ue077: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue077: Latency = 95.89 ms, Response: {'message': 'Handover process completed', 'duration': 0.09250974655151367}
UE Context Setup for ue042: Latency = 3.22 ms, Response: {'message': 'UE context created'}
Handover for ue042: Latency = 97.11 ms, Response: {'message': 'Handover process completed', 'duration': 0.09341645240783691}
UE Context Setup for ue095: Latency = 4.21 ms, Response: {'message': 'UE context created'}
Handover for ue095: Latency = 96.72 ms, Response: {'message': 'Handover process completed', 'duration': 0.09338951110839844}
UE Context Setup for ue021: Latency = 4.27 ms, Response: {'message': 'UE context created'}
Handover for ue021: Latency = 96.68 ms, Response: {'message': 'Handover process completed', 'duration': 0.09327530860900879}
UE Context Setup for ue039: Latency = 3.77 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 101.37 ms, Response: {'message': 'Handover process completed', 'duration': 0.09834122657775879}
UE Context Setup for ue003: Latency = 2.66 ms, Response: {'message': 'UE context created'}
Handover for ue003: Latency = 94.72 ms, Response: {'message': 'Handover process completed', 'duration': 0.09211254119873047}
UE Context Setup for ue079: Latency = 4.18 ms, Response: {'message': 'UE context created'}
Handover for ue079: Latency = 96.21 ms, Response: {'message': 'Handover process completed', 'duration': 0.09306573867797852}
UE Context Setup for ue054: Latency = 2.49 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 95.03 ms, Response: {'message': 'Handover process completed', 'duration': 0.0923457145690918}
UE Context Setup for ue018: Latency = 2.61 ms, Response: {'message': 'UE context created'}
Handover for ue018: Latency = 94.32 ms, Response: {'message': 'Handover process completed', 'duration': 0.09215664863586426}
UE Context Setup for ue008: Latency = 2.76 ms, Response: {'message': 'UE context created'}
Handover for ue008: Latency = 94.67 ms, Response: {'message': 'Handover process completed', 'duration': 0.09207916259765625}
UE Context Setup for ue018: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue018: Latency = 95.50 ms, Response: {'message': 'Handover process completed', 'duration': 0.0921175479888916}
UE Context Setup for ue095: Latency = 3.78 ms, Response: {'message': 'UE context created'}
Handover for ue095: Latency = 95.53 ms, Response: {'message': 'Handover process completed', 'duration': 0.09229850769042969}
UE Context Setup for ue022: Latency = 4.64 ms, Response: {'message': 'UE context created'}
Handover for ue022: Latency = 97.65 ms, Response: {'message': 'Handover process completed', 'duration': 0.09363102912902832}
UE Context Setup for ue040: Latency = 4.05 ms, Response: {'message': 'UE context created'}
Handover for ue040: Latency = 95.36 ms, Response: {'message': 'Handover process completed', 'duration': 0.09220194816589355}
UE Context Setup for ue056: Latency = 3.51 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 95.95 ms, Response: {'message': 'Handover process completed', 'duration': 0.0923311710357666}
UE Context Setup for ue085: Latency = 3.65 ms, Response: {'message': 'UE context created'}
Handover for ue085: Latency = 96.17 ms, Response: {'message': 'Handover process completed', 'duration': 0.0933370590209961}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.66 ms
  95th Percentile: 4.37 ms
  99th Percentile: 5.39 ms
Handover:
  Average Latency: 95.87 ms
  95th Percentile: 97.65 ms
  99th Percentile: 117.04 ms
UE Context Setup for ue030: Latency = 3.24 ms, Response: {'message': 'UE context created'}
Handover for ue030: Latency = 95.45 ms, Response: {'message': 'Handover process completed', 'duration': 0.09273719787597656}
UE Context Setup for ue032: Latency = 2.62 ms, Response: {'message': 'UE context created'}
Handover for ue032: Latency = 99.25 ms, Response: {'message': 'Handover process completed', 'duration': 0.09715056419372559}
UE Context Setup for ue044: Latency = 2.57 ms, Response: {'message': 'UE context created'}
Handover for ue044: Latency = 94.00 ms, Response: {'message': 'Handover process completed', 'duration': 0.0920419692993164}
UE Context Setup for ue037: Latency = 4.05 ms, Response: {'message': 'UE context created'}
Handover for ue037: Latency = 97.00 ms, Response: {'message': 'Handover process completed', 'duration': 0.09367799758911133}
UE Context Setup for ue050: Latency = 2.54 ms, Response: {'message': 'UE context created'}
Handover for ue050: Latency = 93.65 ms, Response: {'message': 'Handover process completed', 'duration': 0.09178566932678223}
UE Context Setup for ue025: Latency = 4.14 ms, Response: {'message': 'UE context created'}
Handover for ue025: Latency = 97.21 ms, Response: {'message': 'Handover process completed', 'duration': 0.09346127510070801}
UE Context Setup for ue094: Latency = 3.18 ms, Response: {'message': 'UE context created'}
Handover for ue094: Latency = 96.99 ms, Response: {'message': 'Handover process completed', 'duration': 0.09399080276489258}
UE Context Setup for ue087: Latency = 3.91 ms, Response: {'message': 'UE context created'}
Handover for ue087: Latency = 95.97 ms, Response: {'message': 'Handover process completed', 'duration': 0.09259033203125}
UE Context Setup for ue017: Latency = 4.14 ms, Response: {'message': 'UE context created'}
Handover for ue017: Latency = 96.12 ms, Response: {'message': 'Handover process completed', 'duration': 0.0926351547241211}
UE Context Setup for ue037: Latency = 3.99 ms, Response: {'message': 'UE context created'}
Handover for ue037: Latency = 96.51 ms, Response: {'message': 'Handover process completed', 'duration': 0.09317898750305176}
UE Context Setup for ue048: Latency = 4.15 ms, Response: {'message': 'UE context created'}
Handover for ue048: Latency = 96.63 ms, Response: {'message': 'Handover process completed', 'duration': 0.09279656410217285}
UE Context Setup for ue072: Latency = 4.23 ms, Response: {'message': 'UE context created'}
Handover for ue072: Latency = 95.04 ms, Response: {'message': 'Handover process completed', 'duration': 0.09208321571350098}
UE Context Setup for ue043: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue043: Latency = 95.41 ms, Response: {'message': 'Handover process completed', 'duration': 0.09227180480957031}
UE Context Setup for ue091: Latency = 3.81 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 95.32 ms, Response: {'message': 'Handover process completed', 'duration': 0.09220433235168457}
UE Context Setup for ue078: Latency = 2.78 ms, Response: {'message': 'UE context created'}
Handover for ue078: Latency = 96.03 ms, Response: {'message': 'Handover process completed', 'duration': 0.0934457778930664}
UE Context Setup for ue074: Latency = 3.33 ms, Response: {'message': 'UE context created'}
Handover for ue074: Latency = 93.43 ms, Response: {'message': 'Handover process completed', 'duration': 0.09134316444396973}
UE Context Setup for ue009: Latency = 3.68 ms, Response: {'message': 'UE context created'}
Handover for ue009: Latency = 95.23 ms, Response: {'message': 'Handover process completed', 'duration': 0.09210395812988281}
UE Context Setup for ue057: Latency = 4.07 ms, Response: {'message': 'UE context created'}
Handover for ue057: Latency = 96.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09275627136230469}
UE Context Setup for ue050: Latency = 3.94 ms, Response: {'message': 'UE context created'}
Handover for ue050: Latency = 96.13 ms, Response: {'message': 'Handover process completed', 'duration': 0.0927731990814209}
UE Context Setup for ue059: Latency = 5.34 ms, Response: {'message': 'UE context created'}
Handover for ue059: Latency = 96.54 ms, Response: {'message': 'Handover process completed', 'duration': 0.09310650825500488}
UE Context Setup for ue002: Latency = 2.69 ms, Response: {'message': 'UE context created'}
Handover for ue002: Latency = 95.13 ms, Response: {'message': 'Handover process completed', 'duration': 0.09218096733093262}
UE Context Setup for ue072: Latency = 3.12 ms, Response: {'message': 'UE context created'}
Handover for ue072: Latency = 94.27 ms, Response: {'message': 'Handover process completed', 'duration': 0.09143638610839844}
UE Context Setup for ue006: Latency = 2.60 ms, Response: {'message': 'UE context created'}
Handover for ue006: Latency = 93.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09173917770385742}
UE Context Setup for ue078: Latency = 3.78 ms, Response: {'message': 'UE context created'}
Handover for ue078: Latency = 95.87 ms, Response: {'message': 'Handover process completed', 'duration': 0.09266495704650879}
UE Context Setup for ue076: Latency = 2.22 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 92.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.09122681617736816}
UE Context Setup for ue004: Latency = 2.37 ms, Response: {'message': 'UE context created'}
Handover for ue004: Latency = 93.46 ms, Response: {'message': 'Handover process completed', 'duration': 0.09125590324401855}
UE Context Setup for ue099: Latency = 2.31 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 92.93 ms, Response: {'message': 'Handover process completed', 'duration': 0.0911703109741211}
UE Context Setup for ue073: Latency = 2.26 ms, Response: {'message': 'UE context created'}
Handover for ue073: Latency = 93.77 ms, Response: {'message': 'Handover process completed', 'duration': 0.0916590690612793}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.57 ms
  95th Percentile: 4.35 ms
  99th Percentile: 5.36 ms
Handover:
  Average Latency: 95.71 ms
  95th Percentile: 97.49 ms
  99th Percentile: 113.91 ms
UE Context Setup for ue037: Latency = 4.31 ms, Response: {'message': 'UE context created'}
Handover for ue037: Latency = 96.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09327983856201172}
UE Context Setup for ue033: Latency = 8.55 ms, Response: {'message': 'UE context created'}
Handover for ue033: Latency = 94.66 ms, Response: {'message': 'Handover process completed', 'duration': 0.09235858917236328}
UE Context Setup for ue069: Latency = 3.73 ms, Response: {'message': 'UE context created'}
Handover for ue069: Latency = 96.21 ms, Response: {'message': 'Handover process completed', 'duration': 0.09262228012084961}
UE Context Setup for ue076: Latency = 4.11 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 95.63 ms, Response: {'message': 'Handover process completed', 'duration': 0.09291791915893555}
UE Context Setup for ue016: Latency = 4.08 ms, Response: {'message': 'UE context created'}
Handover for ue016: Latency = 96.66 ms, Response: {'message': 'Handover process completed', 'duration': 0.09324002265930176}
UE Context Setup for ue062: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 96.37 ms, Response: {'message': 'Handover process completed', 'duration': 0.09285831451416016}
UE Context Setup for ue036: Latency = 4.51 ms, Response: {'message': 'UE context created'}
Handover for ue036: Latency = 96.42 ms, Response: {'message': 'Handover process completed', 'duration': 0.09369230270385742}
UE Context Setup for ue039: Latency = 3.90 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 96.70 ms, Response: {'message': 'Handover process completed', 'duration': 0.09350013732910156}
UE Context Setup for ue028: Latency = 4.28 ms, Response: {'message': 'UE context created'}
Handover for ue028: Latency = 97.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09411478042602539}
UE Context Setup for ue058: Latency = 4.10 ms, Response: {'message': 'UE context created'}
Handover for ue058: Latency = 97.89 ms, Response: {'message': 'Handover process completed', 'duration': 0.09424948692321777}
UE Context Setup for ue086: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue086: Latency = 96.29 ms, Response: {'message': 'Handover process completed', 'duration': 0.09296989440917969}
UE Context Setup for ue023: Latency = 2.98 ms, Response: {'message': 'UE context created'}
Handover for ue023: Latency = 94.24 ms, Response: {'message': 'Handover process completed', 'duration': 0.09183311462402344}
UE Context Setup for ue018: Latency = 3.78 ms, Response: {'message': 'UE context created'}
Handover for ue018: Latency = 95.08 ms, Response: {'message': 'Handover process completed', 'duration': 0.09235620498657227}
UE Context Setup for ue087: Latency = 3.26 ms, Response: {'message': 'UE context created'}
Handover for ue087: Latency = 95.39 ms, Response: {'message': 'Handover process completed', 'duration': 0.09288883209228516}
UE Context Setup for ue028: Latency = 3.92 ms, Response: {'message': 'UE context created'}
Handover for ue028: Latency = 96.07 ms, Response: {'message': 'Handover process completed', 'duration': 0.09292316436767578}
UE Context Setup for ue017: Latency = 3.54 ms, Response: {'message': 'UE context created'}
Handover for ue017: Latency = 95.25 ms, Response: {'message': 'Handover process completed', 'duration': 0.09192681312561035}
UE Context Setup for ue099: Latency = 2.89 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 94.58 ms, Response: {'message': 'Handover process completed', 'duration': 0.09212660789489746}
UE Context Setup for ue099: Latency = 3.74 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 96.07 ms, Response: {'message': 'Handover process completed', 'duration': 0.09279894828796387}
UE Context Setup for ue051: Latency = 3.91 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 96.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09297585487365723}
UE Context Setup for ue087: Latency = 4.53 ms, Response: {'message': 'UE context created'}
Handover for ue087: Latency = 95.45 ms, Response: {'message': 'Handover process completed', 'duration': 0.09242677688598633}
UE Context Setup for ue045: Latency = 3.13 ms, Response: {'message': 'UE context created'}
Handover for ue045: Latency = 95.48 ms, Response: {'message': 'Handover process completed', 'duration': 0.0928342342376709}
UE Context Setup for ue087: Latency = 3.97 ms, Response: {'message': 'UE context created'}
Handover for ue087: Latency = 96.72 ms, Response: {'message': 'Handover process completed', 'duration': 0.09332132339477539}
UE Context Setup for ue051: Latency = 4.09 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 96.13 ms, Response: {'message': 'Handover process completed', 'duration': 0.09292125701904297}
UE Context Setup for ue031: Latency = 4.02 ms, Response: {'message': 'UE context created'}
Handover for ue031: Latency = 96.75 ms, Response: {'message': 'Handover process completed', 'duration': 0.09326958656311035}
UE Context Setup for ue062: Latency = 3.81 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 97.23 ms, Response: {'message': 'Handover process completed', 'duration': 0.09353876113891602}
UE Context Setup for ue056: Latency = 4.24 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 96.21 ms, Response: {'message': 'Handover process completed', 'duration': 0.09275436401367188}
UE Context Setup for ue013: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue013: Latency = 95.95 ms, Response: {'message': 'Handover process completed', 'duration': 0.09273862838745117}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.66 ms
  95th Percentile: 4.50 ms
  99th Percentile: 8.52 ms
Handover:
  Average Latency: 95.65 ms
  95th Percentile: 97.57 ms
  99th Percentile: 101.34 ms
UE Context Setup for ue092: Latency = 2.66 ms, Response: {'message': 'UE context created'}
Handover for ue092: Latency = 98.36 ms, Response: {'message': 'Handover process completed', 'duration': 0.09578943252563477}
UE Context Setup for ue064: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue064: Latency = 95.79 ms, Response: {'message': 'Handover process completed', 'duration': 0.0926215648651123}
UE Context Setup for ue047: Latency = 2.58 ms, Response: {'message': 'UE context created'}
Handover for ue047: Latency = 93.82 ms, Response: {'message': 'Handover process completed', 'duration': 0.09172201156616211}
UE Context Setup for ue027: Latency = 2.52 ms, Response: {'message': 'UE context created'}
Handover for ue027: Latency = 93.72 ms, Response: {'message': 'Handover process completed', 'duration': 0.09181094169616699}
UE Context Setup for ue063: Latency = 2.41 ms, Response: {'message': 'UE context created'}
Handover for ue063: Latency = 93.64 ms, Response: {'message': 'Handover process completed', 'duration': 0.09161853790283203}
UE Context Setup for ue030: Latency = 5.38 ms, Response: {'message': 'UE context created'}
Handover for ue030: Latency = 101.14 ms, Response: {'message': 'Handover process completed', 'duration': 0.09720087051391602}
UE Context Setup for ue001: Latency = 4.57 ms, Response: {'message': 'UE context created'}
Handover for ue001: Latency = 95.78 ms, Response: {'message': 'Handover process completed', 'duration': 0.0923149585723877}
UE Context Setup for ue007: Latency = 3.11 ms, Response: {'message': 'UE context created'}
Handover for ue007: Latency = 94.31 ms, Response: {'message': 'Handover process completed', 'duration': 0.09183073043823242}
UE Context Setup for ue072: Latency = 3.65 ms, Response: {'message': 'UE context created'}
Handover for ue072: Latency = 96.45 ms, Response: {'message': 'Handover process completed', 'duration': 0.09276032447814941}
UE Context Setup for ue076: Latency = 3.01 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 94.87 ms, Response: {'message': 'Handover process completed', 'duration': 0.09178042411804199}
UE Context Setup for ue079: Latency = 3.62 ms, Response: {'message': 'UE context created'}
Handover for ue079: Latency = 96.10 ms, Response: {'message': 'Handover process completed', 'duration': 0.09283113479614258}
UE Context Setup for ue091: Latency = 3.17 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 95.86 ms, Response: {'message': 'Handover process completed', 'duration': 0.09274125099182129}
UE Context Setup for ue078: Latency = 4.25 ms, Response: {'message': 'UE context created'}
Handover for ue078: Latency = 96.56 ms, Response: {'message': 'Handover process completed', 'duration': 0.09306597709655762}
UE Context Setup for ue097: Latency = 2.55 ms, Response: {'message': 'UE context created'}
Handover for ue097: Latency = 93.17 ms, Response: {'message': 'Handover process completed', 'duration': 0.09132909774780273}
UE Context Setup for ue083: Latency = 4.04 ms, Response: {'message': 'UE context created'}
Handover for ue083: Latency = 95.06 ms, Response: {'message': 'Handover process completed', 'duration': 0.09229898452758789}
UE Context Setup for ue051: Latency = 3.66 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 94.97 ms, Response: {'message': 'Handover process completed', 'duration': 0.09180498123168945}
UE Context Setup for ue012: Latency = 3.63 ms, Response: {'message': 'UE context created'}
Handover for ue012: Latency = 96.35 ms, Response: {'message': 'Handover process completed', 'duration': 0.09296965599060059}
UE Context Setup for ue074: Latency = 4.59 ms, Response: {'message': 'UE context created'}
Handover for ue074: Latency = 95.30 ms, Response: {'message': 'Handover process completed', 'duration': 0.09214568138122559}
UE Context Setup for ue062: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 95.60 ms, Response: {'message': 'Handover process completed', 'duration': 0.09215784072875977}
UE Context Setup for ue058: Latency = 4.03 ms, Response: {'message': 'UE context created'}
Handover for ue058: Latency = 95.21 ms, Response: {'message': 'Handover process completed', 'duration': 0.09199953079223633}
UE Context Setup for ue049: Latency = 4.13 ms, Response: {'message': 'UE context created'}
Handover for ue049: Latency = 95.46 ms, Response: {'message': 'Handover process completed', 'duration': 0.09214591979980469}
UE Context Setup for ue092: Latency = 4.08 ms, Response: {'message': 'UE context created'}
Handover for ue092: Latency = 96.18 ms, Response: {'message': 'Handover process completed', 'duration': 0.0926661491394043}
UE Context Setup for ue066: Latency = 3.76 ms, Response: {'message': 'UE context created'}
Handover for ue066: Latency = 95.13 ms, Response: {'message': 'Handover process completed', 'duration': 0.091888427734375}
UE Context Setup for ue083: Latency = 3.70 ms, Response: {'message': 'UE context created'}
Handover for ue083: Latency = 95.00 ms, Response: {'message': 'Handover process completed', 'duration': 0.09185934066772461}
UE Context Setup for ue087: Latency = 3.22 ms, Response: {'message': 'UE context created'}
Handover for ue087: Latency = 94.79 ms, Response: {'message': 'Handover process completed', 'duration': 0.09230208396911621}
UE Context Setup for ue017: Latency = 4.03 ms, Response: {'message': 'UE context created'}
Handover for ue017: Latency = 94.20 ms, Response: {'message': 'Handover process completed', 'duration': 0.09202432632446289}
UE Context Setup for ue023: Latency = 3.55 ms, Response: {'message': 'UE context created'}
Handover for ue023: Latency = 95.89 ms, Response: {'message': 'Handover process completed', 'duration': 0.09276938438415527}
UE Context Setup for ue041: Latency = 4.02 ms, Response: {'message': 'UE context created'}
Handover for ue041: Latency = 95.87 ms, Response: {'message': 'Handover process completed', 'duration': 0.09253168106079102}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.67 ms
  95th Percentile: 4.59 ms
  99th Percentile: 8.52 ms
Handover:
  Average Latency: 95.73 ms
  95th Percentile: 97.88 ms
  99th Percentile: 101.36 ms
UE Context Setup for ue044: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue044: Latency = 96.08 ms, Response: {'message': 'Handover process completed', 'duration': 0.09249758720397949}
UE Context Setup for ue020: Latency = 3.77 ms, Response: {'message': 'UE context created'}
Handover for ue020: Latency = 95.28 ms, Response: {'message': 'Handover process completed', 'duration': 0.09208440780639648}
UE Context Setup for ue059: Latency = 3.91 ms, Response: {'message': 'UE context created'}
Handover for ue059: Latency = 96.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09281015396118164}
UE Context Setup for ue076: Latency = 2.58 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 93.75 ms, Response: {'message': 'Handover process completed', 'duration': 0.09167003631591797}
UE Context Setup for ue058: Latency = 3.76 ms, Response: {'message': 'UE context created'}
Handover for ue058: Latency = 95.66 ms, Response: {'message': 'Handover process completed', 'duration': 0.09198784828186035}
UE Context Setup for ue049: Latency = 3.91 ms, Response: {'message': 'UE context created'}
Handover for ue049: Latency = 95.57 ms, Response: {'message': 'Handover process completed', 'duration': 0.09229898452758789}
UE Context Setup for ue093: Latency = 3.18 ms, Response: {'message': 'UE context created'}
Handover for ue093: Latency = 94.83 ms, Response: {'message': 'Handover process completed', 'duration': 0.09228110313415527}
UE Context Setup for ue079: Latency = 3.82 ms, Response: {'message': 'UE context created'}
Handover for ue079: Latency = 97.52 ms, Response: {'message': 'Handover process completed', 'duration': 0.09338998794555664}
UE Context Setup for ue036: Latency = 3.93 ms, Response: {'message': 'UE context created'}
Handover for ue036: Latency = 95.12 ms, Response: {'message': 'Handover process completed', 'duration': 0.09189009666442871}
UE Context Setup for ue031: Latency = 3.75 ms, Response: {'message': 'UE context created'}
Handover for ue031: Latency = 94.91 ms, Response: {'message': 'Handover process completed', 'duration': 0.09200859069824219}
UE Context Setup for ue015: Latency = 2.46 ms, Response: {'message': 'UE context created'}
Handover for ue015: Latency = 93.58 ms, Response: {'message': 'Handover process completed', 'duration': 0.09174060821533203}
UE Context Setup for ue030: Latency = 4.31 ms, Response: {'message': 'UE context created'}
Handover for ue030: Latency = 96.61 ms, Response: {'message': 'Handover process completed', 'duration': 0.09278416633605957}
UE Context Setup for ue009: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue009: Latency = 96.01 ms, Response: {'message': 'Handover process completed', 'duration': 0.09267234802246094}
UE Context Setup for ue076: Latency = 3.53 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 94.67 ms, Response: {'message': 'Handover process completed', 'duration': 0.09198832511901855}
UE Context Setup for ue032: Latency = 3.87 ms, Response: {'message': 'UE context created'}
Handover for ue032: Latency = 95.67 ms, Response: {'message': 'Handover process completed', 'duration': 0.09238815307617188}
UE Context Setup for ue066: Latency = 3.84 ms, Response: {'message': 'UE context created'}
Handover for ue066: Latency = 96.15 ms, Response: {'message': 'Handover process completed', 'duration': 0.09283185005187988}
UE Context Setup for ue055: Latency = 4.09 ms, Response: {'message': 'UE context created'}
Handover for ue055: Latency = 96.28 ms, Response: {'message': 'Handover process completed', 'duration': 0.09275031089782715}
UE Context Setup for ue099: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 95.98 ms, Response: {'message': 'Handover process completed', 'duration': 0.09272909164428711}
UE Context Setup for ue081: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue081: Latency = 96.16 ms, Response: {'message': 'Handover process completed', 'duration': 0.09280824661254883}
UE Context Setup for ue034: Latency = 2.64 ms, Response: {'message': 'UE context created'}
Handover for ue034: Latency = 94.52 ms, Response: {'message': 'Handover process completed', 'duration': 0.09207749366760254}
UE Context Setup for ue021: Latency = 3.13 ms, Response: {'message': 'UE context created'}
Handover for ue021: Latency = 94.84 ms, Response: {'message': 'Handover process completed', 'duration': 0.09200668334960938}
UE Context Setup for ue080: Latency = 2.56 ms, Response: {'message': 'UE context created'}
Handover for ue080: Latency = 93.45 ms, Response: {'message': 'Handover process completed', 'duration': 0.09160661697387695}
UE Context Setup for ue029: Latency = 2.44 ms, Response: {'message': 'UE context created'}
Handover for ue029: Latency = 93.75 ms, Response: {'message': 'Handover process completed', 'duration': 0.09186339378356934}
UE Context Setup for ue038: Latency = 3.93 ms, Response: {'message': 'UE context created'}
Handover for ue038: Latency = 100.22 ms, Response: {'message': 'Handover process completed', 'duration': 0.09688806533813477}
UE Context Setup for ue038: Latency = 4.26 ms, Response: {'message': 'UE context created'}
Handover for ue038: Latency = 97.87 ms, Response: {'message': 'Handover process completed', 'duration': 0.09419107437133789}
UE Context Setup for ue018: Latency = 3.80 ms, Response: {'message': 'UE context created'}
Handover for ue018: Latency = 94.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09195137023925781}
UE Context Setup for ue086: Latency = 2.39 ms, Response: {'message': 'UE context created'}
Handover for ue086: Latency = 93.42 ms, Response: {'message': 'Handover process completed', 'duration': 0.09130096435546875}
UE Context Setup for ue068: Latency = 3.86 ms, Response: {'message': 'UE context created'}
Handover for ue068: Latency = 95.64 ms, Response: {'message': 'Handover process completed', 'duration': 0.09290003776550293}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.67 ms
  95th Percentile: 4.57 ms
  99th Percentile: 8.52 ms
Handover:
  Average Latency: 95.54 ms
  95th Percentile: 97.86 ms
  99th Percentile: 101.14 ms
UE Context Setup for ue047: Latency = 3.87 ms, Response: {'message': 'UE context created'}
Handover for ue047: Latency = 95.35 ms, Response: {'message': 'Handover process completed', 'duration': 0.09168529510498047}
UE Context Setup for ue079: Latency = 3.83 ms, Response: {'message': 'UE context created'}
Handover for ue079: Latency = 95.69 ms, Response: {'message': 'Handover process completed', 'duration': 0.09187102317810059}
UE Context Setup for ue092: Latency = 3.81 ms, Response: {'message': 'UE context created'}
Handover for ue092: Latency = 96.67 ms, Response: {'message': 'Handover process completed', 'duration': 0.09373188018798828}
UE Context Setup for ue042: Latency = 5.22 ms, Response: {'message': 'UE context created'}
Handover for ue042: Latency = 94.09 ms, Response: {'message': 'Handover process completed', 'duration': 0.0916898250579834}
UE Context Setup for ue062: Latency = 3.73 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 95.06 ms, Response: {'message': 'Handover process completed', 'duration': 0.09171080589294434}
UE Context Setup for ue049: Latency = 3.82 ms, Response: {'message': 'UE context created'}
Handover for ue049: Latency = 95.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.09226799011230469}
UE Context Setup for ue080: Latency = 3.84 ms, Response: {'message': 'UE context created'}
Handover for ue080: Latency = 95.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09204483032226562}
UE Context Setup for ue079: Latency = 2.44 ms, Response: {'message': 'UE context created'}
Handover for ue079: Latency = 93.80 ms, Response: {'message': 'Handover process completed', 'duration': 0.09165596961975098}
UE Context Setup for ue016: Latency = 3.12 ms, Response: {'message': 'UE context created'}
Handover for ue016: Latency = 94.05 ms, Response: {'message': 'Handover process completed', 'duration': 0.09189534187316895}
UE Context Setup for ue091: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 95.77 ms, Response: {'message': 'Handover process completed', 'duration': 0.09244012832641602}
UE Context Setup for ue028: Latency = 4.24 ms, Response: {'message': 'UE context created'}
Handover for ue028: Latency = 97.53 ms, Response: {'message': 'Handover process completed', 'duration': 0.09374880790710449}
UE Context Setup for ue062: Latency = 5.18 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 95.78 ms, Response: {'message': 'Handover process completed', 'duration': 0.09254980087280273}
UE Context Setup for ue082: Latency = 4.05 ms, Response: {'message': 'UE context created'}
Handover for ue082: Latency = 97.58 ms, Response: {'message': 'Handover process completed', 'duration': 0.09447717666625977}
UE Context Setup for ue052: Latency = 4.31 ms, Response: {'message': 'UE context created'}
Handover for ue052: Latency = 94.96 ms, Response: {'message': 'Handover process completed', 'duration': 0.09182095527648926}
UE Context Setup for ue065: Latency = 3.79 ms, Response: {'message': 'UE context created'}
Handover for ue065: Latency = 95.14 ms, Response: {'message': 'Handover process completed', 'duration': 0.09172701835632324}
UE Context Setup for ue006: Latency = 3.81 ms, Response: {'message': 'UE context created'}
Handover for ue006: Latency = 96.04 ms, Response: {'message': 'Handover process completed', 'duration': 0.09260225296020508}
UE Context Setup for ue091: Latency = 4.05 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 95.43 ms, Response: {'message': 'Handover process completed', 'duration': 0.09184861183166504}
UE Context Setup for ue017: Latency = 4.01 ms, Response: {'message': 'UE context created'}
Handover for ue017: Latency = 96.02 ms, Response: {'message': 'Handover process completed', 'duration': 0.0925896167755127}
UE Context Setup for ue044: Latency = 4.40 ms, Response: {'message': 'UE context created'}
Handover for ue044: Latency = 95.89 ms, Response: {'message': 'Handover process completed', 'duration': 0.09229588508605957}
UE Context Setup for ue023: Latency = 4.12 ms, Response: {'message': 'UE context created'}
Handover for ue023: Latency = 95.79 ms, Response: {'message': 'Handover process completed', 'duration': 0.09253501892089844}
UE Context Setup for ue010: Latency = 3.37 ms, Response: {'message': 'UE context created'}
Handover for ue010: Latency = 95.51 ms, Response: {'message': 'Handover process completed', 'duration': 0.0923464298248291}
UE Context Setup for ue005: Latency = 3.61 ms, Response: {'message': 'UE context created'}
Handover for ue005: Latency = 94.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09221720695495605}
UE Context Setup for ue042: Latency = 3.93 ms, Response: {'message': 'UE context created'}
Handover for ue042: Latency = 96.43 ms, Response: {'message': 'Handover process completed', 'duration': 0.09279680252075195}
UE Context Setup for ue044: Latency = 3.40 ms, Response: {'message': 'UE context created'}
Handover for ue044: Latency = 94.71 ms, Response: {'message': 'Handover process completed', 'duration': 0.09234380722045898}
UE Context Setup for ue082: Latency = 2.97 ms, Response: {'message': 'UE context created'}
Handover for ue082: Latency = 95.34 ms, Response: {'message': 'Handover process completed', 'duration': 0.09234929084777832}
UE Context Setup for ue002: Latency = 3.18 ms, Response: {'message': 'UE context created'}
Handover for ue002: Latency = 95.12 ms, Response: {'message': 'Handover process completed', 'duration': 0.0922994613647461}
UE Context Setup for ue037: Latency = 3.90 ms, Response: {'message': 'UE context created'}
Handover for ue037: Latency = 95.57 ms, Response: {'message': 'Handover process completed', 'duration': 0.09262251853942871}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.69 ms
  95th Percentile: 4.57 ms
  99th Percentile: 5.37 ms
Handover:
  Average Latency: 95.57 ms
  95th Percentile: 97.57 ms
  99th Percentile: 101.14 ms
UE Context Setup for ue053: Latency = 3.28 ms, Response: {'message': 'UE context created'}
Handover for ue053: Latency = 95.35 ms, Response: {'message': 'Handover process completed', 'duration': 0.09264540672302246}
UE Context Setup for ue077: Latency = 2.53 ms, Response: {'message': 'UE context created'}
Handover for ue077: Latency = 93.44 ms, Response: {'message': 'Handover process completed', 'duration': 0.09160280227661133}
UE Context Setup for ue082: Latency = 3.59 ms, Response: {'message': 'UE context created'}
Handover for ue082: Latency = 95.85 ms, Response: {'message': 'Handover process completed', 'duration': 0.09274697303771973}
UE Context Setup for ue077: Latency = 3.45 ms, Response: {'message': 'UE context created'}
Handover for ue077: Latency = 94.66 ms, Response: {'message': 'Handover process completed', 'duration': 0.0923306941986084}
UE Context Setup for ue002: Latency = 2.62 ms, Response: {'message': 'UE context created'}
Handover for ue002: Latency = 95.44 ms, Response: {'message': 'Handover process completed', 'duration': 0.09305453300476074}
UE Context Setup for ue049: Latency = 3.63 ms, Response: {'message': 'UE context created'}
Handover for ue049: Latency = 96.72 ms, Response: {'message': 'Handover process completed', 'duration': 0.0931692123413086}
UE Context Setup for ue051: Latency = 3.40 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 96.81 ms, Response: {'message': 'Handover process completed', 'duration': 0.09376096725463867}
UE Context Setup for ue040: Latency = 3.64 ms, Response: {'message': 'UE context created'}
Handover for ue040: Latency = 96.76 ms, Response: {'message': 'Handover process completed', 'duration': 0.09340214729309082}
UE Context Setup for ue019: Latency = 4.16 ms, Response: {'message': 'UE context created'}
Handover for ue019: Latency = 97.16 ms, Response: {'message': 'Handover process completed', 'duration': 0.0938863754272461}
UE Context Setup for ue002: Latency = 2.71 ms, Response: {'message': 'UE context created'}
Handover for ue002: Latency = 93.96 ms, Response: {'message': 'Handover process completed', 'duration': 0.09203124046325684}
UE Context Setup for ue014: Latency = 2.61 ms, Response: {'message': 'UE context created'}
Handover for ue014: Latency = 94.20 ms, Response: {'message': 'Handover process completed', 'duration': 0.09225201606750488}
UE Context Setup for ue030: Latency = 2.90 ms, Response: {'message': 'UE context created'}
Handover for ue030: Latency = 93.62 ms, Response: {'message': 'Handover process completed', 'duration': 0.09128570556640625}
UE Context Setup for ue050: Latency = 2.69 ms, Response: {'message': 'UE context created'}
Handover for ue050: Latency = 93.47 ms, Response: {'message': 'Handover process completed', 'duration': 0.0915069580078125}
UE Context Setup for ue001: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue001: Latency = 96.51 ms, Response: {'message': 'Handover process completed', 'duration': 0.09306955337524414}
UE Context Setup for ue001: Latency = 4.36 ms, Response: {'message': 'UE context created'}
Handover for ue001: Latency = 95.95 ms, Response: {'message': 'Handover process completed', 'duration': 0.09226202964782715}
UE Context Setup for ue050: Latency = 4.38 ms, Response: {'message': 'UE context created'}
Handover for ue050: Latency = 95.79 ms, Response: {'message': 'Handover process completed', 'duration': 0.09238648414611816}
UE Context Setup for ue067: Latency = 2.60 ms, Response: {'message': 'UE context created'}
Handover for ue067: Latency = 93.47 ms, Response: {'message': 'Handover process completed', 'duration': 0.091644287109375}
UE Context Setup for ue005: Latency = 3.21 ms, Response: {'message': 'UE context created'}
Handover for ue005: Latency = 94.58 ms, Response: {'message': 'Handover process completed', 'duration': 0.09196877479553223}
UE Context Setup for ue056: Latency = 3.35 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 95.22 ms, Response: {'message': 'Handover process completed', 'duration': 0.09261465072631836}
UE Context Setup for ue081: Latency = 3.91 ms, Response: {'message': 'UE context created'}
Handover for ue081: Latency = 95.76 ms, Response: {'message': 'Handover process completed', 'duration': 0.09279847145080566}
UE Context Setup for ue031: Latency = 3.10 ms, Response: {'message': 'UE context created'}
Handover for ue031: Latency = 95.58 ms, Response: {'message': 'Handover process completed', 'duration': 0.09267926216125488}
UE Context Setup for ue088: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue088: Latency = 96.39 ms, Response: {'message': 'Handover process completed', 'duration': 0.09287786483764648}
UE Context Setup for ue088: Latency = 3.34 ms, Response: {'message': 'UE context created'}
Handover for ue088: Latency = 95.07 ms, Response: {'message': 'Handover process completed', 'duration': 0.09239077568054199}
UE Context Setup for ue001: Latency = 3.21 ms, Response: {'message': 'UE context created'}
Handover for ue001: Latency = 94.27 ms, Response: {'message': 'Handover process completed', 'duration': 0.09174418449401855}
UE Context Setup for ue078: Latency = 3.09 ms, Response: {'message': 'UE context created'}
Handover for ue078: Latency = 94.22 ms, Response: {'message': 'Handover process completed', 'duration': 0.09152436256408691}
UE Context Setup for ue023: Latency = 3.59 ms, Response: {'message': 'UE context created'}
Handover for ue023: Latency = 95.30 ms, Response: {'message': 'Handover process completed', 'duration': 0.09217357635498047}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.61 ms
  95th Percentile: 4.38 ms
  99th Percentile: 5.22 ms
Handover:
  Average Latency: 95.40 ms
  95th Percentile: 97.50 ms
  99th Percentile: 100.19 ms
UE Context Setup for ue039: Latency = 3.75 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 95.32 ms, Response: {'message': 'Handover process completed', 'duration': 0.09244775772094727}
UE Context Setup for ue036: Latency = 3.01 ms, Response: {'message': 'UE context created'}
Handover for ue036: Latency = 94.14 ms, Response: {'message': 'Handover process completed', 'duration': 0.09196305274963379}
UE Context Setup for ue025: Latency = 3.77 ms, Response: {'message': 'UE context created'}
Handover for ue025: Latency = 98.56 ms, Response: {'message': 'Handover process completed', 'duration': 0.09515500068664551}
UE Context Setup for ue004: Latency = 2.97 ms, Response: {'message': 'UE context created'}
Handover for ue004: Latency = 95.47 ms, Response: {'message': 'Handover process completed', 'duration': 0.0930478572845459}
UE Context Setup for ue069: Latency = 3.58 ms, Response: {'message': 'UE context created'}
Handover for ue069: Latency = 95.42 ms, Response: {'message': 'Handover process completed', 'duration': 0.09219074249267578}
UE Context Setup for ue066: Latency = 3.70 ms, Response: {'message': 'UE context created'}
Handover for ue066: Latency = 95.06 ms, Response: {'message': 'Handover process completed', 'duration': 0.09238910675048828}
UE Context Setup for ue066: Latency = 3.88 ms, Response: {'message': 'UE context created'}
Handover for ue066: Latency = 96.60 ms, Response: {'message': 'Handover process completed', 'duration': 0.09350991249084473}
UE Context Setup for ue059: Latency = 3.77 ms, Response: {'message': 'UE context created'}
Handover for ue059: Latency = 96.42 ms, Response: {'message': 'Handover process completed', 'duration': 0.09331583976745605}
UE Context Setup for ue016: Latency = 3.28 ms, Response: {'message': 'UE context created'}
Handover for ue016: Latency = 94.48 ms, Response: {'message': 'Handover process completed', 'duration': 0.09139633178710938}
UE Context Setup for ue076: Latency = 3.97 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 95.31 ms, Response: {'message': 'Handover process completed', 'duration': 0.09212064743041992}
UE Context Setup for ue031: Latency = 4.00 ms, Response: {'message': 'UE context created'}
Handover for ue031: Latency = 95.43 ms, Response: {'message': 'Handover process completed', 'duration': 0.09239649772644043}
UE Context Setup for ue091: Latency = 3.27 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 95.34 ms, Response: {'message': 'Handover process completed', 'duration': 0.09262418746948242}
UE Context Setup for ue081: Latency = 3.79 ms, Response: {'message': 'UE context created'}
Handover for ue081: Latency = 95.11 ms, Response: {'message': 'Handover process completed', 'duration': 0.09212470054626465}
UE Context Setup for ue030: Latency = 3.55 ms, Response: {'message': 'UE context created'}
Handover for ue030: Latency = 96.39 ms, Response: {'message': 'Handover process completed', 'duration': 0.09310293197631836}
UE Context Setup for ue035: Latency = 4.94 ms, Response: {'message': 'UE context created'}
Handover for ue035: Latency = 95.99 ms, Response: {'message': 'Handover process completed', 'duration': 0.09209299087524414}
UE Context Setup for ue058: Latency = 3.61 ms, Response: {'message': 'UE context created'}
Handover for ue058: Latency = 98.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09409689903259277}
UE Context Setup for ue017: Latency = 4.24 ms, Response: {'message': 'UE context created'}
Handover for ue017: Latency = 95.03 ms, Response: {'message': 'Handover process completed', 'duration': 0.0917961597442627}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.61 ms
  95th Percentile: 4.38 ms
  99th Percentile: 5.22 ms
Handover:
  Average Latency: 95.49 ms
  95th Percentile: 97.57 ms
  99th Percentile: 100.20 ms
UE Context Setup for ue033: Latency = 3.88 ms, Response: {'message': 'UE context created'}
Handover for ue033: Latency = 95.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09202384948730469}
UE Context Setup for ue062: Latency = 4.07 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 97.62 ms, Response: {'message': 'Handover process completed', 'duration': 0.09348464012145996}
UE Context Setup for ue072: Latency = 2.56 ms, Response: {'message': 'UE context created'}
Handover for ue072: Latency = 98.05 ms, Response: {'message': 'Handover process completed', 'duration': 0.09400677680969238}
UE Context Setup for ue090: Latency = 4.31 ms, Response: {'message': 'UE context created'}
Handover for ue090: Latency = 95.93 ms, Response: {'message': 'Handover process completed', 'duration': 0.09217023849487305}
UE Context Setup for ue100: Latency = 4.55 ms, Response: {'message': 'UE context created'}
Handover for ue100: Latency = 97.95 ms, Response: {'message': 'Handover process completed', 'duration': 0.09358501434326172}
UE Context Setup for ue064: Latency = 4.30 ms, Response: {'message': 'UE context created'}
Handover for ue064: Latency = 94.90 ms, Response: {'message': 'Handover process completed', 'duration': 0.09214353561401367}
UE Context Setup for ue024: Latency = 3.78 ms, Response: {'message': 'UE context created'}
Handover for ue024: Latency = 96.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.0927588939666748}
UE Context Setup for ue063: Latency = 4.10 ms, Response: {'message': 'UE context created'}
Handover for ue063: Latency = 96.02 ms, Response: {'message': 'Handover process completed', 'duration': 0.09273195266723633}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.63 ms
  95th Percentile: 4.40 ms
  99th Percentile: 5.22 ms
Handover:
  Average Latency: 95.57 ms
  95th Percentile: 97.94 ms
  99th Percentile: 100.20 ms
UE Context Setup for ue032: Latency = 3.81 ms, Response: {'message': 'UE context created'}
Handover for ue032: Latency = 96.09 ms, Response: {'message': 'Handover process completed', 'duration': 0.09250736236572266}
UE Context Setup for ue057: Latency = 3.91 ms, Response: {'message': 'UE context created'}
Handover for ue057: Latency = 96.65 ms, Response: {'message': 'Handover process completed', 'duration': 0.09261131286621094}
UE Context Setup for ue062: Latency = 4.61 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 96.10 ms, Response: {'message': 'Handover process completed', 'duration': 0.09222602844238281}
UE Context Setup for ue019: Latency = 3.09 ms, Response: {'message': 'UE context created'}
Handover for ue019: Latency = 94.17 ms, Response: {'message': 'Handover process completed', 'duration': 0.0915982723236084}
UE Context Setup for ue041: Latency = 2.48 ms, Response: {'message': 'UE context created'}
Handover for ue041: Latency = 93.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.09159564971923828}
UE Context Setup for ue002: Latency = 3.96 ms, Response: {'message': 'UE context created'}
Handover for ue002: Latency = 96.82 ms, Response: {'message': 'Handover process completed', 'duration': 0.09288573265075684}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.63 ms
  95th Percentile: 4.54 ms
  99th Percentile: 5.22 ms
Handover:
  Average Latency: 95.58 ms
  95th Percentile: 97.94 ms
  99th Percentile: 100.20 ms
UE Context Setup for ue048: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue048: Latency = 96.36 ms, Response: {'message': 'Handover process completed', 'duration': 0.09296393394470215}
UE Context Setup for ue030: Latency = 2.59 ms, Response: {'message': 'UE context created'}
Handover for ue030: Latency = 93.47 ms, Response: {'message': 'Handover process completed', 'duration': 0.09157490730285645}
UE Context Setup for ue094: Latency = 2.46 ms, Response: {'message': 'UE context created'}
Handover for ue094: Latency = 93.91 ms, Response: {'message': 'Handover process completed', 'duration': 0.09188532829284668}
UE Context Setup for ue013: Latency = 2.75 ms, Response: {'message': 'UE context created'}
Handover for ue013: Latency = 94.05 ms, Response: {'message': 'Handover process completed', 'duration': 0.09179019927978516}
UE Context Setup for ue046: Latency = 2.61 ms, Response: {'message': 'UE context created'}
Handover for ue046: Latency = 93.82 ms, Response: {'message': 'Handover process completed', 'duration': 0.09175896644592285}
UE Context Setup for ue062: Latency = 2.71 ms, Response: {'message': 'UE context created'}
Handover for ue062: Latency = 94.48 ms, Response: {'message': 'Handover process completed', 'duration': 0.09199976921081543}
UE Context Setup for ue003: Latency = 3.91 ms, Response: {'message': 'UE context created'}
Handover for ue003: Latency = 96.26 ms, Response: {'message': 'Handover process completed', 'duration': 0.09292840957641602}
UE Context Setup for ue041: Latency = 3.96 ms, Response: {'message': 'UE context created'}
Handover for ue041: Latency = 95.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.092437744140625}
UE Context Setup for ue049: Latency = 2.72 ms, Response: {'message': 'UE context created'}
Handover for ue049: Latency = 95.27 ms, Response: {'message': 'Handover process completed', 'duration': 0.09238886833190918}
UE Context Setup for ue055: Latency = 3.17 ms, Response: {'message': 'UE context created'}
Handover for ue055: Latency = 95.41 ms, Response: {'message': 'Handover process completed', 'duration': 0.09233307838439941}
UE Context Setup for ue045: Latency = 4.10 ms, Response: {'message': 'UE context created'}
Handover for ue045: Latency = 100.85 ms, Response: {'message': 'Handover process completed', 'duration': 0.09617996215820312}
UE Context Setup for ue010: Latency = 4.12 ms, Response: {'message': 'UE context created'}
Handover for ue010: Latency = 96.67 ms, Response: {'message': 'Handover process completed', 'duration': 0.09345769882202148}
UE Context Setup for ue006: Latency = 4.22 ms, Response: {'message': 'UE context created'}
Handover for ue006: Latency = 101.24 ms, Response: {'message': 'Handover process completed', 'duration': 0.09784197807312012}
UE Context Setup for ue088: Latency = 2.60 ms, Response: {'message': 'UE context created'}
Handover for ue088: Latency = 93.91 ms, Response: {'message': 'Handover process completed', 'duration': 0.09174704551696777}
UE Context Setup for ue079: Latency = 3.97 ms, Response: {'message': 'UE context created'}
Handover for ue079: Latency = 96.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.09331893920898438}
UE Context Setup for ue024: Latency = 4.10 ms, Response: {'message': 'UE context created'}
Handover for ue024: Latency = 97.06 ms, Response: {'message': 'Handover process completed', 'duration': 0.09358358383178711}
UE Context Setup for ue037: Latency = 4.08 ms, Response: {'message': 'UE context created'}
Handover for ue037: Latency = 94.78 ms, Response: {'message': 'Handover process completed', 'duration': 0.0920553207397461}
UE Context Setup for ue083: Latency = 4.19 ms, Response: {'message': 'UE context created'}
Handover for ue083: Latency = 96.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.09325861930847168}
UE Context Setup for ue047: Latency = 4.17 ms, Response: {'message': 'UE context created'}
Handover for ue047: Latency = 96.93 ms, Response: {'message': 'Handover process completed', 'duration': 0.09354162216186523}
UE Context Setup for ue070: Latency = 3.31 ms, Response: {'message': 'UE context created'}
Handover for ue070: Latency = 94.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.09230399131774902}
UE Context Setup for ue091: Latency = 4.19 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 99.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.09651947021484375}
UE Context Setup for ue028: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue028: Latency = 96.11 ms, Response: {'message': 'Handover process completed', 'duration': 0.09287881851196289}
UE Context Setup for ue016: Latency = 4.05 ms, Response: {'message': 'UE context created'}
Handover for ue016: Latency = 96.62 ms, Response: {'message': 'Handover process completed', 'duration': 0.09329104423522949}
UE Context Setup for ue061: Latency = 3.78 ms, Response: {'message': 'UE context created'}
Handover for ue061: Latency = 95.22 ms, Response: {'message': 'Handover process completed', 'duration': 0.09278202056884766}
UE Context Setup for ue056: Latency = 3.98 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 95.84 ms, Response: {'message': 'Handover process completed', 'duration': 0.09290766716003418}
UE Context Setup for ue018: Latency = 3.90 ms, Response: {'message': 'UE context created'}
Handover for ue018: Latency = 95.83 ms, Response: {'message': 'Handover process completed', 'duration': 0.09288740158081055}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.63 ms
  95th Percentile: 4.40 ms
  99th Percentile: 5.18 ms
Handover:
  Average Latency: 95.75 ms
  95th Percentile: 98.53 ms
  99th Percentile: 101.23 ms
UE Context Setup for ue054: Latency = 3.23 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 95.54 ms, Response: {'message': 'Handover process completed', 'duration': 0.09277105331420898}
UE Context Setup for ue057: Latency = 2.47 ms, Response: {'message': 'UE context created'}
Handover for ue057: Latency = 93.79 ms, Response: {'message': 'Handover process completed', 'duration': 0.09198260307312012}
UE Context Setup for ue088: Latency = 4.02 ms, Response: {'message': 'UE context created'}
Handover for ue088: Latency = 95.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09241008758544922}
UE Context Setup for ue082: Latency = 3.59 ms, Response: {'message': 'UE context created'}
Handover for ue082: Latency = 94.77 ms, Response: {'message': 'Handover process completed', 'duration': 0.09169125556945801}
UE Context Setup for ue042: Latency = 3.18 ms, Response: {'message': 'UE context created'}
Handover for ue042: Latency = 94.82 ms, Response: {'message': 'Handover process completed', 'duration': 0.09208464622497559}
UE Context Setup for ue018: Latency = 3.93 ms, Response: {'message': 'UE context created'}
Handover for ue018: Latency = 96.34 ms, Response: {'message': 'Handover process completed', 'duration': 0.09292030334472656}
UE Context Setup for ue044: Latency = 3.73 ms, Response: {'message': 'UE context created'}
Handover for ue044: Latency = 95.56 ms, Response: {'message': 'Handover process completed', 'duration': 0.09232759475708008}
UE Context Setup for ue041: Latency = 4.04 ms, Response: {'message': 'UE context created'}
Handover for ue041: Latency = 96.25 ms, Response: {'message': 'Handover process completed', 'duration': 0.09295463562011719}
UE Context Setup for ue050: Latency = 4.61 ms, Response: {'message': 'UE context created'}
Handover for ue050: Latency = 95.02 ms, Response: {'message': 'Handover process completed', 'duration': 0.0917356014251709}
UE Context Setup for ue001: Latency = 3.88 ms, Response: {'message': 'UE context created'}
Handover for ue001: Latency = 95.99 ms, Response: {'message': 'Handover process completed', 'duration': 0.0927126407623291}
UE Context Setup for ue025: Latency = 3.81 ms, Response: {'message': 'UE context created'}
Handover for ue025: Latency = 94.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.09179449081420898}
UE Context Setup for ue063: Latency = 3.69 ms, Response: {'message': 'UE context created'}
Handover for ue063: Latency = 95.16 ms, Response: {'message': 'Handover process completed', 'duration': 0.09197354316711426}
UE Context Setup for ue028: Latency = 3.84 ms, Response: {'message': 'UE context created'}
Handover for ue028: Latency = 95.68 ms, Response: {'message': 'Handover process completed', 'duration': 0.09246325492858887}
UE Context Setup for ue065: Latency = 3.79 ms, Response: {'message': 'UE context created'}
Handover for ue065: Latency = 95.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.09247875213623047}
UE Context Setup for ue014: Latency = 2.70 ms, Response: {'message': 'UE context created'}
Handover for ue014: Latency = 94.17 ms, Response: {'message': 'Handover process completed', 'duration': 0.09206819534301758}
UE Context Setup for ue007: Latency = 4.23 ms, Response: {'message': 'UE context created'}
Handover for ue007: Latency = 94.69 ms, Response: {'message': 'Handover process completed', 'duration': 0.091796875}
UE Context Setup for ue017: Latency = 3.63 ms, Response: {'message': 'UE context created'}
Handover for ue017: Latency = 95.76 ms, Response: {'message': 'Handover process completed', 'duration': 0.09243535995483398}
UE Context Setup for ue037: Latency = 3.83 ms, Response: {'message': 'UE context created'}
Handover for ue037: Latency = 96.14 ms, Response: {'message': 'Handover process completed', 'duration': 0.09282326698303223}
UE Context Setup for ue099: Latency = 3.96 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 96.79 ms, Response: {'message': 'Handover process completed', 'duration': 0.09336662292480469}
UE Context Setup for ue011: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue011: Latency = 98.20 ms, Response: {'message': 'Handover process completed', 'duration': 0.09453678131103516}
UE Context Setup for ue076: Latency = 2.83 ms, Response: {'message': 'UE context created'}
Handover for ue076: Latency = 96.04 ms, Response: {'message': 'Handover process completed', 'duration': 0.09350061416625977}
UE Context Setup for ue071: Latency = 4.29 ms, Response: {'message': 'UE context created'}
Handover for ue071: Latency = 96.09 ms, Response: {'message': 'Handover process completed', 'duration': 0.09311127662658691}
UE Context Setup for ue089: Latency = 4.21 ms, Response: {'message': 'UE context created'}
Handover for ue089: Latency = 96.41 ms, Response: {'message': 'Handover process completed', 'duration': 0.0932619571685791}
UE Context Setup for ue067: Latency = 4.11 ms, Response: {'message': 'UE context created'}
Handover for ue067: Latency = 96.88 ms, Response: {'message': 'Handover process completed', 'duration': 0.0935823917388916}
UE Context Setup for ue012: Latency = 4.02 ms, Response: {'message': 'UE context created'}
Handover for ue012: Latency = 96.52 ms, Response: {'message': 'Handover process completed', 'duration': 0.09325051307678223}
UE Context Setup for ue020: Latency = 4.71 ms, Response: {'message': 'UE context created'}
Handover for ue020: Latency = 97.02 ms, Response: {'message': 'Handover process completed', 'duration': 0.09348702430725098}
UE Context Setup for ue047: Latency = 4.20 ms, Response: {'message': 'UE context created'}
Handover for ue047: Latency = 97.64 ms, Response: {'message': 'Handover process completed', 'duration': 0.09357166290283203}
UE Context Setup for ue051: Latency = 2.84 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 94.96 ms, Response: {'message': 'Handover process completed', 'duration': 0.09279441833496094}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.67 ms
  95th Percentile: 4.54 ms
  99th Percentile: 4.94 ms
Handover:
  Average Latency: 95.79 ms
  95th Percentile: 98.54 ms
  99th Percentile: 101.23 ms
UE Context Setup for ue050: Latency = 4.23 ms, Response: {'message': 'UE context created'}
Handover for ue050: Latency = 95.87 ms, Response: {'message': 'Handover process completed', 'duration': 0.09273958206176758}
UE Context Setup for ue022: Latency = 4.43 ms, Response: {'message': 'UE context created'}
Handover for ue022: Latency = 95.97 ms, Response: {'message': 'Handover process completed', 'duration': 0.09264993667602539}
UE Context Setup for ue078: Latency = 3.33 ms, Response: {'message': 'UE context created'}
Handover for ue078: Latency = 95.05 ms, Response: {'message': 'Handover process completed', 'duration': 0.09228515625}
UE Context Setup for ue073: Latency = 3.98 ms, Response: {'message': 'UE context created'}
Handover for ue073: Latency = 95.80 ms, Response: {'message': 'Handover process completed', 'duration': 0.09246110916137695}
UE Context Setup for ue020: Latency = 4.65 ms, Response: {'message': 'UE context created'}
Handover for ue020: Latency = 95.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.09234476089477539}
UE Context Setup for ue070: Latency = 3.25 ms, Response: {'message': 'UE context created'}
Handover for ue070: Latency = 96.25 ms, Response: {'message': 'Handover process completed', 'duration': 0.0929718017578125}
UE Context Setup for ue056: Latency = 4.47 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 96.62 ms, Response: {'message': 'Handover process completed', 'duration': 0.09356808662414551}
UE Context Setup for ue007: Latency = 4.03 ms, Response: {'message': 'UE context created'}
Handover for ue007: Latency = 95.80 ms, Response: {'message': 'Handover process completed', 'duration': 0.09256243705749512}
UE Context Setup for ue007: Latency = 4.55 ms, Response: {'message': 'UE context created'}
Handover for ue007: Latency = 96.09 ms, Response: {'message': 'Handover process completed', 'duration': 0.09278035163879395}
UE Context Setup for ue049: Latency = 3.69 ms, Response: {'message': 'UE context created'}
Handover for ue049: Latency = 95.02 ms, Response: {'message': 'Handover process completed', 'duration': 0.09187030792236328}
UE Context Setup for ue054: Latency = 3.62 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 95.05 ms, Response: {'message': 'Handover process completed', 'duration': 0.09203028678894043}
UE Context Setup for ue071: Latency = 3.80 ms, Response: {'message': 'UE context created'}
Handover for ue071: Latency = 95.94 ms, Response: {'message': 'Handover process completed', 'duration': 0.0925593376159668}
UE Context Setup for ue099: Latency = 3.20 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 95.44 ms, Response: {'message': 'Handover process completed', 'duration': 0.09242463111877441}
UE Context Setup for ue054: Latency = 4.05 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 97.54 ms, Response: {'message': 'Handover process completed', 'duration': 0.09400320053100586}
UE Context Setup for ue024: Latency = 3.99 ms, Response: {'message': 'UE context created'}
Handover for ue024: Latency = 96.75 ms, Response: {'message': 'Handover process completed', 'duration': 0.09343671798706055}
UE Context Setup for ue094: Latency = 3.61 ms, Response: {'message': 'UE context created'}
Handover for ue094: Latency = 98.16 ms, Response: {'message': 'Handover process completed', 'duration': 0.09445929527282715}
UE Context Setup for ue087: Latency = 4.51 ms, Response: {'message': 'UE context created'}
Handover for ue087: Latency = 97.16 ms, Response: {'message': 'Handover process completed', 'duration': 0.09325432777404785}
UE Context Setup for ue091: Latency = 2.60 ms, Response: {'message': 'UE context created'}
Handover for ue091: Latency = 94.05 ms, Response: {'message': 'Handover process completed', 'duration': 0.09206461906433105}
UE Context Setup for ue036: Latency = 4.14 ms, Response: {'message': 'UE context created'}
Handover for ue036: Latency = 97.76 ms, Response: {'message': 'Handover process completed', 'duration': 0.09423017501831055}
UE Context Setup for ue005: Latency = 4.16 ms, Response: {'message': 'UE context created'}
Handover for ue005: Latency = 96.23 ms, Response: {'message': 'Handover process completed', 'duration': 0.09253954887390137}
UE Context Setup for ue088: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue088: Latency = 96.48 ms, Response: {'message': 'Handover process completed', 'duration': 0.09273505210876465}
UE Context Setup for ue043: Latency = 4.19 ms, Response: {'message': 'UE context created'}
Handover for ue043: Latency = 96.66 ms, Response: {'message': 'Handover process completed', 'duration': 0.09323382377624512}
UE Context Setup for ue098: Latency = 4.16 ms, Response: {'message': 'UE context created'}
Handover for ue098: Latency = 96.43 ms, Response: {'message': 'Handover process completed', 'duration': 0.09304666519165039}
UE Context Setup for ue086: Latency = 4.22 ms, Response: {'message': 'UE context created'}
Handover for ue086: Latency = 96.63 ms, Response: {'message': 'Handover process completed', 'duration': 0.09322023391723633}
UE Context Setup for ue024: Latency = 3.32 ms, Response: {'message': 'UE context created'}
Handover for ue024: Latency = 95.70 ms, Response: {'message': 'Handover process completed', 'duration': 0.09300851821899414}
UE Context Setup for ue096: Latency = 4.02 ms, Response: {'message': 'UE context created'}
Handover for ue096: Latency = 96.31 ms, Response: {'message': 'Handover process completed', 'duration': 0.09306716918945312}
UE Context Setup for ue077: Latency = 4.07 ms, Response: {'message': 'UE context created'}
Handover for ue077: Latency = 96.32 ms, Response: {'message': 'Handover process completed', 'duration': 0.09327840805053711}
UE Context Setup for ue051: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 95.64 ms, Response: {'message': 'Handover process completed', 'duration': 0.09251999855041504}
UE Context Setup for ue099: Latency = 3.18 ms, Response: {'message': 'UE context created'}
Handover for ue099: Latency = 94.94 ms, Response: {'message': 'Handover process completed', 'duration': 0.09231948852539062}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.78 ms
  95th Percentile: 4.61 ms
  99th Percentile: 4.94 ms
Handover:
  Average Latency: 96.02 ms
  95th Percentile: 98.20 ms
  99th Percentile: 101.23 ms
UE Context Setup for ue097: Latency = 4.02 ms, Response: {'message': 'UE context created'}
Handover for ue097: Latency = 95.35 ms, Response: {'message': 'Handover process completed', 'duration': 0.09250760078430176}
UE Context Setup for ue098: Latency = 3.89 ms, Response: {'message': 'UE context created'}
Handover for ue098: Latency = 96.09 ms, Response: {'message': 'Handover process completed', 'duration': 0.09282827377319336}
UE Context Setup for ue079: Latency = 4.32 ms, Response: {'message': 'UE context created'}
Handover for ue079: Latency = 96.63 ms, Response: {'message': 'Handover process completed', 'duration': 0.0931248664855957}
UE Context Setup for ue029: Latency = 5.05 ms, Response: {'message': 'UE context created'}
Handover for ue029: Latency = 96.62 ms, Response: {'message': 'Handover process completed', 'duration': 0.09318923950195312}
UE Context Setup for ue044: Latency = 4.02 ms, Response: {'message': 'UE context created'}
Handover for ue044: Latency = 96.80 ms, Response: {'message': 'Handover process completed', 'duration': 0.09356164932250977}
UE Context Setup for ue013: Latency = 3.15 ms, Response: {'message': 'UE context created'}
Handover for ue013: Latency = 94.87 ms, Response: {'message': 'Handover process completed', 'duration': 0.09215068817138672}
UE Context Setup for ue042: Latency = 5.30 ms, Response: {'message': 'UE context created'}
Handover for ue042: Latency = 95.44 ms, Response: {'message': 'Handover process completed', 'duration': 0.09251952171325684}
UE Context Setup for ue054: Latency = 4.08 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 96.06 ms, Response: {'message': 'Handover process completed', 'duration': 0.09291315078735352}
UE Context Setup for ue011: Latency = 3.93 ms, Response: {'message': 'UE context created'}
Handover for ue011: Latency = 96.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.09325718879699707}
UE Context Setup for ue094: Latency = 3.53 ms, Response: {'message': 'UE context created'}
Handover for ue094: Latency = 94.85 ms, Response: {'message': 'Handover process completed', 'duration': 0.09195232391357422}
UE Context Setup for ue095: Latency = 4.03 ms, Response: {'message': 'UE context created'}
Handover for ue095: Latency = 96.57 ms, Response: {'message': 'Handover process completed', 'duration': 0.09308123588562012}
UE Context Setup for ue053: Latency = 3.18 ms, Response: {'message': 'UE context created'}
Handover for ue053: Latency = 96.03 ms, Response: {'message': 'Handover process completed', 'duration': 0.0932154655456543}
UE Context Setup for ue048: Latency = 4.01 ms, Response: {'message': 'UE context created'}
Handover for ue048: Latency = 96.61 ms, Response: {'message': 'Handover process completed', 'duration': 0.09333038330078125}
UE Context Setup for ue031: Latency = 4.04 ms, Response: {'message': 'UE context created'}
Handover for ue031: Latency = 97.46 ms, Response: {'message': 'Handover process completed', 'duration': 0.09427428245544434}
UE Context Setup for ue027: Latency = 3.35 ms, Response: {'message': 'UE context created'}
Handover for ue027: Latency = 95.03 ms, Response: {'message': 'Handover process completed', 'duration': 0.09232282638549805}
UE Context Setup for ue065: Latency = 3.64 ms, Response: {'message': 'UE context created'}
Handover for ue065: Latency = 94.85 ms, Response: {'message': 'Handover process completed', 'duration': 0.09183096885681152}
UE Context Setup for ue001: Latency = 3.49 ms, Response: {'message': 'UE context created'}
Handover for ue001: Latency = 96.28 ms, Response: {'message': 'Handover process completed', 'duration': 0.0933678150177002}
UE Context Setup for ue040: Latency = 4.00 ms, Response: {'message': 'UE context created'}
Handover for ue040: Latency = 95.59 ms, Response: {'message': 'Handover process completed', 'duration': 0.09279012680053711}
UE Context Setup for ue048: Latency = 2.62 ms, Response: {'message': 'UE context created'}
Handover for ue048: Latency = 93.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.09205079078674316}
UE Context Setup for ue021: Latency = 3.99 ms, Response: {'message': 'UE context created'}
Handover for ue021: Latency = 96.60 ms, Response: {'message': 'Handover process completed', 'duration': 0.09331035614013672}
UE Context Setup for ue057: Latency = 2.88 ms, Response: {'message': 'UE context created'}
Handover for ue057: Latency = 93.98 ms, Response: {'message': 'Handover process completed', 'duration': 0.09194064140319824}
UE Context Setup for ue039: Latency = 2.74 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 94.12 ms, Response: {'message': 'Handover process completed', 'duration': 0.09198212623596191}
UE Context Setup for ue006: Latency = 2.55 ms, Response: {'message': 'UE context created'}
Handover for ue006: Latency = 93.94 ms, Response: {'message': 'Handover process completed', 'duration': 0.09188318252563477}
UE Context Setup for ue021: Latency = 2.81 ms, Response: {'message': 'UE context created'}
Handover for ue021: Latency = 94.94 ms, Response: {'message': 'Handover process completed', 'duration': 0.0925900936126709}
UE Context Setup for ue068: Latency = 2.79 ms, Response: {'message': 'UE context created'}
Handover for ue068: Latency = 94.95 ms, Response: {'message': 'Handover process completed', 'duration': 0.09232783317565918}
UE Context Setup for ue066: Latency = 3.22 ms, Response: {'message': 'UE context created'}
Handover for ue066: Latency = 95.80 ms, Response: {'message': 'Handover process completed', 'duration': 0.0926668643951416}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.79 ms
  95th Percentile: 4.61 ms
  99th Percentile: 5.30 ms
Handover:
  Average Latency: 95.99 ms
  95th Percentile: 98.14 ms
  99th Percentile: 101.23 ms
UE Context Setup for ue056: Latency = 4.11 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 96.01 ms, Response: {'message': 'Handover process completed', 'duration': 0.09289669990539551}
UE Context Setup for ue007: Latency = 3.15 ms, Response: {'message': 'UE context created'}
Handover for ue007: Latency = 96.57 ms, Response: {'message': 'Handover process completed', 'duration': 0.09342551231384277}
UE Context Setup for ue060: Latency = 2.51 ms, Response: {'message': 'UE context created'}
Handover for ue060: Latency = 93.65 ms, Response: {'message': 'Handover process completed', 'duration': 0.09177732467651367}
UE Context Setup for ue011: Latency = 3.64 ms, Response: {'message': 'UE context created'}
Handover for ue011: Latency = 97.63 ms, Response: {'message': 'Handover process completed', 'duration': 0.093658447265625}
UE Context Setup for ue056: Latency = 3.97 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 95.08 ms, Response: {'message': 'Handover process completed', 'duration': 0.09252309799194336}
UE Context Setup for ue006: Latency = 4.09 ms, Response: {'message': 'UE context created'}
Handover for ue006: Latency = 97.03 ms, Response: {'message': 'Handover process completed', 'duration': 0.09327459335327148}
UE Context Setup for ue081: Latency = 2.60 ms, Response: {'message': 'UE context created'}
Handover for ue081: Latency = 97.27 ms, Response: {'message': 'Handover process completed', 'duration': 0.09343338012695312}
UE Context Setup for ue056: Latency = 3.94 ms, Response: {'message': 'UE context created'}
Handover for ue056: Latency = 96.86 ms, Response: {'message': 'Handover process completed', 'duration': 0.09312677383422852}
UE Context Setup for ue087: Latency = 4.01 ms, Response: {'message': 'UE context created'}
Handover for ue087: Latency = 96.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.09313082695007324}
UE Context Setup for ue073: Latency = 3.67 ms, Response: {'message': 'UE context created'}
Handover for ue073: Latency = 94.74 ms, Response: {'message': 'Handover process completed', 'duration': 0.09184837341308594}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.76 ms
  95th Percentile: 4.61 ms
  99th Percentile: 5.30 ms
Handover:
  Average Latency: 95.91 ms
  95th Percentile: 97.64 ms
  99th Percentile: 99.90 ms
UE Context Setup for ue058: Latency = 4.11 ms, Response: {'message': 'UE context created'}
Handover for ue058: Latency = 96.07 ms, Response: {'message': 'Handover process completed', 'duration': 0.09213590621948242}
UE Context Setup for ue019: Latency = 4.50 ms, Response: {'message': 'UE context created'}
Handover for ue019: Latency = 94.77 ms, Response: {'message': 'Handover process completed', 'duration': 0.0919041633605957}
UE Context Setup for ue064: Latency = 2.54 ms, Response: {'message': 'UE context created'}
Handover for ue064: Latency = 97.26 ms, Response: {'message': 'Handover process completed', 'duration': 0.09321451187133789}
UE Context Setup for ue055: Latency = 4.53 ms, Response: {'message': 'UE context created'}
Handover for ue055: Latency = 96.24 ms, Response: {'message': 'Handover process completed', 'duration': 0.09288239479064941}
UE Context Setup for ue055: Latency = 4.06 ms, Response: {'message': 'UE context created'}
Handover for ue055: Latency = 97.06 ms, Response: {'message': 'Handover process completed', 'duration': 0.09319233894348145}
UE Context Setup for ue025: Latency = 3.94 ms, Response: {'message': 'UE context created'}
Handover for ue025: Latency = 95.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.09289956092834473}
UE Context Setup for ue074: Latency = 4.68 ms, Response: {'message': 'UE context created'}
Handover for ue074: Latency = 97.78 ms, Response: {'message': 'Handover process completed', 'duration': 0.09380674362182617}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.77 ms
  95th Percentile: 4.64 ms
  99th Percentile: 5.30 ms
Handover:
  Average Latency: 95.92 ms
  95th Percentile: 97.64 ms
  99th Percentile: 98.20 ms
UE Context Setup for ue039: Latency = 3.88 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 96.11 ms, Response: {'message': 'Handover process completed', 'duration': 0.09233927726745605}
UE Context Setup for ue039: Latency = 4.13 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 97.06 ms, Response: {'message': 'Handover process completed', 'duration': 0.09302759170532227}
UE Context Setup for ue083: Latency = 4.28 ms, Response: {'message': 'UE context created'}
Handover for ue083: Latency = 97.02 ms, Response: {'message': 'Handover process completed', 'duration': 0.09310054779052734}
UE Context Setup for ue022: Latency = 4.30 ms, Response: {'message': 'UE context created'}
Handover for ue022: Latency = 97.33 ms, Response: {'message': 'Handover process completed', 'duration': 0.0938565731048584}
UE Context Setup for ue025: Latency = 3.97 ms, Response: {'message': 'UE context created'}
Handover for ue025: Latency = 96.13 ms, Response: {'message': 'Handover process completed', 'duration': 0.09299588203430176}
UE Context Setup for ue089: Latency = 3.22 ms, Response: {'message': 'UE context created'}
Handover for ue089: Latency = 95.39 ms, Response: {'message': 'Handover process completed', 'duration': 0.0927577018737793}
UE Context Setup for ue034: Latency = 4.14 ms, Response: {'message': 'UE context created'}
Handover for ue034: Latency = 95.71 ms, Response: {'message': 'Handover process completed', 'duration': 0.09262371063232422}
UE Context Setup for ue060: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue060: Latency = 95.52 ms, Response: {'message': 'Handover process completed', 'duration': 0.09261631965637207}
UE Context Setup for ue014: Latency = 3.78 ms, Response: {'message': 'UE context created'}
Handover for ue014: Latency = 94.69 ms, Response: {'message': 'Handover process completed', 'duration': 0.09195780754089355}
UE Context Setup for ue052: Latency = 2.50 ms, Response: {'message': 'UE context created'}
Handover for ue052: Latency = 94.12 ms, Response: {'message': 'Handover process completed', 'duration': 0.09214329719543457}
UE Context Setup for ue054: Latency = 4.12 ms, Response: {'message': 'UE context created'}
Handover for ue054: Latency = 96.37 ms, Response: {'message': 'Handover process completed', 'duration': 0.09298944473266602}
UE Context Setup for ue034: Latency = 3.96 ms, Response: {'message': 'UE context created'}
Handover for ue034: Latency = 96.38 ms, Response: {'message': 'Handover process completed', 'duration': 0.0930483341217041}
UE Context Setup for ue022: Latency = 3.83 ms, Response: {'message': 'UE context created'}
Handover for ue022: Latency = 96.48 ms, Response: {'message': 'Handover process completed', 'duration': 0.0929415225982666}
UE Context Setup for ue094: Latency = 4.10 ms, Response: {'message': 'UE context created'}
Handover for ue094: Latency = 98.23 ms, Response: {'message': 'Handover process completed', 'duration': 0.09501171112060547}
UE Context Setup for ue086: Latency = 4.27 ms, Response: {'message': 'UE context created'}
Handover for ue086: Latency = 96.68 ms, Response: {'message': 'Handover process completed', 'duration': 0.0934138298034668}
UE Context Setup for ue093: Latency = 2.61 ms, Response: {'message': 'UE context created'}
Handover for ue093: Latency = 94.01 ms, Response: {'message': 'Handover process completed', 'duration': 0.09212112426757812}
UE Context Setup for ue020: Latency = 2.63 ms, Response: {'message': 'UE context created'}
Handover for ue020: Latency = 94.99 ms, Response: {'message': 'Handover process completed', 'duration': 0.09268760681152344}
UE Context Setup for ue067: Latency = 4.19 ms, Response: {'message': 'UE context created'}
Handover for ue067: Latency = 96.49 ms, Response: {'message': 'Handover process completed', 'duration': 0.09312987327575684}
UE Context Setup for ue024: Latency = 3.23 ms, Response: {'message': 'UE context created'}
Handover for ue024: Latency = 95.69 ms, Response: {'message': 'Handover process completed', 'duration': 0.09318232536315918}
UE Context Setup for ue052: Latency = 2.48 ms, Response: {'message': 'UE context created'}
Handover for ue052: Latency = 93.58 ms, Response: {'message': 'Handover process completed', 'duration': 0.09170031547546387}
UE Context Setup for ue093: Latency = 3.38 ms, Response: {'message': 'UE context created'}
Handover for ue093: Latency = 95.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09227800369262695}
UE Context Setup for ue075: Latency = 3.24 ms, Response: {'message': 'UE context created'}
Handover for ue075: Latency = 95.03 ms, Response: {'message': 'Handover process completed', 'duration': 0.09219479560852051}
UE Context Setup for ue016: Latency = 3.17 ms, Response: {'message': 'UE context created'}
Handover for ue016: Latency = 95.96 ms, Response: {'message': 'Handover process completed', 'duration': 0.09312868118286133}
UE Context Setup for ue024: Latency = 3.83 ms, Response: {'message': 'UE context created'}
Handover for ue024: Latency = 96.90 ms, Response: {'message': 'Handover process completed', 'duration': 0.09334254264831543}
UE Context Setup for ue039: Latency = 3.83 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 96.53 ms, Response: {'message': 'Handover process completed', 'duration': 0.09303450584411621}
UE Context Setup for ue011: Latency = 3.80 ms, Response: {'message': 'UE context created'}
Handover for ue011: Latency = 96.65 ms, Response: {'message': 'Handover process completed', 'duration': 0.0933539867401123}
UE Context Setup for ue040: Latency = 4.62 ms, Response: {'message': 'UE context created'}
Handover for ue040: Latency = 94.36 ms, Response: {'message': 'Handover process completed', 'duration': 0.09172582626342773}
UE Context Setup for ue066: Latency = 3.59 ms, Response: {'message': 'UE context created'}
Handover for ue066: Latency = 96.14 ms, Response: {'message': 'Handover process completed', 'duration': 0.09291601181030273}
UE Context Setup for ue045: Latency = 3.44 ms, Response: {'message': 'UE context created'}
Handover for ue045: Latency = 96.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09301090240478516}
UE Context Setup for ue041: Latency = 4.29 ms, Response: {'message': 'UE context created'}
Handover for ue041: Latency = 96.72 ms, Response: {'message': 'Handover process completed', 'duration': 0.09334659576416016}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.74 ms
  95th Percentile: 4.61 ms
  99th Percentile: 5.30 ms
Handover:
  Average Latency: 95.96 ms
  95th Percentile: 97.63 ms
  99th Percentile: 98.23 ms
UE Context Setup for ue043: Latency = 4.11 ms, Response: {'message': 'UE context created'}
Handover for ue043: Latency = 94.90 ms, Response: {'message': 'Handover process completed', 'duration': 0.0921623706817627}
UE Context Setup for ue092: Latency = 2.63 ms, Response: {'message': 'UE context created'}
Handover for ue092: Latency = 93.98 ms, Response: {'message': 'Handover process completed', 'duration': 0.09204554557800293}
UE Context Setup for ue033: Latency = 2.93 ms, Response: {'message': 'UE context created'}
Handover for ue033: Latency = 94.07 ms, Response: {'message': 'Handover process completed', 'duration': 0.09212446212768555}
UE Context Setup for ue090: Latency = 2.56 ms, Response: {'message': 'UE context created'}
Handover for ue090: Latency = 94.79 ms, Response: {'message': 'Handover process completed', 'duration': 0.09246206283569336}
UE Context Setup for ue031: Latency = 3.32 ms, Response: {'message': 'UE context created'}
Handover for ue031: Latency = 95.31 ms, Response: {'message': 'Handover process completed', 'duration': 0.09214401245117188}
UE Context Setup for ue035: Latency = 4.28 ms, Response: {'message': 'UE context created'}
Handover for ue035: Latency = 95.81 ms, Response: {'message': 'Handover process completed', 'duration': 0.09241914749145508}
UE Context Setup for ue051: Latency = 4.19 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 95.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.09253191947937012}
UE Context Setup for ue084: Latency = 4.01 ms, Response: {'message': 'UE context created'}
Handover for ue084: Latency = 95.92 ms, Response: {'message': 'Handover process completed', 'duration': 0.09262347221374512}
UE Context Setup for ue036: Latency = 2.42 ms, Response: {'message': 'UE context created'}
Handover for ue036: Latency = 94.73 ms, Response: {'message': 'Handover process completed', 'duration': 0.09249448776245117}
UE Context Setup for ue069: Latency = 4.03 ms, Response: {'message': 'UE context created'}
Handover for ue069: Latency = 94.57 ms, Response: {'message': 'Handover process completed', 'duration': 0.09203934669494629}
UE Context Setup for ue061: Latency = 4.60 ms, Response: {'message': 'UE context created'}
Handover for ue061: Latency = 96.54 ms, Response: {'message': 'Handover process completed', 'duration': 0.09285902976989746}
UE Context Setup for ue013: Latency = 3.16 ms, Response: {'message': 'UE context created'}
Handover for ue013: Latency = 95.74 ms, Response: {'message': 'Handover process completed', 'duration': 0.0929412841796875}
UE Context Setup for ue078: Latency = 3.98 ms, Response: {'message': 'UE context created'}
Handover for ue078: Latency = 96.25 ms, Response: {'message': 'Handover process completed', 'duration': 0.09281373023986816}
UE Context Setup for ue051: Latency = 4.05 ms, Response: {'message': 'UE context created'}
Handover for ue051: Latency = 94.66 ms, Response: {'message': 'Handover process completed', 'duration': 0.09205079078674316}
UE Context Setup for ue011: Latency = 4.03 ms, Response: {'message': 'UE context created'}
Handover for ue011: Latency = 96.20 ms, Response: {'message': 'Handover process completed', 'duration': 0.09275126457214355}
UE Context Setup for ue048: Latency = 4.06 ms, Response: {'message': 'UE context created'}
Handover for ue048: Latency = 96.00 ms, Response: {'message': 'Handover process completed', 'duration': 0.09269452095031738}
UE Context Setup for ue004: Latency = 3.23 ms, Response: {'message': 'UE context created'}
Handover for ue004: Latency = 95.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09229564666748047}
UE Context Setup for ue095: Latency = 3.99 ms, Response: {'message': 'UE context created'}
Handover for ue095: Latency = 95.34 ms, Response: {'message': 'Handover process completed', 'duration': 0.09259843826293945}
UE Context Setup for ue098: Latency = 3.26 ms, Response: {'message': 'UE context created'}
Handover for ue098: Latency = 95.67 ms, Response: {'message': 'Handover process completed', 'duration': 0.09280920028686523}
UE Context Setup for ue030: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue030: Latency = 96.24 ms, Response: {'message': 'Handover process completed', 'duration': 0.09269142150878906}
UE Context Setup for ue075: Latency = 3.21 ms, Response: {'message': 'UE context created'}
Handover for ue075: Latency = 94.83 ms, Response: {'message': 'Handover process completed', 'duration': 0.09238171577453613}
UE Context Setup for ue058: Latency = 4.12 ms, Response: {'message': 'UE context created'}
Handover for ue058: Latency = 95.44 ms, Response: {'message': 'Handover process completed', 'duration': 0.09221076965332031}
UE Context Setup for ue039: Latency = 3.45 ms, Response: {'message': 'UE context created'}
Handover for ue039: Latency = 94.19 ms, Response: {'message': 'Handover process completed', 'duration': 0.09183597564697266}
UE Context Setup for ue044: Latency = 3.18 ms, Response: {'message': 'UE context created'}
Handover for ue044: Latency = 94.76 ms, Response: {'message': 'Handover process completed', 'duration': 0.09195685386657715}
UE Context Setup for ue059: Latency = 2.47 ms, Response: {'message': 'UE context created'}
Handover for ue059: Latency = 93.82 ms, Response: {'message': 'Handover process completed', 'duration': 0.09175825119018555}
UE Context Setup for ue016: Latency = 3.24 ms, Response: {'message': 'UE context created'}
Handover for ue016: Latency = 94.61 ms, Response: {'message': 'Handover process completed', 'duration': 0.0919489860534668}

Current Stats (Last 100 requests for each operation):
Setup:
  Average Latency: 3.65 ms
  95th Percentile: 4.60 ms
  99th Percentile: 5.30 ms
Handover:
  Average Latency: 95.71 ms
  95th Percentile: 97.33 ms
  99th Percentile: 98.23 ms
UE Context Setup for ue006: Latency = 3.95 ms, Response: {'message': 'UE context created'}
Handover for ue006: Latency = 96.23 ms, Response: {'message': 'Handover process completed', 'duration': 0.09317851066589355}
UE Context Setup for ue009: Latency = 2.48 ms, Response: {'message': 'UE context created'}
Handover for ue009: Latency = 93.75 ms, Response: {'message': 'Handover process completed', 'duration': 0.0918126106262207}
UE Context Setup for ue042: Latency = 4.00 ms, Response: {'message': 'UE context created'}
Handover for ue042: Latency = 95.51 ms, Response: {'message': 'Handover process completed', 'duration': 0.09297299385070801}
UE Context Setup for ue070: Latency = 3.85 ms, Response: {'message': 'UE context created'}
Handover for ue070: Latency = 96.09 ms, Response: {'message': 'Handover process completed', 'duration': 0.09285593032836914}
UE Context Setup for ue052: Latency = 3.83 ms, Response: {'message': 'UE context created'}
Handover for ue052: Latency = 96.37 ms, Response: {'message': 'Handover process completed', 'duration': 0.09328293800354004}
UE Context Setup for ue070: Latency = 3.76 ms, Response: {'message': 'UE context created'}
Handover for ue070: Latency = 96.12 ms, Response: {'message': 'Handover process completed', 'duration': 0.09290790557861328}
UE Context Setup for ue071: Latency = 3.36 ms, Response: {'message': 'UE context created'}
Handover for ue071: Latency = 94.34 ms, Response: {'message': 'Handover process completed', 'duration': 0.09225702285766602}
UE Context Setup for ue002: Latency = 2.67 ms, Response: {'message': 'UE context created'}
Handover for ue002: Latency = 93.44 ms, Response: {'message': 'Handover process completed', 'duration': 0.09162068367004395}

Final Test Stats:
Setup:
  Total Requests: 444
  Average Latency: 3.67 ms
  Median Latency: 3.83 ms
  95th Percentile: 4.53 ms
  99th Percentile: 5.27 ms
  Max Latency: 8.55 ms
Handover:
  Total Requests: 444
  Average Latency: 95.73 ms
  Median Latency: 95.69 ms
  95th Percentile: 97.64 ms
  99th Percentile: 101.01 ms
  Max Latency: 112.56 ms
(venv) [student@telcocloud core_network]$ 

```