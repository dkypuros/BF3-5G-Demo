import time
import requests
from datetime import datetime, timedelta
import random
import statistics

class AMFLatencyTester:
    def __init__(self, base_url="http://localhost:9000", duration_minutes=10):
        self.base_url = base_url
        self.duration = timedelta(minutes=duration_minutes)
        self.latencies = {'setup': [], 'handover': []}
        self.start_time = datetime.now()
        self.end_time = self.start_time + self.duration
        self.ue_ids = [f"ue{i:03d}" for i in range(1, 101)]  # 100 UE IDs
        self.gnb_ids = ["gnb001", "gnb002"]
        self.anomaly_periods = []
        self.generate_anomaly_periods()

    def generate_anomaly_periods(self):
        # Create 2-3 anomaly periods
        num_anomalies = random.randint(2, 3)
        for _ in range(num_anomalies):
            start = self.start_time + timedelta(seconds=random.randint(60, int(self.duration.total_seconds() - 120)))
            end = start + timedelta(seconds=random.randint(30, 90))
            self.anomaly_periods.append((start, end))
        self.anomaly_periods.sort(key=lambda x: x[0])
        print("Anomaly periods:", [(ap[0].strftime("%M:%S"), ap[1].strftime("%M:%S")) for ap in self.anomaly_periods])

    def is_in_anomaly_period(self):
        now = datetime.now()
        return any(start <= now <= end for start, end in self.anomaly_periods)

    def introduce_anomaly(self):
        if self.is_in_anomaly_period():
            if random.random() < 0.7:  # 70% chance of high latency during anomaly
                time.sleep(random.uniform(1, 3))  # 1-3 second delay
            elif random.random() < 0.2:  # 20% chance of very high latency
                time.sleep(random.uniform(3, 5))  # 3-5 second delay

    def setup_ue_context(self, ue_id):
        gnb_id = random.choice(self.gnb_ids)
        payload = {"initial_gnb_id": gnb_id}
        
        self.introduce_anomaly()
        
        start_time = time.time()
        response = requests.post(f"{self.base_url}/amf/ue/{ue_id}", json=payload)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.latencies['setup'].append(latency)
        
        return response.json(), latency, gnb_id

    def trigger_handover(self, ue_id, source_gnb_id):
        target_gnb_id = [gnb for gnb in self.gnb_ids if gnb != source_gnb_id][0]
        payload = {"ue_id": ue_id, "source_gnb_id": source_gnb_id}
        
        self.introduce_anomaly()
        
        start_time = time.time()
        response = requests.post(f"{self.base_url}/amf/handover", json=payload)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        self.latencies['handover'].append(latency)
        
        return response.json(), latency

    def run_test(self):
        print("Starting AMF API latency test...")
        request_count = 0
        last_stats_time = self.start_time

        while datetime.now() < self.end_time:
            ue_id = random.choice(self.ue_ids)
            
            setup_result, setup_latency, initial_gnb = self.setup_ue_context(ue_id)
            print(f"UE Context Setup for {ue_id}: Latency = {setup_latency:.2f} ms, Response: {setup_result}")
            
            handover_result, handover_latency = self.trigger_handover(ue_id, initial_gnb)
            print(f"Handover for {ue_id}: Latency = {handover_latency:.2f} ms, Response: {handover_result}")
            
            request_count += 2
            
            if (datetime.now() - last_stats_time).total_seconds() >= 30:
                self.print_current_stats()
                last_stats_time = datetime.now()

            time.sleep(random.uniform(0.5, 1.5))

        self.print_final_stats()

    def print_current_stats(self):
        print("\nCurrent Stats (Last 100 requests for each operation):")
        for op in ['setup', 'handover']:
            recent_latencies = self.latencies[op][-100:]
            if recent_latencies:
                print(f"{op.capitalize()}:")
                print(f"  Average Latency: {statistics.mean(recent_latencies):.2f} ms")
                print(f"  95th Percentile: {statistics.quantiles(recent_latencies, n=20)[-1]:.2f} ms")
                print(f"  99th Percentile: {statistics.quantiles(recent_latencies, n=100)[-1]:.2f} ms")

    def print_final_stats(self):
        print("\nFinal Test Stats:")
        for op in ['setup', 'handover']:
            latencies = self.latencies[op]
            if latencies:
                print(f"{op.capitalize()}:")
                print(f"  Total Requests: {len(latencies)}")
                print(f"  Average Latency: {statistics.mean(latencies):.2f} ms")
                print(f"  Median Latency: {statistics.median(latencies):.2f} ms")
                print(f"  95th Percentile: {statistics.quantiles(latencies, n=20)[-1]:.2f} ms")
                print(f"  99th Percentile: {statistics.quantiles(latencies, n=100)[-1]:.2f} ms")
                print(f"  Max Latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    tester = AMFLatencyTester()
    tester.run_test()