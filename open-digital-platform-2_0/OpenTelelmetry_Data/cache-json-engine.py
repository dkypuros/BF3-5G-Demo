# File: enhanced_json_cache.py

import json
import os
import time
from threading import Lock

class EnhancedJSONCache:
    def __init__(self, cache_file, expiry_time=3600, max_size=1000):
        self.cache_file = cache_file
        self.expiry_time = expiry_time
        self.max_size = max_size
        self.cache = self._load_cache()
        self.lock = Lock()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get(self, key):
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                if time.time() - item['timestamp'] < self.expiry_time:
                    return item['data']
                else:
                    del self.cache[key]
        return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = {
                'data': value,
                'timestamp': time.time()
            }
            if len(self.cache) > self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
            self._save_cache()

    def clear_expired(self):
        with self.lock:
            current_time = time.time()
            expired_keys = [k for k, v in self.cache.items() if current_time - v['timestamp'] > self.expiry_time]
            for key in expired_keys:
                del self.cache[key]
            self._save_cache()

    def get_all(self):
        with self.lock:
            return {k: v['data'] for k, v in self.cache.items() if time.time() - v['timestamp'] < self.expiry_time}

    def remove(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self._save_cache()

# Usage example
if __name__ == "__main__":
    cache = EnhancedJSONCache('large_cache.json', expiry_time=300, max_size=10000)

    # Simulating large data insertion
    for i in range(1000):
        cache.set(f"key_{i}", f"value_{i}")

    # Retrieving some values
    print(cache.get("key_500"))
    print(cache.get("key_999"))

    # Get all valid entries
    all_entries = cache.get_all()
    print(f"Total entries: {len(all_entries)}")

    # Clear expired entries (none should be expired in this example)
    cache.clear_expired()

    # Remove a specific entry
    cache.remove("key_500")

    print("Cache operation completed. Check large_cache.json for results.")