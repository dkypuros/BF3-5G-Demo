# File location: 5G_Emulator_API/ptp/ptp.py
# File location: 5G_Emulator_API/ptp/ptp.py
import time

def synchronize_clock():
    print("Synchronizing clock using PTP")
    while True:
        print("PTP synchronization in progress...")
        time.sleep(60)

if __name__ == "__main__":
    synchronize_clock()