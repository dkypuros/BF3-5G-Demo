import subprocess
import logging
import os
import psutil
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def kill_process_on_port(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                process = psutil.Process(conn.pid)
                process.terminate()
                logger.info(f"Terminated process on port {port}")
                return True
            except psutil.NoSuchProcess:
                pass
    return False

def start_nf(file_path, port):
    full_path = os.path.join(current_dir, file_path)
    if kill_process_on_port(port):
        time.sleep(1)  # Wait for the port to be released
    try:
        process = subprocess.Popen(["python", full_path, "--host", "0.0.0.0", "--port", str(port)])
        logger.info(f"Started {file_path} on port {port}")
        return process
    except Exception as e:
        logger.error(f"Failed to start {file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    processes = []

    # Start NRF
    processes.append(start_nf("core_network/nrf.py", 8000))

    # Start Network Functions
    processes.append(start_nf("core_network/amf.py", 9000))
    processes.append(start_nf("core_network/smf.py", 9001))
    processes.append(start_nf("core_network/upf.py", 9002))
    processes.append(start_nf("core_network/ausf.py", 9003))
    processes.append(start_nf("core_network/udm.py", 9004))
    processes.append(start_nf("core_network/udr.py", 9005))
    processes.append(start_nf("core_network/udsf.py", 9006))
    
    # Start CU and DU
    processes.append(start_nf("ran/cu/cu.py", 9008))
    processes.append(start_nf("ran/du/du.py", 9007))

    # Start RRU
    processes.append(start_nf("ran/rru/rru.py", 9009))

    # Start PTP
    processes.append(start_nf("ptp/ptp.py", 9010))

    # Start Service Assurance
    processes.append(start_nf("service_assurance/assurance_api.py", 9011))

    logger.info("All components started (including Service Assurance on port 9011)")

    # Wait for all processes to finish
    for process in processes:
        if process:
            process.wait()

    logger.info("All components have finished")