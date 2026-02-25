Turn off all python applications using ports

```bash
sudo lsof -i :8000
```

output

```bash
COMMAND   PID     USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python  24872 dkypuros    3u  IPv4 266565      0t0  TCP *:irdmi (LISTEN)
python  26601 dkypuros    3u  IPv4 266565      0t0  TCP *:irdmi (LISTEN)
```


```bash
pip install -r requirements.txt
```
Old Instructions
```bash
python test_cu_du.py
```

Newer Instructions

- test_5g_network.py

```bash
pip install -r requirements.txt
```

Run Test

```bash
python 5G_Emulator_API/main.py

python 5G_Emulator_API/test_5g_network.py
```

**Notes:**

- This test script simulates a UE connecting to the network and going through the following steps:

1. Checking gNodeB status
2. Sending an initial UE message
3. Sending a registration request
4. Responding to authentication
5. Completing the security mode procedure
6. Requesting a PDU session establishment

The script will print the responses from each step, allowing you to see how the different components of your 5G network are interacting.
Note: This test assumes that all your network functions are running on localhost with the ports specified in your original code. If you've changed any IP addresses or ports, make sure to update them in the test script.

Also, keep in mind that this is a simplified simulation and doesn't include all the complexities of a real 5G network. However, it should give you a good starting point for understanding the basic flow of 5G network operations and how the different components interact.
