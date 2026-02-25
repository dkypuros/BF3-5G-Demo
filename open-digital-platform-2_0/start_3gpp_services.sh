#!/bin/bash

# Start 3GPP-Enhanced 5G Core Network Services
# This script starts the enhanced AMF, SMF, and UPF with 3GPP compliance

echo "🚀 Starting 3GPP-Enhanced 5G Core Network Services"

# Check if Python virtual environment exists
if [ ! -d "5G_Emulator_API/venv" ]; then
    echo "Creating Python virtual environment..."
    cd 5G_Emulator_API
    python3 -m venv venv
    source venv/bin/activate
    pip install fastapi uvicorn requests opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi prometheus_client
    cd ..
fi

# Function to start a service in background
start_service() {
    local service_name=$1
    local service_file=$2
    local port=$3
    
    echo "Starting $service_name on port $port..."
    cd 5G_Emulator_API
    source venv/bin/activate
    python $service_file &
    local pid=$!
    echo "$service_name started with PID $pid"
    cd ..
    
    # Store PID for later cleanup
    echo $pid >> .service_pids
    
    # Wait a moment for service to start
    sleep 2
    
    # Check if service is responding
    if curl -s http://localhost:$port > /dev/null 2>&1; then
        echo "✅ $service_name is responding on port $port"
    else
        echo "⚠️  $service_name may not be fully ready yet on port $port"
    fi
}

# Clean up any existing PID file
rm -f .service_pids

echo "Starting services in dependency order..."

# Start NRF first (service discovery)
echo "📡 Starting Network Repository Function (NRF)..."
start_service "NRF" "core_network/nrf.py" 8000

# Start core network functions
echo "🏢 Starting Core Network Functions..."
start_service "AMF" "core_network/amf.py" 9000
start_service "SMF" "core_network/smf.py" 9001  
start_service "UPF" "core_network/upf.py" 9002

# Start other supporting services
echo "🔐 Starting Supporting Services..."
start_service "AUSF" "core_network/ausf.py" 9003
start_service "UDM" "core_network/udm.py" 9004

echo ""
echo "=== 3GPP-Enhanced 5G Core Network Status ==="
echo "✅ All services have been started!"
echo ""
echo "Service Endpoints:"
echo "  - NRF (Service Discovery): http://localhost:8000"
echo "  - AMF (Access & Mobility):  http://localhost:9000"
echo "  - SMF (Session Management): http://localhost:9001"  
echo "  - UPF (User Plane):         http://localhost:9002"
echo "  - AUSF (Authentication):    http://localhost:9003"
echo "  - UDM (Data Management):    http://localhost:9004"
echo ""
echo "3GPP Compliance Features:"
echo "  ✅ N11 Interface (AMF ↔ SMF): Nsmf_PDUSession service"
echo "  ✅ N4 Interface (SMF ↔ UPF):  PFCP protocol simulation"
echo "  ✅ 3GPP Message Validation:   TS 23.502 compliant payloads"
echo "  ✅ OpenTelemetry Tracing:     Procedure-level observability"
echo ""
echo "Frontend Dashboard: http://localhost:3000"
echo ""
echo "To test 3GPP compliance:"
echo "  python3 test_3gpp_compliance.py"
echo ""
echo "To stop all services:"
echo "  ./stop_services.sh"

# Create stop script
cat > stop_services.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping 5G Core Network Services..."

if [ -f .service_pids ]; then
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping process $pid"
            kill $pid
        fi
    done < .service_pids
    rm -f .service_pids
    echo "✅ All services stopped"
else
    echo "No service PIDs found"
fi
EOF

chmod +x stop_services.sh