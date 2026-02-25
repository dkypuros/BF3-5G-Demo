# 5G Mission Control Dashboard

A Next.js-based mission control dashboard for real-time monitoring and control of 5G Core Network simulation with NVIDIA BlueField-3 DPU integration.

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Live-brightgreen)
![Next.js](https://img.shields.io/badge/Next.js-15.4.6-black)
![React](https://img.shields.io/badge/React-19.1.0-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)

## 🚀 Quick Start

### Prerequisites

- **Node.js**: 18.0+ (LTS recommended)
- **npm**: 9.0+ or **yarn**: 1.22+
- **Backend**: 5G Core Network simulation with N6 firewall running
- **OS**: Fedora 39+ (or compatible Linux distribution)

### 1. Frontend Installation & Setup

```bash
# Navigate to the frontend directory
cd demo_front-end

# Install dependencies
npm install

# Start the development server
npm run dev
```

The dashboard will be available at: **http://localhost:3000**

### 2. Backend Integration Requirements

The frontend requires the following backend services to be running:

#### 2.1 Start the 5G Core Network Simulation

```bash
# Navigate to the main simulation directory
cd ../open-digital-platform-2_0/n6-interface-simulation

# Start core network functions
./scripts/start_core_network.sh

# Verify services are running
./scripts/health_check.sh
```

#### 2.2 Start the N6 Interface Firewall

```bash
# Start the BlueField-3 N6 firewall
sudo ./build/n6_firewall_simulation

# Or run the complete test suite
./scripts/test_with_devemu.sh
```

#### 2.3 Backend API Endpoints

The frontend expects these API endpoints to be available:

- **Core Network Status**: `http://localhost:8080/api/nf/status`
- **Simulation Control**: `http://localhost:8080/api/simulation/*`
- **Metrics & Analytics**: `http://localhost:8080/api/metrics/*`
- **Logs**: `http://localhost:8080/api/logs`
- **N6 Firewall**: `http://localhost:8081/api/firewall/status`

## 📊 Dashboard Features

### Network Topology View
- **Interactive 5G Network Functions**: AMF, SMF, UPF, NRF, AUSF, UDM, gNB components
- **Real-time Status Monitoring**: Online/offline status, load metrics, throughput data
- **BlueField-3 Firewall Integration**: N6 interface traffic filtering visualization
- **Control/User Plane Separation**: Toggle visibility of different network planes

### Simulation Control Panel
- **Scenario Management**: Handover procedures, registration tests, load testing
- **Device Management**: UE simulation, gNB configuration
- **Traffic Control**: Start/stop/pause simulation scenarios
- **Parameter Tuning**: Real-time adjustment of simulation parameters

### Analytics & Monitoring
- **Real-time Metrics**: Handover duration, session counts, throughput graphs
- **System Logs**: Filtered logging with search and export capabilities
- **OpenTelemetry Traces**: Procedure timeline analysis with performance insights
- **3GPP Compliance**: Specification adherence validation

### 3GPP Compliance Inspector
- **Call Flow Validation**: Step-by-step procedure verification against 3GPP specs
- **Message Payload Analysis**: IE (Information Element) comparison with standard
- **Compliance Scoring**: Real-time adherence percentage and warnings

## 🛠️ Development

### Available Scripts

```bash
# Development server with hot reload
npm run dev

# Production build
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

### Project Structure

```
demo_front-end/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx          # Main dashboard page
│   ├── components/
│   │   ├── dashboard/         # Main dashboard components
│   │   │   └── MissionControlDashboard.tsx
│   │   ├── network/           # Network topology
│   │   │   └── NetworkTopologyView.tsx
│   │   ├── simulation/        # Simulation controls
│   │   │   └── SimulationControlPanel.tsx
│   │   ├── analytics/         # Analytics and monitoring
│   │   │   └── AnalyticsView.tsx
│   │   ├── compliance/        # 3GPP compliance tools
│   │   │   └── ComplianceInspector.tsx
│   │   └── ui/               # Shadcn/ui components
│   └── lib/
│       └── utils.ts          # Utility functions
├── package.json
└── README.md
```

### Technology Stack

- **Framework**: Next.js 15.4.6 with App Router
- **UI Library**: Shadcn/ui + Radix UI primitives
- **Styling**: Tailwind CSS 4.0
- **Charts**: Recharts for data visualization
- **Flow Diagrams**: React Flow for network topology
- **Icons**: Lucide React
- **State Management**: React hooks (useState, useEffect)

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the frontend directory:

```bash
# Backend API endpoints
NEXT_PUBLIC_CORE_API_URL=http://localhost:8080
NEXT_PUBLIC_FIREWALL_API_URL=http://localhost:8081

# Simulation settings
NEXT_PUBLIC_REFRESH_INTERVAL=2000
NEXT_PUBLIC_MAX_LOG_ENTRIES=1000
```

### Backend Configuration

Ensure your backend services are configured to accept requests from the frontend:

```bash
# In your backend configuration
CORS_ORIGINS=http://localhost:3000
API_PORT=8080
FIREWALL_API_PORT=8081
```

## 🧪 Testing the Complete System

### 1. Start All Services

```bash
# Terminal 1: Start 5G Core Network
cd ../open-digital-platform-2_0/n6-interface-simulation
./scripts/start_core_network.sh

# Terminal 2: Start N6 Firewall
sudo ./build/n6_firewall_simulation

# Terminal 3: Start Frontend Dashboard
cd demo_front-end
npm run dev
```

### 2. Verify Integration

1. **Open Dashboard**: Navigate to http://localhost:3000
2. **Check Network Status**: All NF cards should show "Online" status
3. **Run Simulation**: Start a handover scenario from the control panel
4. **Monitor Analytics**: Observe real-time metrics and logs
5. **Test Compliance**: Use the 3GPP inspector on any procedure

### 3. Expected Behavior

- **Network Topology**: Interactive NF cards with live status updates
- **Real-time Data**: Metrics update every 2 seconds during simulation
- **Log Streaming**: New log entries appear automatically
- **Firewall Integration**: N6 traffic filtering shown in throughput charts

## 🐛 Troubleshooting

### Frontend Issues

**Port 3000 already in use:**
```bash
# Kill existing process
npx kill-port 3000
# Or use different port
npm run dev -- -p 3001
```

**Module not found errors:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Backend Integration Issues

**API connection failed:**
1. Verify backend services are running: `netstat -tulpn | grep :8080`
2. Check CORS configuration in backend
3. Verify firewall settings allow local connections

**No real-time data:**
1. Ensure simulation is actually running
2. Check API endpoints return valid JSON
3. Verify WebSocket connections (if used)

**N6 Firewall not showing:**
1. Confirm BlueField-3 services are active
2. Check firewall API is accessible
3. Verify DOCA DevEmu is properly initialized

## 📈 Performance Optimization

### Production Deployment

```bash
# Build optimized production bundle
npm run build

# Start production server
npm start
```

### Monitoring & Observability

The dashboard includes built-in performance monitoring:
- Component render times
- API response latencies  
- Real-time data update frequencies
- Memory usage tracking

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for all new components
3. Add proper error handling for API calls
4. Test with both mock and live backend data
5. Update this README for any new features

## 📄 License

This project is part of the BF3-5G-Demo repository. See the main repository LICENSE file for details.

---

**Next Steps**: Once both frontend and backend are running, you should see a fully functional 5G mission control dashboard with real-time network monitoring, simulation controls, and 3GPP compliance validation.
