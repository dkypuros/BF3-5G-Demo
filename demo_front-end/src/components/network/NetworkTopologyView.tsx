'use client'

import { useCallback, useState, useEffect } from 'react'
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Node,
  Edge,
  Connection,
  BackgroundVariant,
} from 'reactflow'
import 'reactflow/dist/style.css'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Progress } from '@/components/ui/progress'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { 
  Server, 
  Radio, 
  Shield, 
  Activity, 
  Wifi, 
  Database,
  Smartphone,
  Globe
} from 'lucide-react'

interface NetworkTopologyViewProps {
  selectedNF: string | null
  onNFSelect: (nf: string | null) => void
  simulationStatus: 'stopped' | 'running' | 'paused'
}

// Custom Node Component for Network Functions
function NFCard({ data }: { data: any }) {
  const getIcon = (type: string) => {
    switch (type) {
      case 'AMF': case 'SMF': case 'NRF': case 'AUSF': return <Server className="w-6 h-6" />
      case 'UPF': return <Shield className="w-6 h-6" />
      case 'UDM': case 'UDR': case 'UDSF': return <Database className="w-6 h-6" />
      case 'gNB-CU': case 'gNB-DU': return <Radio className="w-6 h-6" />
      case 'gNB-RRU': return <Wifi className="w-6 h-6" />
      case 'UE': return <Smartphone className="w-6 h-6" />
      case 'DN': return <Globe className="w-6 h-6" />
      default: return <Server className="w-6 h-6" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500'
      case 'degraded': return 'bg-yellow-500'
      case 'offline': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <Card className={`w-48 transition-all duration-200 hover:shadow-lg ${
      data.selected ? 'ring-2 ring-primary' : ''
    }`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Avatar className="w-8 h-8">
              <AvatarFallback className="text-xs font-bold">
                {data.label}
              </AvatarFallback>
            </Avatar>
            <div>
              <CardTitle className="text-sm">{data.label}</CardTitle>
              <CardDescription className="text-xs">{data.description}</CardDescription>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            {getIcon(data.label)}
            <div className={`w-3 h-3 rounded-full ${getStatusColor(data.status)}`} />
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span>Load</span>
            <span>{data.load}%</span>
          </div>
          <Progress value={data.load} className="h-2" />
          
          {data.sessions && (
            <div className="flex justify-between text-xs">
              <span>Active Sessions</span>
              <span>{data.sessions}</span>
            </div>
          )}
          
          {data.throughput && (
            <div className="flex justify-between text-xs">
              <span>Throughput</span>
              <span>{data.throughput}</span>
            </div>
          )}

          <Badge 
            variant={data.status === 'online' ? 'default' : 
                    data.status === 'degraded' ? 'secondary' : 'destructive'}
            className="w-full justify-center text-xs"
          >
            {data.status === 'online' && <Activity className="w-3 h-3 mr-1" />}
            {data.status.charAt(0).toUpperCase() + data.status.slice(1)}
          </Badge>
        </div>
      </CardContent>
    </Card>
  )
}

const nodeTypes = {
  nfCard: NFCard,
}

export function NetworkTopologyView({ selectedNF, onNFSelect, simulationStatus }: NetworkTopologyViewProps) {
  const [showControlPlane, setShowControlPlane] = useState(true)
  const [showUserPlane, setShowUserPlane] = useState(true)
  const [showActiveSessions, setShowActiveSessions] = useState(true)

  // Initial network topology
  const initialNodes: Node[] = [
    {
      id: 'nrf',
      type: 'nfCard',
      position: { x: 400, y: 50 },
      data: { 
        label: 'NRF', 
        description: 'Network Repository Function',
        status: 'online', 
        load: 12,
        selected: selectedNF === 'NRF'
      },
    },
    {
      id: 'amf',
      type: 'nfCard',
      position: { x: 200, y: 200 },
      data: { 
        label: 'AMF', 
        description: 'Access & Mobility Mgmt',
        status: 'online', 
        load: 45,
        sessions: 23,
        selected: selectedNF === 'AMF'
      },
    },
    {
      id: 'smf',
      type: 'nfCard',
      position: { x: 600, y: 200 },
      data: { 
        label: 'SMF', 
        description: 'Session Management',
        status: 'online', 
        load: 32,
        sessions: 18,
        selected: selectedNF === 'SMF'
      },
    },
    {
      id: 'upf',
      type: 'nfCard',
      position: { x: 600, y: 400 },
      data: { 
        label: 'UPF', 
        description: 'User Plane Function',
        status: 'online', 
        load: 78,
        throughput: '156 Gbps',
        selected: selectedNF === 'UPF'
      },
    },
    {
      id: 'ausf',
      type: 'nfCard',
      position: { x: 50, y: 350 },
      data: { 
        label: 'AUSF', 
        description: 'Authentication Server',
        status: 'online', 
        load: 28,
        selected: selectedNF === 'AUSF'
      },
    },
    {
      id: 'udm',
      type: 'nfCard',
      position: { x: 200, y: 500 },
      data: { 
        label: 'UDM', 
        description: 'Unified Data Mgmt',
        status: 'online', 
        load: 34,
        selected: selectedNF === 'UDM'
      },
    },
    {
      id: 'gnb-cu',
      type: 'nfCard',
      position: { x: 50, y: 200 },
      data: { 
        label: 'gNB-CU', 
        description: 'Central Unit',
        status: 'online', 
        load: 56,
        selected: selectedNF === 'gNB-CU'
      },
    },
    {
      id: 'gnb-du',
      type: 'nfCard',
      position: { x: 50, y: 50 },
      data: { 
        label: 'gNB-DU', 
        description: 'Distributed Unit',
        status: 'online', 
        load: 67,
        selected: selectedNF === 'gNB-DU'
      },
    },
    {
      id: 'bf3-firewall',
      type: 'nfCard',
      position: { x: 800, y: 400 },
      data: { 
        label: 'BF3-FW', 
        description: 'BlueField-3 Firewall',
        status: 'online', 
        load: 45,
        throughput: '400 Gbps',
        selected: selectedNF === 'BF3-FW'
      },
    },
    {
      id: 'data-network',
      type: 'nfCard',
      position: { x: 1000, y: 400 },
      data: { 
        label: 'DN', 
        description: 'Data Network',
        status: 'online', 
        load: 23,
        selected: selectedNF === 'DN'
      },
    },
  ]

  const initialEdges: Edge[] = [
    // Control Plane (N1, N2, N11, etc.)
    { id: 'e1', source: 'gnb-cu', target: 'amf', label: 'N2', type: 'default', className: 'control-plane' },
    { id: 'e2', source: 'amf', target: 'smf', label: 'N11', type: 'default', className: 'control-plane' },
    { id: 'e3', source: 'amf', target: 'ausf', label: 'N12', type: 'default', className: 'control-plane' },
    { id: 'e4', source: 'amf', target: 'udm', label: 'N8', type: 'default', className: 'control-plane' },
    { id: 'e5', source: 'smf', target: 'upf', label: 'N4', type: 'default', className: 'control-plane' },
    { id: 'e6', source: 'amf', target: 'nrf', label: 'Nnrf', type: 'default', className: 'control-plane' },
    { id: 'e7', source: 'smf', target: 'nrf', label: 'Nnrf', type: 'default', className: 'control-plane' },
    
    // User Plane (N3, N6, N9)
    { id: 'e8', source: 'gnb-cu', target: 'upf', label: 'N3', type: 'default', className: 'user-plane' },
    { id: 'e9', source: 'upf', target: 'bf3-firewall', label: 'N6', type: 'default', className: 'user-plane' },
    { id: 'e10', source: 'bf3-firewall', target: 'data-network', label: 'Filtered', type: 'default', className: 'user-plane' },
    
    // RAN Internal
    { id: 'e11', source: 'gnb-du', target: 'gnb-cu', label: 'F1', type: 'default', className: 'control-plane' },
  ]

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onConnect = useCallback(
    (params: Edge | Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  )

  // Update node selection when selectedNF changes
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: {
          ...node.data,
          selected: node.data.label === selectedNF,
        },
      }))
    )
  }, [selectedNF, setNodes])

  // Simulate traffic flow animation
  useEffect(() => {
    if (simulationStatus === 'running') {
      const interval = setInterval(() => {
        // Update mock data to simulate activity
        setNodes((nds) =>
          nds.map((node) => ({
            ...node,
            data: {
              ...node.data,
              load: Math.max(10, Math.min(95, node.data.load + (Math.random() - 0.5) * 10)),
            },
          }))
        )
      }, 2000)

      return () => clearInterval(interval)
    }
  }, [simulationStatus, setNodes])

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    onNFSelect(node.data.label === selectedNF ? null : node.data.label)
  }, [onNFSelect, selectedNF])

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Network Topology
            </CardTitle>
            <CardDescription>Live 5G Core & RAN Visualization</CardDescription>
          </div>
          
          {/* Topology View Options */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Switch
                id="control-plane"
                checked={showControlPlane}
                onCheckedChange={setShowControlPlane}
              />
              <Label htmlFor="control-plane" className="text-sm">Control Plane</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="user-plane"
                checked={showUserPlane}
                onCheckedChange={setShowUserPlane}
              />
              <Label htmlFor="user-plane" className="text-sm">User Plane</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="active-sessions"
                checked={showActiveSessions}
                onCheckedChange={setShowActiveSessions}
              />
              <Label htmlFor="active-sessions" className="text-sm">Active Sessions</Label>
            </div>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0 h-[calc(100%-100px)]">
        <div className="h-full w-full">
          <ReactFlow
            nodes={nodes}
            edges={edges.filter(edge => {
              if (!showControlPlane && edge.className === 'control-plane') return false
              if (!showUserPlane && edge.className === 'user-plane') return false
              return true
            })}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
            attributionPosition="bottom-left"
            className="bg-muted/30"
          >
            <Background 
              variant={BackgroundVariant.Dots} 
              gap={20} 
              size={1} 
              className="opacity-30"
            />
            <MiniMap 
              className="bg-background border rounded-lg" 
              nodeColor={(node) => {
                switch (node.data.status) {
                  case 'online': return '#22c55e'
                  case 'degraded': return '#eab308'
                  case 'offline': return '#ef4444'
                  default: return '#6b7280'
                }
              }}
            />
            <Controls className="bg-background border rounded-lg" />
          </ReactFlow>
        </div>
      </CardContent>
      
      {/* Legend */}
      <div className="p-4 border-t bg-muted/30">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span>Online</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-yellow-500" />
              <span>Degraded</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              <span>Offline</span>
            </div>
          </div>
          
          {selectedNF && (
            <Badge variant="outline" className="text-xs">
              Filtered: {selectedNF}
            </Badge>
          )}
          
          {simulationStatus === 'running' && (
            <div className="flex items-center space-x-1 text-green-600">
              <Activity className="w-3 h-3" />
              <span>Live Traffic Flow</span>
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}