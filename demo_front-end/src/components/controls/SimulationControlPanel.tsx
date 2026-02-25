'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { 
  Play, 
  Pause, 
  Square, 
  ChevronDown,
  ChevronRight,
  Smartphone,
  Wifi,
  AlertTriangle,
  Zap,
  Shield,
  Settings
} from 'lucide-react'
import { toast } from 'sonner'

interface SimulationControlPanelProps {
  simulationStatus: 'stopped' | 'running' | 'paused'
  onSimulationToggle: (status: 'stopped' | 'running' | 'paused') => void
  selectedNF: string | null
}

export function SimulationControlPanel({ 
  simulationStatus, 
  onSimulationToggle, 
  selectedNF 
}: SimulationControlPanelProps) {
  const [selectedProcedure, setSelectedProcedure] = useState('')
  const [ueId, setUeId] = useState('')
  const [anomalyType, setAnomalyType] = useState('')
  const [anomalyIntensity, setAnomalyIntensity] = useState([50])
  const [sidebarExpanded, setSidebarExpanded] = useState({
    coreNFs: true,
    ranNFs: true,
    activeUEs: false
  })

  // Mock data for UEs
  const mockUEs = [
    { id: 'ue001', status: 'Registered', gnb: 'gnb001', sessions: 2 },
    { id: 'ue002', status: 'Connected', gnb: 'gnb001', sessions: 1 },
    { id: 'ue003', status: 'Idle', gnb: 'gnb002', sessions: 0 },
  ]

  // Mock NF status data
  const coreNFs = [
    { name: 'AMF', status: 'online', load: 45 },
    { name: 'SMF', status: 'online', load: 32 },
    { name: 'UPF', status: 'online', load: 78 },
    { name: 'NRF', status: 'online', load: 12 },
    { name: 'AUSF', status: 'online', load: 28 },
    { name: 'UDM', status: 'online', load: 34 },
  ]

  const ranNFs = [
    { name: 'gNB-CU', status: 'online', load: 56 },
    { name: 'gNB-DU', status: 'online', load: 67 },
    { name: 'gNB-RRU', status: 'degraded', load: 89 },
  ]

  const procedures = [
    'UE Initial Registration',
    'PDU Session Establishment',
    'Xn Handover',
    'N2 Handover',
    'Service Request',
    'UE Context Release'
  ]

  const anomalies = [
    'AMF Latency Spike',
    'UPF Packet Drop',
    'gNB Connection Loss',
    'Database Timeout',
    'Network Congestion'
  ]

  const handleProcedureTrigger = () => {
    if (!selectedProcedure || !ueId) {
      toast.error('Please select a procedure and enter UE ID')
      return
    }
    toast.success(`Triggered ${selectedProcedure} for UE ${ueId}`)
    // TODO: Make API call to backend
  }

  const handleAnomalyInject = () => {
    if (!anomalyType) {
      toast.error('Please select an anomaly type')
      return
    }
    toast.warning(`Injected ${anomalyType} with ${anomalyIntensity[0]}% intensity`)
    // TODO: Make API call to backend
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500'
      case 'degraded': return 'bg-yellow-500'
      case 'offline': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'online': return 'default'
      case 'degraded': return 'secondary'
      case 'offline': return 'destructive'
      default: return 'outline'
    }
  }

  return (
    <div className="space-y-4 h-full overflow-y-auto">
      
      {/* Context Sidebar */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Network Context
          </CardTitle>
          <CardDescription>Filter dashboard by network component</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          
          {/* Core Network Functions */}
          <Collapsible 
            open={sidebarExpanded.coreNFs} 
            onOpenChange={(open) => setSidebarExpanded(prev => ({ ...prev, coreNFs: open }))}
          >
            <CollapsibleTrigger className="flex items-center space-x-2 w-full text-left">
              {sidebarExpanded.coreNFs ? 
                <ChevronDown className="w-4 h-4" /> : 
                <ChevronRight className="w-4 h-4" />
              }
              <span className="font-medium">Core Network</span>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-2 mt-2 ml-6">
              {coreNFs.map((nf) => (
                <div 
                  key={nf.name}
                  className={`flex items-center justify-between p-2 rounded cursor-pointer hover:bg-muted ${
                    selectedNF === nf.name ? 'bg-muted' : ''
                  }`}
                  onClick={() => selectedNF === nf.name ? null : null}
                >
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(nf.status)}`} />
                    <span className="text-sm font-medium">{nf.name}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Progress value={nf.load} className="w-12 h-2" />
                    <span className="text-xs text-muted-foreground">{nf.load}%</span>
                  </div>
                </div>
              ))}
            </CollapsibleContent>
          </Collapsible>

          {/* RAN Network Functions */}
          <Collapsible 
            open={sidebarExpanded.ranNFs} 
            onOpenChange={(open) => setSidebarExpanded(prev => ({ ...prev, ranNFs: open }))}
          >
            <CollapsibleTrigger className="flex items-center space-x-2 w-full text-left">
              {sidebarExpanded.ranNFs ? 
                <ChevronDown className="w-4 h-4" /> : 
                <ChevronRight className="w-4 h-4" />
              }
              <span className="font-medium">RAN Components</span>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-2 mt-2 ml-6">
              {ranNFs.map((nf) => (
                <div 
                  key={nf.name}
                  className={`flex items-center justify-between p-2 rounded cursor-pointer hover:bg-muted ${
                    selectedNF === nf.name ? 'bg-muted' : ''
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(nf.status)}`} />
                    <span className="text-sm font-medium">{nf.name}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Progress value={nf.load} className="w-12 h-2" />
                    <span className="text-xs text-muted-foreground">{nf.load}%</span>
                  </div>
                </div>
              ))}
            </CollapsibleContent>
          </Collapsible>

        </CardContent>
      </Card>

      {/* Simulation Controls */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center">
            <Play className="w-5 h-5 mr-2" />
            Simulation Control
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            
            {/* Trigger Procedure */}
            <AccordionItem value="procedures">
              <AccordionTrigger className="text-base">
                <div className="flex items-center">
                  <Zap className="w-4 h-4 mr-2" />
                  Trigger Procedure
                </div>
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="procedure">5G Procedure</Label>
                  <Select value={selectedProcedure} onValueChange={setSelectedProcedure}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a 5G procedure..." />
                    </SelectTrigger>
                    <SelectContent>
                      {procedures.map((procedure) => (
                        <SelectItem key={procedure} value={procedure}>
                          {procedure}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="ue-id">UE ID</Label>
                  <Input
                    id="ue-id"
                    placeholder="e.g., ue001"
                    value={ueId}
                    onChange={(e) => setUeId(e.target.value)}
                  />
                </div>
                <Button onClick={handleProcedureTrigger} className="w-full">
                  <Play className="w-4 h-4 mr-2" />
                  Execute Procedure
                </Button>
              </AccordionContent>
            </AccordionItem>

            {/* Inject Anomaly */}
            <AccordionItem value="anomalies">
              <AccordionTrigger className="text-base">
                <div className="flex items-center">
                  <AlertTriangle className="w-4 h-4 mr-2" />
                  Inject Anomaly
                </div>
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="anomaly">Anomaly Type</Label>
                  <Select value={anomalyType} onValueChange={setAnomalyType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select anomaly..." />
                    </SelectTrigger>
                    <SelectContent>
                      {anomalies.map((anomaly) => (
                        <SelectItem key={anomaly} value={anomaly}>
                          {anomaly}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Intensity: {anomalyIntensity[0]}%</Label>
                  <Slider
                    value={anomalyIntensity}
                    onValueChange={setAnomalyIntensity}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>
                <Button onClick={handleAnomalyInject} variant="destructive" className="w-full">
                  <AlertTriangle className="w-4 h-4 mr-2" />
                  Inject Anomaly
                </Button>
              </AccordionContent>
            </AccordionItem>

            {/* Manage UEs */}
            <AccordionItem value="ues">
              <AccordionTrigger className="text-base">
                <div className="flex items-center">
                  <Smartphone className="w-4 h-4 mr-2" />
                  Manage UEs ({mockUEs.length})
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>UE ID</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Sessions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {mockUEs.map((ue) => (
                        <TableRow key={ue.id}>
                          <TableCell className="font-medium">{ue.id}</TableCell>
                          <TableCell>
                            <Badge variant={getStatusVariant(ue.status)}>
                              {ue.status}
                            </Badge>
                          </TableCell>
                          <TableCell>{ue.sessions}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  <Button size="sm" className="w-full">
                    <Smartphone className="w-4 h-4 mr-2" />
                    Add New UE
                  </Button>
                </div>
              </AccordionContent>
            </AccordionItem>

          </Accordion>
        </CardContent>
      </Card>

      {/* N6 Interface Firewall Controls */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center">
            <Shield className="w-5 h-5 mr-2" />
            N6 Firewall
          </CardTitle>
          <CardDescription>BlueField-3 DPU Controls</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Hardware Acceleration</span>
            <Badge variant="default">DOCA Active</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Throughput</span>
            <span className="text-sm text-muted-foreground">156 Gbps</span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Packets Processed</span>
              <span>1.2M</span>
            </div>
            <Progress value={65} className="w-full" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Packets Dropped</span>
              <span>48K</span>
            </div>
            <Progress value={25} className="w-full" />
          </div>
          <Button size="sm" className="w-full" variant="outline">
            <Settings className="w-4 h-4 mr-2" />
            Configure Rules
          </Button>
        </CardContent>
      </Card>

    </div>
  )
}

function getStatusVariant(status: string): "default" | "destructive" | "outline" | "secondary" {
  switch (status.toLowerCase()) {
    case 'registered':
    case 'connected':
      return 'default'
    case 'idle':
      return 'secondary'
    case 'offline':
      return 'destructive'
    default:
      return 'outline'
  }
}