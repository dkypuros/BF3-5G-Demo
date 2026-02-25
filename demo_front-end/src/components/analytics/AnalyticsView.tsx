'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts'
import { 
  BarChart3, 
  FileText, 
  Activity, 
  Search, 
  Filter,
  Download,
  RefreshCw,
  MessageSquare
} from 'lucide-react'
import { MessageValidator } from '@/components/compliance/MessageValidator'

interface AnalyticsViewProps {
  selectedNF: string | null
  simulationStatus: 'stopped' | 'running' | 'paused'
}

export function AnalyticsView({ selectedNF, simulationStatus }: AnalyticsViewProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [logLevel, setLogLevel] = useState('all')
  const [refreshing, setRefreshing] = useState(false)

  // Mock metrics data
  const metricsData = [
    { time: '14:30', handoverDuration: 85, registrations: 12, sessions: 23 },
    { time: '14:31', handoverDuration: 92, registrations: 8, sessions: 25 },
    { time: '14:32', handoverDuration: 78, registrations: 15, sessions: 28 },
    { time: '14:33', handoverDuration: 88, registrations: 11, sessions: 26 },
    { time: '14:34', handoverDuration: 95, registrations: 9, sessions: 30 },
    { time: '14:35', handoverDuration: 82, registrations: 13, sessions: 32 },
  ]

  const throughputData = [
    { time: '14:30', upf: 120, firewall: 118, dropped: 2 },
    { time: '14:31', upf: 145, firewall: 142, dropped: 3 },
    { time: '14:32', upf: 163, firewall: 160, dropped: 3 },
    { time: '14:33', upf: 178, firewall: 175, dropped: 3 },
    { time: '14:34', upf: 156, firewall: 152, dropped: 4 },
    { time: '14:35', upf: 171, firewall: 168, dropped: 3 },
  ]

  // Mock logs data
  const mockLogs = [
    {
      id: 1,
      timestamp: '2024-08-14 14:35:23',
      nf: 'AMF',
      level: 'INFO',
      message: 'UE ue001 successfully registered with gNB gnb001'
    },
    {
      id: 2,
      timestamp: '2024-08-14 14:35:22',
      nf: 'SMF',
      level: 'INFO',
      message: 'PDU session established for UE ue001'
    },
    {
      id: 3,
      timestamp: '2024-08-14 14:35:21',
      nf: 'UPF',
      level: 'WARN',
      message: 'High packet rate detected from UE ue002'
    },
    {
      id: 4,
      timestamp: '2024-08-14 14:35:20',
      nf: 'BF3-FW',
      level: 'INFO',
      message: 'Blocked 127 packets on port 8001 (firewall rule match)'
    },
    {
      id: 5,
      timestamp: '2024-08-14 14:35:19',
      nf: 'gNB-CU',
      level: 'ERROR',
      message: 'Failed to establish connection with UE ue003'
    },
  ]

  // Mock trace data for procedure analysis
  const mockTrace = [
    { span: 'amf_receive_registration_request', duration: 2.3, start: 0 },
    { span: 'ausf_authentication', duration: 12.8, start: 2.3 },
    { span: 'udm_subscription_data', duration: 8.1, start: 15.1 },
    { span: 'amf_registration_accept', duration: 1.9, start: 23.2 },
    { span: 'smf_pdu_session_create', duration: 15.4, start: 25.1 },
    { span: 'upf_session_establishment', duration: 8.7, start: 40.5 },
  ]

  const filteredLogs = mockLogs.filter(log => {
    const matchesSearch = log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         log.nf.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesLevel = logLevel === 'all' || log.level === logLevel
    const matchesNF = !selectedNF || log.nf === selectedNF
    return matchesSearch && matchesLevel && matchesNF
  })

  const handleRefresh = () => {
    setRefreshing(true)
    setTimeout(() => setRefreshing(false), 1000)
  }

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return 'destructive'
      case 'WARN': return 'secondary'
      case 'INFO': return 'default'
      default: return 'outline'
    }
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center">
              <BarChart3 className="w-5 h-5 mr-2" />
              Analytics & Monitoring
            </CardTitle>
            <CardDescription>
              {selectedNF ? `Filtered by: ${selectedNF}` : 'Real-time system analytics'}
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="p-0 h-[calc(100%-100px)]">
        <Tabs defaultValue="metrics" className="h-full">
          <TabsList className="grid w-full grid-cols-4 mx-4 mt-2">
            <TabsTrigger value="metrics">
              <BarChart3 className="w-4 h-4 mr-2" />
              Metrics
            </TabsTrigger>
            <TabsTrigger value="logs">
              <FileText className="w-4 h-4 mr-2" />
              Logs
            </TabsTrigger>
            <TabsTrigger value="traces">
              <Activity className="w-4 h-4 mr-2" />
              Traces
            </TabsTrigger>
            <TabsTrigger value="compliance">
              <MessageSquare className="w-4 h-4 mr-2" />
              3GPP
            </TabsTrigger>
          </TabsList>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="p-4 h-[calc(100%-60px)] overflow-y-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
              
              {/* Handover Duration Chart */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Handover Duration (p95)</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={metricsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="handoverDuration" 
                        stroke="#8884d8" 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Active Sessions */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Active PDU Sessions</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={150}>
                    <AreaChart data={metricsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="sessions" 
                        stroke="#82ca9d" 
                        fill="#82ca9d" 
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Throughput Data */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">N6 Interface Throughput (Gbps)</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={throughputData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Bar dataKey="upf" fill="#8884d8" name="UPF Input" />
                      <Bar dataKey="firewall" fill="#82ca9d" name="Firewall Output" />
                      <Bar dataKey="dropped" fill="#ff7300" name="Dropped" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Registration Success Rate */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Registration Success Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={metricsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="registrations" 
                        stroke="#ff7300" 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

            </div>
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs" className="p-4 h-[calc(100%-60px)]">
            <div className="space-y-4 h-full">
              {/* Log Filters */}
              <div className="flex items-center space-x-2">
                <div className="flex-1">
                  <Input
                    placeholder="Search logs..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-8"
                  />
                  <Search className="w-4 h-4 absolute left-2 top-3 text-muted-foreground" />
                </div>
                <Select value={logLevel} onValueChange={setLogLevel}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Levels</SelectItem>
                    <SelectItem value="ERROR">ERROR</SelectItem>
                    <SelectItem value="WARN">WARN</SelectItem>
                    <SelectItem value="INFO">INFO</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
              </div>

              {/* Logs Table */}
              <div className="border rounded-lg h-[calc(100%-60px)] overflow-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-40">Timestamp</TableHead>
                      <TableHead className="w-20">NF</TableHead>
                      <TableHead className="w-20">Level</TableHead>
                      <TableHead>Message</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredLogs.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell className="font-mono text-xs">{log.timestamp}</TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-xs">
                            {log.nf}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={getLogLevelColor(log.level)} className="text-xs">
                            {log.level}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-sm">{log.message}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          </TabsContent>

          {/* Traces Tab */}
          <TabsContent value="traces" className="p-4 h-[calc(100%-60px)]">
            <div className="space-y-4 h-full">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold">OpenTelemetry Traces</h3>
                  <p className="text-sm text-muted-foreground">
                    Last UE Registration Procedure (ue001)
                  </p>
                </div>
                <Badge variant="default">Total: 49.2ms</Badge>
              </div>

              {/* Trace Timeline */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Procedure Timeline</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {mockTrace.map((span, index) => (
                      <div key={span.span} className="relative">
                        <div className="flex items-center justify-between py-2">
                          <div className="flex items-center space-x-3">
                            <div className="w-2 h-2 rounded-full bg-blue-500" />
                            <span className="text-sm font-medium">{span.span}</span>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            {span.duration}ms
                          </Badge>
                        </div>
                        
                        {/* Timeline bar */}
                        <div className="ml-5 h-2 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-500 transition-all duration-1000"
                            style={{ 
                              width: `${(span.duration / 50) * 100}%`,
                              marginLeft: `${(span.start / 50) * 100}%`
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Trace Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Performance Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Total Duration</span>
                      <span className="font-mono">49.2ms</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Spans</span>
                      <span className="font-mono">{mockTrace.length}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Critical Path</span>
                      <span className="font-mono">smf_pdu_session_create</span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">3GPP Compliance</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Spec Adherence</span>
                      <Badge variant="default">98.5%</Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Timing Requirements</span>
                      <Badge variant="default">✓ Met</Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Message Format</span>
                      <Badge variant="secondary">1 Warning</Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* 3GPP Compliance Tab */}
          <TabsContent value="compliance" className="p-0 h-[calc(100%-60px)]">
            <MessageValidator 
              selectedNF={selectedNF}
              simulationStatus={simulationStatus}
            />
          </TabsContent>

        </Tabs>
      </CardContent>
    </Card>
  )
}

function getLogLevelColor(level: string): "default" | "destructive" | "outline" | "secondary" {
  switch (level) {
    case 'ERROR': return 'destructive'
    case 'WARN': return 'secondary'
    case 'INFO': return 'default'
    default: return 'outline'
  }
}