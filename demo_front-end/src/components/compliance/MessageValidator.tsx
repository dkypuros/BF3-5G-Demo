'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  RefreshCw,
  MessageSquare,
  Network
} from 'lucide-react'

interface MessageValidationResult {
  messageType: string
  interface: string
  timestamp: string
  complianceScore: number
  requiredFields: string[]
  presentFields: string[]
  missingFields: string[]
  extraFields: string[]
  violations: string[]
}

interface MessageValidatorProps {
  selectedNF?: string | null
  simulationStatus: 'stopped' | 'running' | 'paused'
}

export function MessageValidator({ selectedNF, simulationStatus }: MessageValidatorProps) {
  const [validationResults, setValidationResults] = useState<MessageValidationResult[]>([])
  const [isValidating, setIsValidating] = useState(false)

  // Mock validation results - in real implementation, this would come from backend
  useEffect(() => {
    if (simulationStatus === 'running') {
      const interval = setInterval(() => {
        const newResult: MessageValidationResult = generateMockValidationResult()
        setValidationResults(prev => [newResult, ...prev.slice(0, 9)]) // Keep last 10 results
      }, 3000)

      return () => clearInterval(interval)
    }
  }, [simulationStatus])

  const generateMockValidationResult = (): MessageValidationResult => {
    const procedures = [
      {
        messageType: 'Create SM Context Request',
        interface: 'N11 (Nsmf_PDUSession)',
        required: ['supi', 'pduSessionId', 'dnn', 'sNssai', 'anType'],
        present: ['supi', 'pduSessionId', 'dnn', 'sNssai', 'anType', 'ratType', 'ueLocation'],
        violations: []
      },
      {
        messageType: 'PFCP Session Establishment Request',
        interface: 'N4 (PFCP)',
        required: ['seid', 'createPDR', 'createFAR'],
        present: ['seid', 'createPDR', 'createFAR', 'createQER'],
        violations: []
      },
      {
        messageType: 'Registration Request',
        interface: 'N1 (NAS)',
        required: ['messageType', '5gmmCapability', 'ueSecurityCapability'],
        present: ['messageType', '5gmmCapability', 'ueSecurityCapability', 'customField'],
        violations: ['Non-standard field: customField']
      }
    ]

    const procedure = procedures[Math.floor(Math.random() * procedures.length)]
    const missing = procedure.required.filter(field => !procedure.present.includes(field))
    const extra = procedure.present.filter(field => !procedure.required.includes(field))
    
    const complianceScore = Math.max(0, 100 - (missing.length * 20) - (procedure.violations.length * 10) - (extra.length * 5))

    return {
      messageType: procedure.messageType,
      interface: procedure.interface,
      timestamp: new Date().toISOString().substring(11, 23),
      complianceScore,
      requiredFields: procedure.required,
      presentFields: procedure.present,
      missingFields: missing,
      extraFields: extra,
      violations: procedure.violations
    }
  }

  const validateMessage = async () => {
    setIsValidating(true)
    // Simulate API call to validate current messages
    setTimeout(() => {
      const result = generateMockValidationResult()
      setValidationResults(prev => [result, ...prev.slice(0, 9)])
      setIsValidating(false)
    }, 1000)
  }

  const getComplianceColor = (score: number) => {
    if (score >= 95) return 'text-green-500'
    if (score >= 80) return 'text-yellow-500'
    return 'text-red-500'
  }

  const getComplianceIcon = (score: number) => {
    if (score >= 95) return <CheckCircle className="w-4 h-4 text-green-500" />
    if (score >= 80) return <AlertCircle className="w-4 h-4 text-yellow-500" />
    return <XCircle className="w-4 h-4 text-red-500" />
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center">
              <MessageSquare className="w-5 h-5 mr-2" />
              3GPP Message Validator
            </CardTitle>
            <CardDescription>
              Real-time validation against 3GPP specifications
              {selectedNF && ` • Filtered by: ${selectedNF}`}
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={validateMessage}
            disabled={isValidating}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isValidating ? 'animate-spin' : ''}`} />
            Validate
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="p-4">
        <Tabs defaultValue="live" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="live">Live Validation</TabsTrigger>
            <TabsTrigger value="summary">Compliance Summary</TabsTrigger>
          </TabsList>

          <TabsContent value="live" className="space-y-4">
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {validationResults.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Network className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>Start simulation to see message validation results</p>
                </div>
              ) : (
                validationResults.map((result, index) => (
                  <Card key={index} className="p-3">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="flex items-center space-x-2">
                          {getComplianceIcon(result.complianceScore)}
                          <span className="font-medium text-sm">{result.messageType}</span>
                        </div>
                        <div className="flex items-center space-x-2 mt-1">
                          <Badge variant="secondary" className="text-xs">
                            {result.interface}
                          </Badge>
                          <span className="text-xs text-muted-foreground font-mono">
                            {result.timestamp}
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-bold text-lg ${getComplianceColor(result.complianceScore)}`}>
                          {result.complianceScore}%
                        </div>
                        <div className="text-xs text-muted-foreground">Compliance</div>
                      </div>
                    </div>

                    {(result.missingFields.length > 0 || result.violations.length > 0 || result.extraFields.length > 0) && (
                      <div className="space-y-2 text-xs">
                        {result.missingFields.length > 0 && (
                          <div className="flex items-center space-x-2">
                            <XCircle className="w-3 h-3 text-red-500" />
                            <span className="text-red-500">
                              Missing: {result.missingFields.join(', ')}
                            </span>
                          </div>
                        )}
                        
                        {result.violations.length > 0 && (
                          <div className="flex items-center space-x-2">
                            <AlertCircle className="w-3 h-3 text-yellow-500" />
                            <span className="text-yellow-500">
                              {result.violations.join(', ')}
                            </span>
                          </div>
                        )}

                        {result.extraFields.length > 0 && (
                          <div className="flex items-center space-x-2">
                            <AlertCircle className="w-3 h-3 text-blue-500" />
                            <span className="text-blue-500">
                              Extra: {result.extraFields.join(', ')}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </Card>
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="summary" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {validationResults.filter(r => r.complianceScore >= 95).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Fully Compliant</div>
                </div>
              </Card>
              
              <Card className="p-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-600">
                    {validationResults.filter(r => r.complianceScore >= 80 && r.complianceScore < 95).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Minor Issues</div>
                </div>
              </Card>
              
              <Card className="p-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-600">
                    {validationResults.filter(r => r.complianceScore < 80).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Non-Compliant</div>
                </div>
              </Card>
            </div>

            <Card className="p-4">
              <CardTitle className="text-base mb-3">Interface Coverage</CardTitle>
              <div className="space-y-2">
                {Array.from(new Set(validationResults.map(r => r.interface))).map(intf => {
                  const interfaceResults = validationResults.filter(r => r.interface === intf)
                  const avgScore = interfaceResults.reduce((sum, r) => sum + r.complianceScore, 0) / interfaceResults.length
                  return (
                    <div key={intf} className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">{intf}</Badge>
                      <div className={`text-sm font-medium ${getComplianceColor(avgScore || 0)}`}>
                        {avgScore ? Math.round(avgScore) : 0}% avg
                      </div>
                    </div>
                  )
                })}
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}