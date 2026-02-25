'use client'

import { useState } from 'react'
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable'
import { DashboardHeader } from './DashboardHeader'
import { SimulationControlPanel } from '@/components/controls/SimulationControlPanel'
import { NetworkTopologyView } from '@/components/network/NetworkTopologyView'
import { AnalyticsView } from '@/components/analytics/AnalyticsView'
import { Toaster } from '@/components/ui/sonner'

export function MissionControlDashboard() {
  const [selectedNF, setSelectedNF] = useState<string | null>(null)
  const [simulationStatus, setSimulationStatus] = useState<'stopped' | 'running' | 'paused'>('stopped')

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <DashboardHeader 
        simulationStatus={simulationStatus}
        onSimulationToggle={(status) => setSimulationStatus(status)}
      />
      
      {/* Main Dashboard Layout */}
      <div className="flex-1 p-4">
        <ResizablePanelGroup direction="horizontal" className="h-full rounded-lg border">
          
          {/* Left Panel: Simulation Control */}
          <ResizablePanel defaultSize={25} minSize={20} maxSize={35}>
            <div className="p-4 h-full">
              <SimulationControlPanel 
                simulationStatus={simulationStatus}
                onSimulationToggle={setSimulationStatus}
                selectedNF={selectedNF}
              />
            </div>
          </ResizablePanel>
          
          <ResizableHandle withHandle />
          
          {/* Center Panel: Network Topology */}
          <ResizablePanel defaultSize={50} minSize={30}>
            <ResizablePanelGroup direction="vertical">
              
              {/* Main Network View */}
              <ResizablePanel defaultSize={70} minSize={40}>
                <div className="p-4 h-full">
                  <NetworkTopologyView 
                    selectedNF={selectedNF}
                    onNFSelect={setSelectedNF}
                    simulationStatus={simulationStatus}
                  />
                </div>
              </ResizablePanel>
              
              <ResizableHandle withHandle />
              
              {/* Bottom Panel: Analytics */}
              <ResizablePanel defaultSize={30} minSize={20}>
                <div className="p-4 h-full">
                  <AnalyticsView 
                    selectedNF={selectedNF}
                    simulationStatus={simulationStatus}
                  />
                </div>
              </ResizablePanel>
              
            </ResizablePanelGroup>
          </ResizablePanel>
          
        </ResizablePanelGroup>
      </div>
      
      {/* Toast Notifications */}
      <Toaster />
    </div>
  )
}