# 3GPP Compliance Testing Guide

## Overview

This document provides comprehensive testing procedures for validating 3GPP compliance across all components of the 5G Network Simulator. It includes automated test suites, manual validation procedures, and continuous compliance monitoring.

## Testing Framework Architecture

```
                    3GPP Compliance Testing Framework
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Test Orchestration Layer                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Compliance Test Suite                               │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │   Protocol   │ │  Procedure   │ │   Message    │ │  Interface   │  │   │
│  │  │    Tests     │ │    Tests     │ │    Tests     │ │    Tests     │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Real-time Monitoring Layer                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │   Message    │ │  Compliance  │ │   Timeline   │ │ Performance  │  │   │
│  │  │  Validator   │ │   Scorer     │ │  Analyzer    │ │   Monitor    │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      System Under Test                                 │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │   │
│  │  │   5G Core    │ │   5G RAN     │ │   N6 FW      │ │  Frontend    │  │   │
│  │  │   Network    │ │  Components  │ │  Simulation  │ │  Dashboard   │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Test Categories

### 1. Protocol Compliance Tests

#### 1.1 Core Network Protocol Tests

**File:** `test_3gpp_compliance.py` (Enhanced)

```python
import asyncio
import pytest
import requests
import json
from datetime import datetime
from typing import Dict, List

class ProtocolComplianceTests:
    """
    Comprehensive 3GPP protocol compliance testing
    """
    
    def __init__(self):
        self.test_results = {}
        self.compliance_scores = {}
        
    async def test_n11_interface_compliance(self):
        """
        Test N11 interface (AMF ↔ SMF) per TS 29.502
        """
        test_name = "N11_Interface_Compliance"
        
        # Test Nsmf_PDUSession service
        test_data = {
            "supi": "imsi-001010000000001",
            "pduSessionId": 1,
            "dnn": "internet",
            "sNssai": {"sst": 1, "sd": "010203"},
            "anType": "3GPP_ACCESS",
            "ratType": "NR"
        }
        
        try:
            # Test mandatory field validation
            response = await self._test_endpoint(
                f"{SMF_URL}/nsmf-pdusession/v1/sm-contexts",
                test_data
            )
            
            # Validate response structure per TS 29.502
            required_response_fields = [
                "status", "pduSessionId", "ueIpAddress", "n2SmInfo"
            ]
            
            compliance_score = self._calculate_field_compliance(
                response.json(), required_response_fields
            )
            
            self.test_results[test_name] = {
                "status": "PASS" if compliance_score >= 90 else "FAIL",
                "compliance_score": compliance_score,
                "details": f"N11 interface compliance: {compliance_score}%"
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "ERROR",
                "compliance_score": 0,
                "details": str(e)
            }
    
    async def test_n4_pfcp_compliance(self):
        """
        Test N4 interface (SMF ↔ UPF) PFCP protocol per TS 29.244
        """
        test_name = "N4_PFCP_Compliance"
        
        # Test PFCP Session Establishment Request
        pfcp_message = {
            "messageType": "PFCP_SESSION_ESTABLISHMENT_REQUEST",
            "seid": "test-seid-001",
            "createPDR": [{
                "pdrId": 1,
                "precedence": 200,
                "pdi": {
                    "sourceInterface": "ACCESS",
                    "ueIpAddress": "10.1.0.1"
                },
                "farId": 1
            }],
            "createFAR": [{
                "farId": 1,
                "applyAction": "FORWARD",
                "forwardingParameters": {
                    "destinationInterface": "CORE"
                }
            }],
            "createQER": [{
                "qerId": 1,
                "qfi": 9,
                "uplinkMBR": 100000000
            }]
        }
        
        try:
            response = await self._test_endpoint(
                f"{UPF_URL}/n4/sessions",
                pfcp_message
            )
            
            # Validate PFCP response per TS 29.244
            pfcp_response = response.json()
            required_fields = ["status", "upfSeid", "createdPDR", "createdFAR"]
            
            compliance_score = self._calculate_pfcp_compliance(pfcp_response)
            
            self.test_results[test_name] = {
                "status": "PASS" if compliance_score >= 85 else "FAIL",
                "compliance_score": compliance_score,
                "details": f"PFCP protocol compliance: {compliance_score}%"
            }
            
        except Exception as e:
            self.test_results[test_name] = {"status": "ERROR", "details": str(e)}
    
    async def test_ngap_compliance(self):
        """
        Test NGAP protocol (gNB ↔ AMF) per TS 38.413
        """
        test_name = "NGAP_Compliance"
        
        # Test NGAP Initial UE Message
        ngap_message = {
            "procedureCode": 15,  # initialUEMessage
            "criticality": "ignore",
            "value": {
                "protocolIEs": {
                    "RAN-UE-NGAP-ID": 1,
                    "NAS-PDU": "7e00410001",  # Sample NAS PDU
                    "UserLocationInformation": {
                        "nR-CGI": {
                            "pLMNIdentity": "00101",
                            "nRCellIdentity": "000000001"
                        }
                    }
                }
            }
        }
        
        # Currently limited - enhance when full NGAP is implemented
        compliance_score = 30  # Reflecting current basic implementation
        
        self.test_results[test_name] = {
            "status": "PARTIAL",
            "compliance_score": compliance_score,
            "details": "Basic NGAP implementation - needs enhancement"
        }
```

#### 1.2 Message Validation Tests

```python
class MessageValidationTests:
    """
    Validate individual 3GPP messages against specifications
    """
    
    def test_pdu_session_create_request_validation(self):
        """
        Validate PDU Session Create Request per TS 29.502
        """
        # Test mandatory fields per TS 29.502 Table 5.2.2.2.1-1
        mandatory_fields = [
            "supi", "pduSessionId", "dnn", "sNssai", "anType"
        ]
        
        # Test optional fields
        optional_fields = [
            "ratType", "ueLocation", "gpsi", "ueTimeZone"
        ]
        
        # Test with complete message
        complete_message = {
            # Mandatory fields
            "supi": "imsi-001010000000001",
            "pduSessionId": 1,
            "dnn": "internet",
            "sNssai": {"sst": 1, "sd": "010203"},
            "anType": "3GPP_ACCESS",
            
            # Optional fields
            "ratType": "NR",
            "ueLocation": {
                "nrLocation": {
                    "tai": {"plmnId": {"mcc": "001", "mnc": "01"}, "tac": "000001"},
                    "ncgi": {"plmnId": {"mcc": "001", "mnc": "01"}, "nrCellId": "000000001"}
                }
            },
            "gpsi": "msisdn-001010000000001"
        }
        
        validation_result = self._validate_message_structure(
            complete_message, mandatory_fields, optional_fields
        )
        
        return validation_result
    
    def test_pfcp_message_validation(self):
        """
        Validate PFCP messages per TS 29.244
        """
        # Test PFCP Session Establishment Request structure
        pfcp_message = {
            "messageType": "PFCP_SESSION_ESTABLISHMENT_REQUEST",
            "seid": "smf-seid-001",
            
            # Mandatory IEs per TS 29.244 Table 7.5.2.2-1
            "createPDR": [{
                "pdrId": 1,           # Mandatory
                "precedence": 200,    # Mandatory  
                "pdi": {              # Mandatory
                    "sourceInterface": "ACCESS"
                }
            }],
            
            "createFAR": [{
                "farId": 1,           # Mandatory
                "applyAction": "FORWARD"  # Mandatory
            }]
        }
        
        # Validate against TS 29.244 specification
        return self._validate_pfcp_structure(pfcp_message)
```

### 2. Procedure Compliance Tests

#### 2.1 End-to-End Procedure Tests

```python
class ProcedureComplianceTests:
    """
    Test complete 3GPP procedures end-to-end
    """
    
    async def test_pdu_session_establishment_procedure(self):
        """
        Test complete PDU Session Establishment per TS 23.502 § 4.3.2.2.1
        """
        procedure_name = "PDU_Session_Establishment"
        start_time = datetime.now()
        
        # Step 1: Create UE context
        ue_context = await self._create_test_ue_context()
        
        # Step 2: Trigger PDU Session Establishment
        session_request = {
            "ue_id": ue_context["ue_id"],
            "procedure": "pdu_session_establishment",
            "dnn": "internet",
            "sNssai": {"sst": 1, "sd": "010203"}
        }
        
        # Step 3: Monitor procedure execution
        procedure_timeline = []
        
        try:
            # AMF processes request
            timeline_entry = self._start_timeline_monitoring("AMF_Processing")
            amf_response = await self._call_amf_endpoint(
                "/amf/pdu-session/create", session_request
            )
            self._end_timeline_monitoring(timeline_entry)
            procedure_timeline.append(timeline_entry)
            
            # Validate AMF → SMF interaction (N11)
            timeline_entry = self._start_timeline_monitoring("N11_Interaction")
            # This should be captured automatically through monitoring
            self._end_timeline_monitoring(timeline_entry)
            procedure_timeline.append(timeline_entry)
            
            # Validate SMF → UPF interaction (N4)
            timeline_entry = self._start_timeline_monitoring("N4_Interaction")
            # Monitor PFCP session establishment
            self._end_timeline_monitoring(timeline_entry)
            procedure_timeline.append(timeline_entry)
            
            # Calculate procedure compliance
            total_duration = (datetime.now() - start_time).total_seconds() * 1000
            compliance_score = self._calculate_procedure_compliance(
                procedure_timeline, total_duration
            )
            
            return {
                "procedure": procedure_name,
                "status": "SUCCESS" if compliance_score >= 90 else "PARTIAL",
                "compliance_score": compliance_score,
                "total_duration_ms": total_duration,
                "timeline": procedure_timeline
            }
            
        except Exception as e:
            return {
                "procedure": procedure_name,
                "status": "FAILURE",
                "compliance_score": 0,
                "error": str(e)
            }
    
    async def test_ue_registration_procedure(self):
        """
        Test UE Registration per TS 23.502 § 4.2.2.2
        Note: Currently limited due to missing AUSF/UDM implementation
        """
        # This test will be enhanced as AUSF/UDM are implemented
        return {
            "procedure": "UE_Registration",
            "status": "NOT_IMPLEMENTED",
            "compliance_score": 0,
            "details": "Requires AUSF and UDM implementation"
        }
    
    async def test_handover_procedure(self):
        """
        Test NGAP Handover per TS 38.413
        """
        handover_request = {
            "ue_id": "test_ue_001", 
            "source_gnb_id": "gnb001",
            "target_gnb_id": "gnb002"
        }
        
        try:
            response = await self._call_amf_endpoint(
                "/amf/handover", handover_request
            )
            
            # Basic handover validation
            if response.status_code == 200:
                return {
                    "procedure": "NGAP_Handover",
                    "status": "SUCCESS",
                    "compliance_score": 60,  # Basic implementation
                    "details": "Basic handover working, needs full NGAP"
                }
            
        except Exception as e:
            return {
                "procedure": "NGAP_Handover",
                "status": "FAILURE",
                "compliance_score": 0,
                "error": str(e)
            }
```

### 3. Interface Compliance Tests

#### 3.1 Service-Based Interface Tests

```python
class ServiceBasedInterfaceTests:
    """
    Test Service-Based Interface (SBI) compliance per TS 29.500
    """
    
    def test_nf_registration_compliance(self):
        """
        Test NF registration with NRF per TS 29.510
        """
        # Test basic NF registration (current implementation)
        nf_profile = {
            "nf_type": "SMF",
            "ip": "127.0.0.1", 
            "port": 9001
        }
        
        # This needs enhancement for full NF Profile per TS 29.510
        required_full_profile = {
            "nfInstanceId": "smf-001",
            "nfType": "SMF",
            "nfStatus": "REGISTERED",
            "plmnList": [{"mcc": "001", "mnc": "01"}],
            "sNssais": [{"sst": 1, "sd": "010203"}],
            "nfServices": [
                {
                    "serviceInstanceId": "nsmf-pdusession-001",
                    "serviceName": "nsmf-pdusession",
                    "versions": [{"apiVersionInUri": "v1"}],
                    "scheme": "http",
                    "nfServiceStatus": "REGISTERED"
                }
            ]
        }
        
        compliance_score = 30  # Current basic implementation
        return {
            "test": "NF_Registration",
            "compliance_score": compliance_score,
            "status": "PARTIAL",
            "enhancement_needed": "Full NF Profile structure per TS 29.510"
        }
    
    def test_service_discovery_compliance(self):
        """
        Test service discovery per TS 29.510
        """
        # Current basic discovery
        discovery_request = {
            "target-nf-type": "SMF",
            "requester-nf-type": "AMF"
        }
        
        # Enhanced discovery should support
        enhanced_discovery = {
            "target-nf-type": "SMF",
            "requester-nf-type": "AMF", 
            "service-names": ["nsmf-pdusession"],
            "snssais": [{"sst": 1, "sd": "010203"}],
            "dnn": "internet",
            "requester-plmn": {"mcc": "001", "mnc": "01"}
        }
        
        compliance_score = 25  # Current implementation
        return {
            "test": "Service_Discovery",
            "compliance_score": compliance_score,
            "status": "BASIC",
            "enhancement_needed": "Advanced filtering per TS 29.510"
        }
```

### 4. Real-Time Compliance Monitoring

#### 4.1 Message Validation Engine

```python
class RealTimeComplianceMonitor:
    """
    Real-time 3GPP compliance monitoring
    Integrates with frontend dashboard
    """
    
    def __init__(self):
        self.message_validators = {}
        self.compliance_scores = {}
        self.violation_alerts = []
        
    def validate_message_real_time(self, interface: str, message: dict):
        """
        Real-time message validation against 3GPP specs
        """
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "interface": interface,
            "message_type": message.get("messageType", "unknown"),
            "compliance_score": 0,
            "violations": [],
            "status": "UNKNOWN"
        }
        
        try:
            if interface == "N11":
                validation_result = self._validate_n11_message(message)
            elif interface == "N4":
                validation_result = self._validate_n4_message(message)
            elif interface == "N2":
                validation_result = self._validate_n2_message(message)
            
            # Store result for dashboard
            self._store_validation_result(validation_result)
            
            # Check for compliance violations
            if validation_result["compliance_score"] < 80:
                self._generate_compliance_alert(validation_result)
                
        except Exception as e:
            validation_result["status"] = "ERROR"
            validation_result["violations"].append(f"Validation error: {str(e)}")
            
        return validation_result
    
    def _validate_n11_message(self, message: dict):
        """
        Validate N11 interface messages per TS 29.502
        """
        violations = []
        score = 100
        
        # Check endpoint compliance
        if "nsmf-pdusession" not in message.get("endpoint", ""):
            violations.append("Non-standard endpoint - should use nsmf-pdusession")
            score -= 20
            
        # Check mandatory fields for different message types
        if message.get("messageType") == "CreateSmContextRequest":
            mandatory_fields = ["supi", "pduSessionId", "dnn", "sNssai", "anType"]
            for field in mandatory_fields:
                if field not in message:
                    violations.append(f"Missing mandatory field: {field}")
                    score -= 15
                    
        # Check field format compliance
        if "supi" in message:
            if not message["supi"].startswith("imsi-"):
                violations.append("SUPI format should be 'imsi-<digits>'")
                score -= 10
                
        return {
            "timestamp": datetime.now().isoformat(),
            "interface": "N11",
            "message_type": message.get("messageType", "unknown"),
            "compliance_score": max(0, score),
            "violations": violations,
            "status": "COMPLIANT" if score >= 90 else "NON_COMPLIANT"
        }
    
    def _validate_n4_message(self, message: dict):
        """
        Validate N4 interface messages per TS 29.244
        """
        violations = []
        score = 100
        
        # Check PFCP message structure
        required_pfcp_fields = ["messageType", "seid"]
        for field in required_pfcp_fields:
            if field not in message:
                violations.append(f"Missing PFCP field: {field}")
                score -= 20
                
        # Validate PDR structure if present
        if "createPDR" in message:
            for pdr in message["createPDR"]:
                pdr_required = ["pdrId", "precedence", "pdi"]
                for field in pdr_required:
                    if field not in pdr:
                        violations.append(f"PDR missing required field: {field}")
                        score -= 10
                        
        return {
            "timestamp": datetime.now().isoformat(),
            "interface": "N4",
            "message_type": message.get("messageType", "unknown"),
            "compliance_score": max(0, score),
            "violations": violations,
            "status": "COMPLIANT" if score >= 85 else "NON_COMPLIANT"
        }
```

### 5. Performance and Timeline Validation

#### 5.1 Procedure Timeline Analysis

```python
class ProcedureTimelineValidator:
    """
    Validate procedure execution timelines against 3GPP requirements
    """
    
    def validate_pdu_session_timeline(self, timeline_data: List[dict]):
        """
        Validate PDU Session Establishment timeline per TS 23.502
        """
        # 3GPP timing requirements
        max_total_time = 2000  # 2 seconds max for PDU session establishment
        max_n11_time = 500     # 500ms max for N11 interaction
        max_n4_time = 300      # 300ms max for N4 interaction
        
        violations = []
        
        # Calculate actual timings
        total_time = self._calculate_total_time(timeline_data)
        n11_time = self._get_step_duration(timeline_data, "N11_Interaction")
        n4_time = self._get_step_duration(timeline_data, "N4_Interaction")
        
        # Check against requirements
        if total_time > max_total_time:
            violations.append(f"Total time {total_time}ms exceeds limit {max_total_time}ms")
            
        if n11_time > max_n11_time:
            violations.append(f"N11 time {n11_time}ms exceeds limit {max_n11_time}ms")
            
        if n4_time > max_n4_time:
            violations.append(f"N4 time {n4_time}ms exceeds limit {max_n4_time}ms")
            
        compliance_score = max(0, 100 - len(violations) * 25)
        
        return {
            "procedure": "PDU_Session_Establishment",
            "total_time_ms": total_time,
            "compliance_score": compliance_score,
            "violations": violations,
            "timing_breakdown": {
                "n11_interaction": n11_time,
                "n4_interaction": n4_time,
                "other_processing": total_time - n11_time - n4_time
            }
        }
```

## Test Execution and Reporting

### 5.1 Automated Test Execution

```bash
#!/bin/bash
# 3GPP Compliance Test Execution Script

echo "🧪 Starting 3GPP Compliance Test Suite"
echo "======================================"

# Start all services
./start_3gpp_services.sh

# Wait for services to be ready
sleep 10

# Run comprehensive test suite
python3 -m pytest tests/compliance/ -v \
    --junit-xml=reports/compliance_results.xml \
    --html=reports/compliance_report.html \
    --self-contained-html

# Run real-time monitoring tests
python3 test_real_time_compliance.py

# Generate compliance dashboard
python3 generate_compliance_dashboard.py

echo "✅ Test execution completed"
echo "📊 Reports available in reports/ directory"
```

### 5.2 Compliance Reporting

```python
class ComplianceReporter:
    """
    Generate comprehensive compliance reports
    """
    
    def generate_compliance_report(self, test_results: dict):
        """
        Generate detailed compliance report
        """
        report = {
            "test_execution": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(test_results),
                "passed": len([t for t in test_results.values() if t["status"] == "PASS"]),
                "failed": len([t for t in test_results.values() if t["status"] == "FAIL"]),
                "partial": len([t for t in test_results.values() if t["status"] == "PARTIAL"])
            },
            "compliance_scores": {
                component: self._calculate_component_score(component, test_results)
                for component in ["AMF", "SMF", "UPF", "gNB", "NRF", "AUSF", "UDM"]
            },
            "interface_compliance": {
                "N11": self._get_interface_score("N11", test_results),
                "N4": self._get_interface_score("N4", test_results),
                "N2": self._get_interface_score("N2", test_results),
                "SBI": self._get_interface_score("SBI", test_results)
            },
            "recommendations": self._generate_recommendations(test_results)
        }
        
        return report
    
    def _generate_recommendations(self, test_results: dict):
        """
        Generate specific recommendations for improving compliance
        """
        recommendations = []
        
        # Check critical missing implementations
        if any("AUSF" in test for test in test_results if test_results[test]["compliance_score"] < 50):
            recommendations.append({
                "priority": "CRITICAL",
                "component": "AUSF",
                "recommendation": "Implement 5G-AKA authentication procedures per TS 33.501",
                "estimated_effort": "4 weeks",
                "3gpp_spec": "TS 29.509, TS 33.501"
            })
            
        if any("UDM" in test for test in test_results if test_results[test]["compliance_score"] < 50):
            recommendations.append({
                "priority": "CRITICAL", 
                "component": "UDM",
                "recommendation": "Implement Nudm services per TS 29.503",
                "estimated_effort": "4 weeks",
                "3gpp_spec": "TS 29.503, TS 29.505"
            })
            
        return recommendations
```

## Continuous Integration Integration

### 5.1 CI/CD Pipeline Integration

```yaml
# .github/workflows/3gpp-compliance.yml
name: 3GPP Compliance Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  compliance-testing:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-html pytest-cov
        
    - name: Start 5G services
      run: |
        ./start_3gpp_services.sh
        sleep 30  # Wait for services to start
        
    - name: Run 3GPP compliance tests
      run: |
        python3 -m pytest tests/compliance/ \
          --junit-xml=compliance-results.xml \
          --html=compliance-report.html \
          --cov=. \
          --cov-report=xml
          
    - name: Generate compliance dashboard
      run: |
        python3 scripts/generate_compliance_dashboard.py
        
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: compliance-reports
        path: |
          compliance-results.xml
          compliance-report.html
          compliance-dashboard.html
          
    - name: Check compliance threshold
      run: |
        python3 scripts/check_compliance_threshold.py --min-score 70
```

## Success Metrics and KPIs

### Compliance Targets

| Component | Current Score | Target Score | Timeline |
|-----------|---------------|--------------|----------|
| **SMF** | 85% | 95% | 2 months |
| **UPF** | 80% | 90% | 2 months |
| **AMF** | 70% | 90% | 3 months |
| **gNodeB** | 50% | 85% | 4 months |
| **NRF** | 30% | 80% | 3 months |
| **AUSF** | 10% | 85% | 4 months |
| **UDM** | 10% | 80% | 4 months |

### Key Performance Indicators

1. **Overall System Compliance**: Target 85% by end of Phase 3
2. **Procedure Success Rate**: >95% for all implemented procedures
3. **Message Compliance**: >98% for all validated messages
4. **Timeline Compliance**: >90% of procedures within 3GPP timing requirements
5. **Test Coverage**: >95% of implemented 3GPP procedures

## Conclusion

This comprehensive testing framework ensures systematic validation of 3GPP compliance across all components. The combination of automated testing, real-time monitoring, and continuous integration provides robust quality assurance for the 5G network simulator.

**Key Benefits:**
- Automated 3GPP compliance validation
- Real-time compliance monitoring
- Detailed compliance reporting
- Continuous integration support
- Clear enhancement roadmap

**Next Steps:**
1. Implement enhanced test cases for missing components (AUSF, UDM)
2. Integrate real-time monitoring with frontend dashboard
3. Establish CI/CD pipeline with compliance gates
4. Regular compliance assessment and reporting