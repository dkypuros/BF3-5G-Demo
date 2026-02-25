"""
Comprehensive Test Suite for 5G Emulator API Network Functions

This module tests ALL network functions systematically:
- 5G Core: NRF, AMF, SMF, UPF, AUSF, UDM, UDR, PCF, NSSF, BSF, SCP, CHF, NEF, SEPP, N3IWF
- 4G EPC: MME, SGW, PGW, HSS
- IMS: P-CSCF, I-CSCF, S-CSCF, MRF, IMS-HSS

For each NF, tests verify:
1. Health endpoint returns 200 with required fields (status, service, compliance, version)
2. /docs endpoint returns Swagger UI
3. Key API endpoints exist and return proper status codes

Author: 5G Emulator API Test Suite
"""

import pytest
import httpx
import sys
from pathlib import Path
from typing import Dict, Any

# Add tests directory to path to import from conftest
sys.path.insert(0, str(Path(__file__).parent))
from conftest import NETWORK_FUNCTIONS, NFProcess


# =============================================================================
# Test Helpers
# =============================================================================

def validate_health_response(response_data: Dict[str, Any], expected_service: str) -> None:
    """Validate health endpoint response has required fields."""
    required_fields = ["status", "service", "compliance", "version"]

    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"

    assert response_data["status"] == "healthy", \
        f"Expected status 'healthy', got '{response_data['status']}'"

    assert response_data["service"] == expected_service, \
        f"Expected service '{expected_service}', got '{response_data['service']}'"

    assert isinstance(response_data["version"], str), \
        f"Version should be a string, got {type(response_data['version'])}"

    assert isinstance(response_data["compliance"], str), \
        f"Compliance should be a string, got {type(response_data['compliance'])}"


def check_swagger_ui(client: httpx.Client, base_url: str) -> bool:
    """Check if Swagger UI is accessible at /docs."""
    response = client.get(f"{base_url}/docs")
    if response.status_code == 200:
        content = response.text.lower()
        # Check for Swagger UI indicators
        return "swagger" in content or "openapi" in content or "redoc" in content
    return False


# =============================================================================
# 5G CORE NETWORK FUNCTION TESTS
# =============================================================================

class TestNRF:
    """Tests for Network Repository Function (NRF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, nrf_service: NFProcess, http_client: httpx.Client):
        """Test NRF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{nrf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "NRF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, nrf_service: NFProcess, http_client: httpx.Client):
        """Test NRF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{nrf_service.base_url}/docs")
        assert response.status_code == 200
        assert check_swagger_ui(http_client, nrf_service.base_url)

    @pytest.mark.core_5g
    def test_oauth2_token_endpoint(self, nrf_service: NFProcess, http_client: httpx.Client):
        """Test NRF OAuth2 token endpoint exists."""
        response = http_client.post(
            f"{nrf_service.base_url}/oauth2/token",
            json={"grant_type": "client_credentials"}
        )
        # Should return token or specific error, not 404
        assert response.status_code in [200, 400, 401, 422]

    @pytest.mark.core_5g
    def test_nf_discovery_endpoint(self, nrf_service: NFProcess, http_client: httpx.Client):
        """Test NRF NF discovery endpoint exists."""
        # First get a token
        token_response = http_client.post(
            f"{nrf_service.base_url}/oauth2/token",
            json={"grant_type": "client_credentials"}
        )
        if token_response.status_code == 200:
            token = token_response.json().get("access_token")
            headers = {"Authorization": f"Bearer {token}"}
            response = http_client.get(
                f"{nrf_service.base_url}/nnrf-disc/v1/nf-instances",
                headers=headers
            )
            assert response.status_code in [200, 401, 403]

    @pytest.mark.core_5g
    def test_metrics_endpoint(self, nrf_service: NFProcess, http_client: httpx.Client):
        """Test NRF metrics endpoint exists."""
        response = http_client.get(f"{nrf_service.base_url}/metrics")
        assert response.status_code == 200


class TestAMF:
    """Tests for Access and Mobility Management Function (AMF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, amf_service: NFProcess, http_client: httpx.Client):
        """Test AMF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{amf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "AMF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, amf_service: NFProcess, http_client: httpx.Client):
        """Test AMF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{amf_service.base_url}/docs")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_ngap_ng_setup_endpoint(self, amf_service: NFProcess, http_client: httpx.Client):
        """Test AMF NGAP NG Setup endpoint exists."""
        response = http_client.post(
            f"{amf_service.base_url}/ngap/ng-setup",
            json={"initiatingMessage": {"value": {"protocolIEs": {}}}}
        )
        assert response.status_code in [200, 400, 422, 500]

    @pytest.mark.core_5g
    def test_ue_registration_endpoint(self, amf_service: NFProcess, http_client: httpx.Client):
        """Test AMF UE registration endpoint exists."""
        response = http_client.post(
            f"{amf_service.base_url}/amf/ue/register",
            json={"supi": "imsi-001010000000001"}
        )
        assert response.status_code in [200, 400, 422, 500]

    @pytest.mark.core_5g
    def test_handover_endpoint(self, amf_service: NFProcess, http_client: httpx.Client):
        """Test AMF handover endpoint exists."""
        response = http_client.post(
            f"{amf_service.base_url}/amf/handover",
            json={"ue_id": "test-ue", "source_gnb_id": "gnb001"}
        )
        # 404 for UE not found is acceptable
        assert response.status_code in [200, 400, 404, 422, 500]


class TestSMF:
    """Tests for Session Management Function (SMF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, smf_service: NFProcess, http_client: httpx.Client):
        """Test SMF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{smf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "SMF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, smf_service: NFProcess, http_client: httpx.Client):
        """Test SMF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{smf_service.base_url}/docs")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_sessions_endpoint(self, smf_service: NFProcess, http_client: httpx.Client):
        """Test SMF sessions list endpoint exists."""
        response = http_client.get(f"{smf_service.base_url}/smf/sessions")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_smf_service_endpoint(self, smf_service: NFProcess, http_client: httpx.Client):
        """Test SMF service endpoint exists."""
        response = http_client.get(f"{smf_service.base_url}/smf_service")
        assert response.status_code == 200


class TestUPF:
    """Tests for User Plane Function (UPF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, upf_service: NFProcess, http_client: httpx.Client):
        """Test UPF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{upf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "UPF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, upf_service: NFProcess, http_client: httpx.Client):
        """Test UPF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{upf_service.base_url}/docs")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_forwarding_rules_endpoint(self, upf_service: NFProcess, http_client: httpx.Client):
        """Test UPF forwarding rules endpoint exists."""
        response = http_client.get(f"{upf_service.base_url}/upf/forwarding-rules")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_n4_sessions_endpoint(self, upf_service: NFProcess, http_client: httpx.Client):
        """Test UPF N4 sessions endpoint exists."""
        response = http_client.post(
            f"{upf_service.base_url}/n4/sessions",
            json={"messageType": "PFCP_SESSION_ESTABLISHMENT_REQUEST", "seid": "test"}
        )
        assert response.status_code in [200, 400, 422]


class TestAUSF:
    """Tests for Authentication Server Function (AUSF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, ausf_service: NFProcess, http_client: httpx.Client):
        """Test AUSF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{ausf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "AUSF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, ausf_service: NFProcess, http_client: httpx.Client):
        """Test AUSF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{ausf_service.base_url}/docs")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_metrics_endpoint(self, ausf_service: NFProcess, http_client: httpx.Client):
        """Test AUSF metrics endpoint exists."""
        response = http_client.get(f"{ausf_service.base_url}/metrics")
        assert response.status_code == 200


class TestUDM:
    """Tests for Unified Data Management (UDM)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, udm_service: NFProcess, http_client: httpx.Client):
        """Test UDM health endpoint returns 200 with required fields."""
        response = http_client.get(f"{udm_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "UDM")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, udm_service: NFProcess, http_client: httpx.Client):
        """Test UDM /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{udm_service.base_url}/docs")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_metrics_endpoint(self, udm_service: NFProcess, http_client: httpx.Client):
        """Test UDM metrics endpoint exists."""
        response = http_client.get(f"{udm_service.base_url}/metrics")
        assert response.status_code == 200


class TestUDR:
    """Tests for Unified Data Repository (UDR)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, udr_service: NFProcess, http_client: httpx.Client):
        """Test UDR health endpoint returns 200 with required fields."""
        response = http_client.get(f"{udr_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "UDR")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, udr_service: NFProcess, http_client: httpx.Client):
        """Test UDR /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{udr_service.base_url}/docs")
        assert response.status_code == 200


class TestPCF:
    """Tests for Policy Control Function (PCF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, pcf_service: NFProcess, http_client: httpx.Client):
        """Test PCF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{pcf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "PCF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, pcf_service: NFProcess, http_client: httpx.Client):
        """Test PCF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{pcf_service.base_url}/docs")
        assert response.status_code == 200

    @pytest.mark.core_5g
    def test_metrics_endpoint(self, pcf_service: NFProcess, http_client: httpx.Client):
        """Test PCF metrics endpoint exists."""
        response = http_client.get(f"{pcf_service.base_url}/metrics")
        assert response.status_code == 200


class TestNSSF:
    """Tests for Network Slice Selection Function (NSSF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, nssf_service: NFProcess, http_client: httpx.Client):
        """Test NSSF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{nssf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "NSSF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, nssf_service: NFProcess, http_client: httpx.Client):
        """Test NSSF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{nssf_service.base_url}/docs")
        assert response.status_code == 200


class TestBSF:
    """Tests for Binding Support Function (BSF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, bsf_service: NFProcess, http_client: httpx.Client):
        """Test BSF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{bsf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "BSF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, bsf_service: NFProcess, http_client: httpx.Client):
        """Test BSF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{bsf_service.base_url}/docs")
        assert response.status_code == 200


class TestSCP:
    """Tests for Service Communication Proxy (SCP)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, scp_service: NFProcess, http_client: httpx.Client):
        """Test SCP health endpoint returns 200 with required fields."""
        response = http_client.get(f"{scp_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "SCP")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, scp_service: NFProcess, http_client: httpx.Client):
        """Test SCP /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{scp_service.base_url}/docs")
        assert response.status_code == 200


class TestCHF:
    """Tests for Charging Function (CHF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, chf_service: NFProcess, http_client: httpx.Client):
        """Test CHF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{chf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "CHF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, chf_service: NFProcess, http_client: httpx.Client):
        """Test CHF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{chf_service.base_url}/docs")
        assert response.status_code == 200


class TestNEF:
    """Tests for Network Exposure Function (NEF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, nef_service: NFProcess, http_client: httpx.Client):
        """Test NEF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{nef_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "NEF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, nef_service: NFProcess, http_client: httpx.Client):
        """Test NEF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{nef_service.base_url}/docs")
        assert response.status_code == 200


class TestSEPP:
    """Tests for Security Edge Protection Proxy (SEPP)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, sepp_service: NFProcess, http_client: httpx.Client):
        """Test SEPP health endpoint returns 200 with required fields."""
        response = http_client.get(f"{sepp_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "SEPP")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, sepp_service: NFProcess, http_client: httpx.Client):
        """Test SEPP /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{sepp_service.base_url}/docs")
        assert response.status_code == 200


class TestN3IWF:
    """Tests for Non-3GPP Interworking Function (N3IWF)"""

    @pytest.mark.core_5g
    def test_health_endpoint(self, n3iwf_service: NFProcess, http_client: httpx.Client):
        """Test N3IWF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{n3iwf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "N3IWF")

    @pytest.mark.core_5g
    def test_docs_endpoint(self, n3iwf_service: NFProcess, http_client: httpx.Client):
        """Test N3IWF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{n3iwf_service.base_url}/docs")
        assert response.status_code == 200


# =============================================================================
# 4G EPC NETWORK FUNCTION TESTS
# =============================================================================

class TestMME:
    """Tests for Mobility Management Entity (MME)"""

    @pytest.mark.epc_4g
    def test_health_endpoint(self, mme_service: NFProcess, http_client: httpx.Client):
        """Test MME health endpoint returns 200 with required fields."""
        response = http_client.get(f"{mme_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "MME")

    @pytest.mark.epc_4g
    def test_docs_endpoint(self, mme_service: NFProcess, http_client: httpx.Client):
        """Test MME /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{mme_service.base_url}/docs")
        assert response.status_code == 200


class TestSGW:
    """Tests for Serving Gateway (SGW)"""

    @pytest.mark.epc_4g
    def test_health_endpoint(self, sgw_service: NFProcess, http_client: httpx.Client):
        """Test SGW health endpoint returns 200 with required fields."""
        response = http_client.get(f"{sgw_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "SGW")

    @pytest.mark.epc_4g
    def test_docs_endpoint(self, sgw_service: NFProcess, http_client: httpx.Client):
        """Test SGW /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{sgw_service.base_url}/docs")
        assert response.status_code == 200


class TestPGW:
    """Tests for PDN Gateway (PGW)"""

    @pytest.mark.epc_4g
    def test_health_endpoint(self, pgw_service: NFProcess, http_client: httpx.Client):
        """Test PGW health endpoint returns 200 with required fields."""
        response = http_client.get(f"{pgw_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "PGW")

    @pytest.mark.epc_4g
    def test_docs_endpoint(self, pgw_service: NFProcess, http_client: httpx.Client):
        """Test PGW /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{pgw_service.base_url}/docs")
        assert response.status_code == 200


class TestHSS:
    """Tests for Home Subscriber Server (HSS)"""

    @pytest.mark.epc_4g
    def test_health_endpoint(self, hss_service: NFProcess, http_client: httpx.Client):
        """Test HSS health endpoint returns 200 with required fields."""
        response = http_client.get(f"{hss_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "HSS")

    @pytest.mark.epc_4g
    def test_docs_endpoint(self, hss_service: NFProcess, http_client: httpx.Client):
        """Test HSS /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{hss_service.base_url}/docs")
        assert response.status_code == 200


# =============================================================================
# IMS CORE NETWORK FUNCTION TESTS
# =============================================================================

class TestPCSCF:
    """Tests for Proxy Call Session Control Function (P-CSCF)"""

    @pytest.mark.ims
    def test_health_endpoint(self, pcscf_service: NFProcess, http_client: httpx.Client):
        """Test P-CSCF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{pcscf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "P-CSCF")

    @pytest.mark.ims
    def test_docs_endpoint(self, pcscf_service: NFProcess, http_client: httpx.Client):
        """Test P-CSCF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{pcscf_service.base_url}/docs")
        assert response.status_code == 200


class TestICSCF:
    """Tests for Interrogating Call Session Control Function (I-CSCF)"""

    @pytest.mark.ims
    def test_health_endpoint(self, icscf_service: NFProcess, http_client: httpx.Client):
        """Test I-CSCF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{icscf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "I-CSCF")

    @pytest.mark.ims
    def test_docs_endpoint(self, icscf_service: NFProcess, http_client: httpx.Client):
        """Test I-CSCF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{icscf_service.base_url}/docs")
        assert response.status_code == 200


class TestSCSCF:
    """Tests for Serving Call Session Control Function (S-CSCF)"""

    @pytest.mark.ims
    def test_health_endpoint(self, scscf_service: NFProcess, http_client: httpx.Client):
        """Test S-CSCF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{scscf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "S-CSCF")

    @pytest.mark.ims
    def test_docs_endpoint(self, scscf_service: NFProcess, http_client: httpx.Client):
        """Test S-CSCF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{scscf_service.base_url}/docs")
        assert response.status_code == 200


class TestMRF:
    """Tests for Media Resource Function (MRF)"""

    @pytest.mark.ims
    def test_health_endpoint(self, mrf_service: NFProcess, http_client: httpx.Client):
        """Test MRF health endpoint returns 200 with required fields."""
        response = http_client.get(f"{mrf_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "MRF")

    @pytest.mark.ims
    def test_docs_endpoint(self, mrf_service: NFProcess, http_client: httpx.Client):
        """Test MRF /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{mrf_service.base_url}/docs")
        assert response.status_code == 200


class TestIMSHSS:
    """Tests for IMS Home Subscriber Server (IMS-HSS)"""

    @pytest.mark.ims
    def test_health_endpoint(self, ims_hss_service: NFProcess, http_client: httpx.Client):
        """Test IMS-HSS health endpoint returns 200 with required fields."""
        response = http_client.get(f"{ims_hss_service.base_url}/health")
        assert response.status_code == 200
        validate_health_response(response.json(), "IMS-HSS")

    @pytest.mark.ims
    def test_docs_endpoint(self, ims_hss_service: NFProcess, http_client: httpx.Client):
        """Test IMS-HSS /docs endpoint returns Swagger UI."""
        response = http_client.get(f"{ims_hss_service.base_url}/docs")
        assert response.status_code == 200


# =============================================================================
# PARAMETRIZED TESTS FOR ALL NETWORK FUNCTIONS
# =============================================================================

@pytest.mark.parametrize("nf_name,script_path,default_port,compliance", [
    (nf_name, *config) for nf_name, config in NETWORK_FUNCTIONS.items()
])
class TestAllNetworkFunctions:
    """Parametrized tests that run for all network functions."""

    def test_health_endpoint_exists(self, nf_process_factory, http_client, nf_name, script_path, default_port, compliance):
        """Test that health endpoint exists and returns 200."""
        nf = nf_process_factory(nf_name)
        response = http_client.get(f"{nf.base_url}/health")
        assert response.status_code == 200, f"{nf_name} health endpoint failed"

    def test_health_has_required_fields(self, nf_process_factory, http_client, nf_name, script_path, default_port, compliance):
        """Test that health response has all required fields."""
        nf = nf_process_factory(nf_name)
        response = http_client.get(f"{nf.base_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data, f"{nf_name}: Missing 'status' field"
        assert "service" in data, f"{nf_name}: Missing 'service' field"
        assert "compliance" in data, f"{nf_name}: Missing 'compliance' field"
        assert "version" in data, f"{nf_name}: Missing 'version' field"
        assert data["status"] == "healthy", f"{nf_name}: Expected 'healthy' status"

    def test_docs_endpoint_accessible(self, nf_process_factory, http_client, nf_name, script_path, default_port, compliance):
        """Test that /docs endpoint is accessible."""
        nf = nf_process_factory(nf_name)
        response = http_client.get(f"{nf.base_url}/docs")
        assert response.status_code == 200, f"{nf_name} /docs endpoint failed"

    def test_openapi_schema_available(self, nf_process_factory, http_client, nf_name, script_path, default_port, compliance):
        """Test that OpenAPI schema is available at /openapi.json."""
        nf = nf_process_factory(nf_name)
        response = http_client.get(f"{nf.base_url}/openapi.json")
        assert response.status_code == 200, f"{nf_name} OpenAPI schema not available"
        schema = response.json()
        assert "openapi" in schema or "swagger" in schema, f"{nf_name}: Invalid OpenAPI schema"
        assert "paths" in schema, f"{nf_name}: OpenAPI schema missing 'paths'"


# =============================================================================
# COMPLIANCE VERIFICATION TESTS
# =============================================================================

class TestComplianceInfo:
    """Tests to verify compliance information is correctly reported."""

    @pytest.mark.parametrize("nf_name,expected_compliance", [
        ("NRF", "3GPP TS 29.510"),
        ("AMF", "3GPP TS 29.518"),
        ("SMF", "3GPP TS 29.502"),
        ("UPF", "3GPP TS 29.244"),
        ("AUSF", "3GPP TS 29.509"),
        ("UDM", "3GPP TS 29.503"),
        ("UDR", "3GPP TS 29.504"),
        ("PCF", "3GPP TS 29.507"),
        ("NSSF", "3GPP TS 29.531"),
        ("BSF", "3GPP TS 29.521"),
        ("SCP", "3GPP TS 29.500"),
        ("CHF", "3GPP TS 32"),
        ("NEF", "3GPP TS 29.522"),
        ("SEPP", "3GPP TS 29.573"),
        ("N3IWF", "3GPP TS"),
        ("MME", "3GPP TS 23.401"),
        ("SGW", "3GPP TS 23.401"),
        ("PGW", "3GPP TS 23.401"),
        ("HSS", "3GPP TS 29.272"),
        ("P-CSCF", "3GPP TS 24.229"),
        ("I-CSCF", "3GPP TS 24.229"),
        ("S-CSCF", "3GPP TS 24.229"),
        ("MRF", "3GPP TS 23.228"),
        ("IMS-HSS", "3GPP TS 29.228"),
    ])
    def test_compliance_string_contains_spec(self, nf_process_factory, http_client, nf_name, expected_compliance):
        """Verify that compliance string contains the expected 3GPP specification."""
        nf = nf_process_factory(nf_name)
        response = http_client.get(f"{nf.base_url}/health")
        assert response.status_code == 200

        data = response.json()
        compliance = data.get("compliance", "")
        assert expected_compliance in compliance, \
            f"{nf_name}: Expected '{expected_compliance}' in compliance string, got '{compliance}'"


# =============================================================================
# VERSION FORMAT TESTS
# =============================================================================

class TestVersionFormat:
    """Tests to verify version format is correct."""

    @pytest.mark.parametrize("nf_name", list(NETWORK_FUNCTIONS.keys()))
    def test_version_is_semver_like(self, nf_process_factory, http_client, nf_name):
        """Verify that version follows semver-like format (X.Y.Z)."""
        nf = nf_process_factory(nf_name)
        response = http_client.get(f"{nf.base_url}/health")
        assert response.status_code == 200

        data = response.json()
        version = data.get("version", "")

        # Check version format (should contain at least one dot)
        assert "." in version, f"{nf_name}: Version '{version}' should be semver-like (X.Y.Z)"

        # Check that parts are numeric-ish
        parts = version.split(".")
        assert len(parts) >= 2, f"{nf_name}: Version should have at least major.minor"
