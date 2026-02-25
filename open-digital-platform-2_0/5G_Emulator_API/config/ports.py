# File location: 5G_Emulator_API/config/ports.py
# Network Function Port Assignments
# Centralized to avoid conflicts and ensure consistency

NF_PORTS = {
    # 5G Core
    "nrf": 8000,
    "amf": 9000,
    "smf": 9001,
    "upf": 9002,
    "ausf": 9003,
    "udm": 9004,
    "udr": 9005,
    "udsf": 9006,
    "pcf": 9007,
    "nssf": 9010,
    "bsf": 9011,
    "scp": 9012,
    "chf": 9013,
    "sepp": 9014,
    "n3iwf": 9015,
    "nef": 9016,

    # 4G EPC
    "mme": 9020,
    "sgw": 9021,
    "pgw": 9022,
    "hss": 9023,

    # IMS
    "pcscf": 9030,
    "icscf": 9031,
    "scscf": 9032,
    "mrf": 9033,
    "ims_hss": 9040,

    # RAN
    "gnb": 9100,
    "cu": 9101,
    "du": 9102,
    "rru": 9103,

    # O-RAN RIC
    "near_rt_ric": 8095,
    "non_rt_ric": 8096,

    # O-RAN O2 Interface
    "smo": 8097,
    "o2_ims": 8098,
    "o2_dms": 8099,

    # ETSI Management
    "rnis": 8092,
    "vnfm": 8093,
    "zsm": 8094,
}


def get_port(nf_name: str) -> int:
    """Get the default port for a network function."""
    return NF_PORTS.get(nf_name.lower())
