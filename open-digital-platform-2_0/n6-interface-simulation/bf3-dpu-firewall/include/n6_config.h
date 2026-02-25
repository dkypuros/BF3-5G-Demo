/**
 * @file n6_config.h
 * @brief Configuration constants for N6 Interface Firewall
 * 
 * Contains compile-time configuration options for the BlueField-3
 * N6 firewall application. Modify these values based on deployment
 * requirements and hardware specifications.
 * 
 * @author NVIDIA DOCA Team
 * @version 2.6.0
 * @date 2024
 */

#ifndef N6_CONFIG_H
#define N6_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================
 * Hardware Platform Configuration
 * ======================================== */

/* BlueField-3 DPU Specifications */
#define BF3_MAX_PORTS                   2       /* Maximum ports per DPU */
#define BF3_MAX_QUEUES_PER_PORT         32      /* Hardware queue limit */
#define BF3_MAX_FLOW_RULES              65536   /* Hardware flow table size */
#define BF3_MAX_METER_RULES             8192    /* Hardware meter table size */
#define BF3_CACHE_LINE_SIZE             64      /* CPU cache line size */
#define BF3_HUGEPAGE_SIZE               (2 * 1024 * 1024)  /* 2MB hugepages */

/* Network Interface Configuration */
#define N6_DEFAULT_MTU                  1500    /* Standard Ethernet MTU */
#define N6_JUMBO_MTU                    9000    /* Jumbo frame MTU */
#define N6_MAX_PACKET_SIZE              (N6_JUMBO_MTU + 256)  /* Buffer size */

/* ========================================
 * DOCA Flow Configuration
 * ======================================== */

/* Flow Processing Parameters */
#define DOCA_FLOW_MAX_PIPES             256     /* Maximum flow pipes */
#define DOCA_FLOW_MAX_ENTRIES_PER_PIPE  1024    /* Entries per pipe */
#define DOCA_FLOW_AGING_TIMEOUT_SEC     300     /* Flow aging timeout */
#define DOCA_FLOW_QUEUE_DEPTH           1024    /* Hardware queue depth */

/* Hardware Steering Mode */
#define DOCA_FLOW_HWS_MODE              "vnf,hws,isolated"
#define DOCA_FLOW_SW_MODE               "vnf,sw"

/* Resource Allocation */
#define DOCA_FLOW_COUNTERS              4096    /* Flow counters */
#define DOCA_FLOW_METERS                2048    /* Rate limit meters */
#define DOCA_FLOW_SHARED_COUNTERS       512     /* Shared statistics */
#define DOCA_FLOW_ENCAP_ACTIONS         256     /* Encapsulation actions */
#define DOCA_FLOW_MODIFY_ACTIONS        512     /* Packet modification actions */

/* ========================================
 * N6 Interface Specific Configuration
 * ======================================== */

/* 5G Network Constants */
#define N6_DEFAULT_GTP_PORT             2152    /* GTP-U port */
#define N6_DEFAULT_HTTP_PORT            80      /* Standard HTTP */
#define N6_DEFAULT_HTTPS_PORT           443     /* Standard HTTPS */
#define N6_DEFAULT_DNS_PORT             53      /* DNS service */

/* Traffic Classification */
#define N6_QOS_CLASS_CONVERSATIONAL     1       /* Voice calls */
#define N6_QOS_CLASS_STREAMING          2       /* Video streaming */
#define N6_QOS_CLASS_INTERACTIVE        3       /* Web browsing */
#define N6_QOS_CLASS_BACKGROUND         4       /* Background data */

/* Rate Limiting */
#define N6_RATE_LIMIT_MIN_PPS           1000    /* Minimum packets/sec */
#define N6_RATE_LIMIT_MAX_PPS           1000000 /* Maximum packets/sec */
#define N6_RATE_LIMIT_MIN_BPS           (64 * 1024)     /* 64 KB/s */
#define N6_RATE_LIMIT_MAX_BPS           (100 * 1024 * 1024)  /* 100 MB/s */

/* ========================================
 * Security and Firewall Configuration
 * ======================================== */

/* Access Control */
#define N6_MAX_BLOCKED_IPS              1024    /* Maximum blocked IPs */
#define N6_MAX_ALLOWED_IPS              2048    /* Maximum allowed IPs */
#define N6_MAX_RATE_LIMITED_IPS         512     /* Maximum rate limited IPs */

/* Attack Detection */
#define N6_DOS_THRESHOLD_PPS            10000   /* DDoS detection threshold */
#define N6_PORT_SCAN_THRESHOLD          100     /* Port scan detection */
#define N6_CONN_RATE_THRESHOLD          1000    /* Connection rate limit */

/* Logging Configuration */
#define N6_LOG_BUFFER_SIZE              (4 * 1024)      /* Log buffer size */
#define N6_MAX_LOG_FILES                10              /* Log rotation */
#define N6_MAX_LOG_FILE_SIZE            (100 * 1024 * 1024)  /* 100MB per file */

/* ========================================
 * Performance and Memory Configuration
 * ======================================== */

/* Memory Pool Configuration */
#define N6_MBUF_POOL_SIZE               8192    /* Packet buffer pool */
#define N6_MBUF_CACHE_SIZE              512     /* Per-core cache */
#define N6_MEMPOOL_ALIGNMENT            64      /* Memory alignment */

/* CPU and Threading */
#define N6_MAX_WORKER_THREADS           8       /* Maximum worker threads */
#define N6_DEFAULT_WORKER_THREADS       4       /* Default worker threads */
#define N6_STATS_UPDATE_INTERVAL_SEC    5       /* Statistics update rate */

/* Burst Processing */
#define N6_RX_BURST_SIZE                32      /* RX burst size */
#define N6_TX_BURST_SIZE                32      /* TX burst size */
#define N6_PREFETCH_DISTANCE            3       /* Cache prefetch distance */

/* ========================================
 * Monitoring and Telemetry
 * ======================================== */

/* Statistics Collection */
#define N6_STATS_HISTORY_DEPTH          3600    /* 1 hour of history */
#define N6_PERF_COUNTER_UPDATE_MS       100     /* Performance counter rate */

/* Health Monitoring */
#define N6_HEALTH_CHECK_INTERVAL_SEC    30      /* Health check interval */
#define N6_MAX_CONSECUTIVE_FAILURES     3       /* Failure threshold */

/* External Integration */
#define N6_PROMETHEUS_DEFAULT_PORT      9090    /* Prometheus metrics */
#define N6_GRAFANA_DEFAULT_PORT         3000    /* Grafana dashboard */
#define N6_SYSLOG_DEFAULT_PORT          514     /* Syslog server */

/* ========================================
 * Development and Debug Configuration
 * ======================================== */

#ifdef DEBUG
    #define N6_DEBUG_ENABLED            1
    #define N6_VERBOSE_LOGGING          1
    #define N6_PACKET_DUMP_ENABLED      1
    #define N6_ASSERT_ENABLED           1
#else
    #define N6_DEBUG_ENABLED            0
    #define N6_VERBOSE_LOGGING          0
    #define N6_PACKET_DUMP_ENABLED      0
    #define N6_ASSERT_ENABLED           0
#endif

/* Debug Packet Capture */
#define N6_MAX_CAPTURED_PACKETS         1000    /* Packet capture limit */
#define N6_PACKET_CAPTURE_SIZE          256     /* Bytes per packet */

/* Test Mode Configuration */
#ifdef TEST_MODE
    #define N6_TEST_MODE_ENABLED        1
    #define N6_SYNTHETIC_TRAFFIC        1
    #define N6_BYPASS_HARDWARE          1
#else
    #define N6_TEST_MODE_ENABLED        0
    #define N6_SYNTHETIC_TRAFFIC        0
    #define N6_BYPASS_HARDWARE          0
#endif

/* ========================================
 * Version and Build Information
 * ======================================== */

#define N6_FIREWALL_VERSION_MAJOR       2
#define N6_FIREWALL_VERSION_MINOR       6
#define N6_FIREWALL_VERSION_PATCH       0
#define N6_FIREWALL_VERSION_BUILD       __DATE__ " " __TIME__

#define N6_FIREWALL_COPYRIGHT           "Copyright (c) 2024 NVIDIA Corporation"
#define N6_FIREWALL_LICENSE             "Proprietary - NVIDIA DOCA License"

/* Version string generation */
#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

#define N6_FIREWALL_VERSION_STRING \
    STRINGIFY(N6_FIREWALL_VERSION_MAJOR) "." \
    STRINGIFY(N6_FIREWALL_VERSION_MINOR) "." \
    STRINGIFY(N6_FIREWALL_VERSION_PATCH)

/* ========================================
 * Deployment Environment Detection
 * ======================================== */

/* Detect BlueField DPU environment */
#if defined(__aarch64__) && defined(__linux__)
    #define N6_TARGET_BLUEFIELD     1
    #define N6_TARGET_X86_HOST      0
#elif defined(__x86_64__) && defined(__linux__)
    #define N6_TARGET_BLUEFIELD     0
    #define N6_TARGET_X86_HOST      1
#else
    #define N6_TARGET_BLUEFIELD     0
    #define N6_TARGET_X86_HOST      0
    #warning "Unknown target platform - using generic configuration"
#endif

/* Platform-specific optimizations */
#if N6_TARGET_BLUEFIELD
    #define N6_USE_NEON_SIMD        1       /* ARM NEON optimizations */
    #define N6_CACHE_COHERENCY      1       /* Cache coherent I/O */
    #define N6_NUMA_AWARE           1       /* NUMA topology awareness */
#else
    #define N6_USE_AVX2_SIMD        1       /* x86 AVX2 optimizations */
    #define N6_CACHE_COHERENCY      0       /* Non-coherent I/O */
    #define N6_NUMA_AWARE           0       /* Single NUMA node */
#endif

/* ========================================
 * Runtime Configuration Validation
 * ======================================== */

/* Compile-time assertions */
#if N6_FIREWALL_QUEUES > BF3_MAX_QUEUES_PER_PORT
    #error "N6_FIREWALL_QUEUES exceeds hardware limit"
#endif

#if N6_MAX_BLOCKED_PORTS > BF3_MAX_FLOW_RULES
    #error "N6_MAX_BLOCKED_PORTS exceeds hardware flow table size"
#endif

#if DOCA_FLOW_COUNTERS > BF3_MAX_FLOW_RULES
    #error "DOCA_FLOW_COUNTERS exceeds hardware capacity"
#endif

#ifdef __cplusplus
}
#endif

#endif /* N6_CONFIG_H */