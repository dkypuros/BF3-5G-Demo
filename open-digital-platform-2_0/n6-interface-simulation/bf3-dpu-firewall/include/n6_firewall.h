/**
 * @file n6_firewall.h
 * @brief Header file for N6 Interface Firewall application
 * 
 * Contains data structures, constants, and function declarations
 * for the production N6 firewall running on BlueField-3 DPU.
 * 
 * @author NVIDIA DOCA Team
 * @version 2.6.0
 * @date 2024
 */

#ifndef N6_FIREWALL_H
#define N6_FIREWALL_H

#include <stdint.h>
#include <stdbool.h>
#include <doca_flow.h>
#include <doca_devemu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Application Constants */
#define N6_FIREWALL_VERSION         "2.6.0"
#define N6_FIREWALL_BUILD_DATE      __DATE__ " " __TIME__

/* Hardware Configuration */
#define N6_FIREWALL_QUEUES          16      /* Number of hardware queues */
#define N6_FIREWALL_COUNTERS        1024    /* Hardware flow counters */
#define N6_FIREWALL_METERS          512     /* Hardware rate limiters */
#define N6_FIREWALL_SHARED_COUNTERS 256     /* Shared statistics counters */

/* Port Configuration */
#define N6_UPLINK_PORT_ID           0       /* Port toward UPF */
#define N6_DOWNLINK_PORT_ID         1       /* Port toward Data Network */

/* Default Values */
#define N6_DEFAULT_BLOCKED_PORT     8001    /* Default port to block */
#define N6_MAX_BLOCKED_PORTS        128     /* Maximum firewall rules */
#define DEFAULT_TIMEOUT_US          1000000 /* 1 second timeout */

/* Network Protocol Constants */
#define N6_ETH_TYPE_IPV4            0x0800
#define N6_ETH_TYPE_IPV6            0x86DD
#define N6_IP_PROTO_TCP             6
#define N6_IP_PROTO_UDP             17
#define N6_IP_PROTO_ICMP            1

/* Firewall Rule Types */
typedef enum {
    N6_RULE_BLOCK_PORT = 0,     /* Block specific TCP/UDP port */
    N6_RULE_BLOCK_IP,           /* Block specific IP address */
    N6_RULE_RATE_LIMIT,         /* Rate limit traffic */
    N6_RULE_REDIRECT,           /* Redirect traffic */
    N6_RULE_LOG_ONLY,           /* Log traffic without action */
    N6_RULE_MAX
} n6_rule_type_t;

/* Firewall Rule Priority */
typedef enum {
    N6_PRIORITY_CRITICAL = 0,   /* Highest priority (security rules) */
    N6_PRIORITY_HIGH,           /* High priority (service rules) */
    N6_PRIORITY_NORMAL,         /* Normal priority (general rules) */
    N6_PRIORITY_LOW,            /* Low priority (logging rules) */
    N6_PRIORITY_MAX
} n6_rule_priority_t;

/* Firewall Rule Structure */
struct n6_firewall_rule {
    uint32_t rule_id;                   /* Unique rule identifier */
    n6_rule_type_t type;                /* Rule type */
    n6_rule_priority_t priority;        /* Rule priority */
    
    /* Match Criteria */
    struct {
        uint32_t src_ip;                /* Source IP (0 = wildcard) */
        uint32_t dst_ip;                /* Destination IP (0 = wildcard) */
        uint16_t src_port;              /* Source port (0 = wildcard) */
        uint16_t dst_port;              /* Destination port (0 = wildcard) */
        uint8_t protocol;               /* IP protocol (0 = wildcard) */
        uint16_t vlan_id;               /* VLAN ID (0 = wildcard) */
    } match;
    
    /* Action Configuration */
    struct {
        bool drop;                      /* Drop packets */
        bool forward;                   /* Forward to specific port */
        uint16_t redirect_port;         /* Port for redirection */
        uint32_t rate_limit_pps;        /* Packets per second limit */
        bool log_enabled;               /* Enable logging for this rule */
    } action;
    
    /* Statistics */
    struct {
        uint64_t packets_matched;       /* Total packets matched */
        uint64_t bytes_matched;         /* Total bytes matched */
        uint64_t packets_dropped;       /* Total packets dropped */
        uint64_t last_match_timestamp;  /* Last match timestamp */
    } stats;
    
    /* DOCA Flow Entry */
    struct doca_flow_pipe_entry *flow_entry;
    
    bool active;                        /* Rule is active */
    char description[256];              /* Human-readable description */
};

/* Main Application Context */
struct n6_firewall_ctx {
    /* DOCA Contexts */
    struct doca_dev *doca_dev;              /* DOCA device handle */
    struct doca_flow_ctx *flow_ctx;         /* DOCA Flow context */
    struct doca_devemu *devemu_ctx;         /* DOCA DevEmu context */
    
    /* DOCA Flow Ports */
    struct doca_flow_port *uplink_port;     /* UPF-facing port */
    struct doca_flow_port *downlink_port;   /* DN-facing port */
    
    /* Firewall Pipes */
    struct doca_flow_pipe *firewall_pipe;   /* Main firewall pipe */
    struct doca_flow_pipe *stats_pipe;      /* Statistics collection pipe */
    struct doca_flow_pipe *logging_pipe;    /* Logging pipe */
    
    /* Firewall Rules */
    struct n6_firewall_rule rules[N6_MAX_BLOCKED_PORTS];
    uint32_t nb_rules;                      /* Number of active rules */
    
    /* Legacy Arrays (for backward compatibility) */
    struct doca_flow_pipe_entry *blocked_port_entries[N6_MAX_BLOCKED_PORTS];
    uint16_t blocked_ports[N6_MAX_BLOCKED_PORTS];
    uint32_t nb_blocked_ports;
    
    /* Runtime Configuration */
    struct {
        bool verbose_mode;              /* Enable verbose logging */
        bool stats_enabled;             /* Enable statistics collection */
        uint32_t stats_interval;        /* Statistics update interval (seconds) */
        uint32_t log_level;             /* Logging level */
        char config_file[256];          /* Configuration file path */
    } config;
    
    /* Performance Counters */
    struct {
        uint64_t total_packets_processed;   /* Total packets processed */
        uint64_t total_packets_dropped;     /* Total packets dropped */
        uint64_t total_packets_forwarded;   /* Total packets forwarded */
        uint64_t total_bytes_processed;     /* Total bytes processed */
        uint64_t uptime_seconds;            /* Application uptime */
        uint32_t rules_hit_per_second;      /* Rules matched per second */
    } perf_stats;
};

/* Configuration Structure */
struct n6_firewall_config {
    /* Network Configuration */
    char uplink_interface[64];          /* Uplink interface name */
    char downlink_interface[64];        /* Downlink interface name */
    uint16_t uplink_port_id;            /* Uplink port ID */
    uint16_t downlink_port_id;          /* Downlink port ID */
    
    /* Hardware Acceleration */
    bool hw_offload_enabled;            /* Enable hardware offloading */
    uint32_t flow_table_size;           /* Hardware flow table size */
    uint32_t meter_table_size;          /* Hardware meter table size */
    
    /* Logging and Monitoring */
    bool syslog_enabled;                /* Enable syslog */
    char syslog_server[256];            /* Syslog server address */
    uint16_t syslog_port;               /* Syslog server port */
    bool prometheus_enabled;            /* Enable Prometheus metrics */
    uint16_t prometheus_port;           /* Prometheus metrics port */
    
    /* Security */
    bool secure_mode;                   /* Enable secure mode */
    char admin_key_file[256];           /* Admin key file path */
    uint32_t max_rules_per_minute;      /* Rate limit for rule changes */
};

/* Function Declarations */

/**
 * @brief Initialize N6 firewall application
 * 
 * @param ctx Application context to initialize
 * @param config Configuration parameters
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t n6_firewall_init(struct n6_firewall_ctx *ctx, 
                               const struct n6_firewall_config *config);

/**
 * @brief Add a new firewall rule
 * 
 * @param ctx Application context
 * @param rule Rule configuration
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t n6_firewall_add_rule(struct n6_firewall_ctx *ctx,
                                   const struct n6_firewall_rule *rule);

/**
 * @brief Remove a firewall rule by ID
 * 
 * @param ctx Application context
 * @param rule_id Rule ID to remove
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t n6_firewall_remove_rule(struct n6_firewall_ctx *ctx,
                                      uint32_t rule_id);

/**
 * @brief Update firewall statistics
 * 
 * @param ctx Application context
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t n6_firewall_update_stats(struct n6_firewall_ctx *ctx);

/**
 * @brief Export firewall statistics in JSON format
 * 
 * @param ctx Application context
 * @param buffer Buffer to write JSON data
 * @param buffer_size Size of the buffer
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t n6_firewall_export_stats_json(const struct n6_firewall_ctx *ctx,
                                            char *buffer, size_t buffer_size);

/**
 * @brief Load configuration from file
 * 
 * @param config_file Path to configuration file
 * @param config Configuration structure to populate
 * @return DOCA_SUCCESS on success, error code otherwise
 */
doca_error_t n6_firewall_load_config(const char *config_file,
                                      struct n6_firewall_config *config);

/**
 * @brief Cleanup firewall resources
 * 
 * @param ctx Application context to cleanup
 */
void n6_firewall_cleanup(struct n6_firewall_ctx *ctx);

/**
 * @brief Get firewall version information
 * 
 * @param version_info Buffer to store version string
 * @param buffer_size Size of the buffer
 */
void n6_firewall_get_version(char *version_info, size_t buffer_size);

/**
 * @brief Validate firewall rule configuration
 * 
 * @param rule Rule to validate
 * @return true if rule is valid, false otherwise
 */
bool n6_firewall_validate_rule(const struct n6_firewall_rule *rule);

/* Inline Helper Functions */

/**
 * @brief Convert rule priority to string
 */
static inline const char *
n6_priority_to_string(n6_rule_priority_t priority)
{
    switch (priority) {
    case N6_PRIORITY_CRITICAL: return "CRITICAL";
    case N6_PRIORITY_HIGH:     return "HIGH";
    case N6_PRIORITY_NORMAL:   return "NORMAL";
    case N6_PRIORITY_LOW:      return "LOW";
    default:                   return "UNKNOWN";
    }
}

/**
 * @brief Convert rule type to string
 */
static inline const char *
n6_rule_type_to_string(n6_rule_type_t type)
{
    switch (type) {
    case N6_RULE_BLOCK_PORT:  return "BLOCK_PORT";
    case N6_RULE_BLOCK_IP:    return "BLOCK_IP";
    case N6_RULE_RATE_LIMIT:  return "RATE_LIMIT";
    case N6_RULE_REDIRECT:    return "REDIRECT";
    case N6_RULE_LOG_ONLY:    return "LOG_ONLY";
    default:                  return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* N6_FIREWALL_H */