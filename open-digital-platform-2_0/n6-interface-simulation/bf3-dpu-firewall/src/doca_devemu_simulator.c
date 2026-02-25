/**
 * @file doca_devemu_simulator.c
 * @brief DOCA DevEmu Simulator for BlueField-3 DPU Testing
 * 
 * This simulator implements a software emulation of NVIDIA DOCA DevEmu APIs,
 * allowing development and testing of DOCA applications without physical
 * BlueField-3 hardware. It provides functional simulation of the key APIs
 * while maintaining the same interface as the real DOCA SDK.
 * 
 * @author NVIDIA DOCA Team (Simulated)
 * @version 2.6.0
 * @date 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <errno.h>

/* Simulated DOCA Headers */
#define DOCA_SUCCESS 0
#define DOCA_ERROR_NOT_FOUND -1
#define DOCA_ERROR_NO_MEMORY -2
#define DOCA_ERROR_INVALID_VALUE -3
#define DOCA_ERROR_INITIALIZATION -4
#define DOCA_ERROR_TIME_OUT -5
#define DOCA_ERROR_SHUTDOWN -6
#define DOCA_ERROR_UNEXPECTED -7

/* DOCA Flow Constants */
#define DOCA_FLOW_PIPE_DOMAIN_DEFAULT 0
#define DOCA_FLOW_PIPE_BASIC 1
#define DOCA_FLOW_L3_TYPE_IP4 0x0800
#define DOCA_FLOW_L4_TYPE_EXT_TCP 6
#define DOCA_FLOW_ACTION_DROP 1
#define DOCA_FLOW_ACTION_FORWARD 2
#define DOCA_FLOW_FWD_PORT 1
#define DOCA_FLOW_NO_WAIT 0
#define DEFAULT_TIMEOUT_US 1000000

/* Logging Macros */
#define DOCA_LOG_LEVEL_DEBUG 0
#define DOCA_LOG_LEVEL_INFO 1
#define DOCA_LOG_LEVEL_WARN 2
#define DOCA_LOG_LEVEL_ERROR 3

#define DOCA_LOG_INFO(...) doca_log_print(DOCA_LOG_LEVEL_INFO, __VA_ARGS__)
#define DOCA_LOG_ERR(...) doca_log_print(DOCA_LOG_LEVEL_ERROR, __VA_ARGS__)
#define DOCA_LOG_DEBUG(...) doca_log_print(DOCA_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define DOCA_LOG_WARN(...) doca_log_print(DOCA_LOG_LEVEL_WARN, __VA_ARGS__)

#define DOCA_LOG_REGISTER(name) static const char *log_module = #name

/* Type Definitions */
typedef int doca_error_t;

/* Simulated DOCA Structures */
struct doca_dev {
    char name[256];
    uint32_t device_id;
    bool is_bluefield;
    uint32_t port_count;
    uint64_t capabilities;
};

struct doca_flow_ctx {
    bool initialized;
    uint32_t pipe_count;
    uint32_t entry_count;
    uint64_t packets_processed;
    uint64_t packets_dropped;
};

struct doca_devemu {
    struct doca_dev *device;
    bool emulation_active;
    uint32_t virtual_functions;
    pthread_t emulation_thread;
};

struct doca_flow_cfg {
    uint32_t pipe_queues;
    const char *mode_args;
    struct {
        uint32_t nb_counters;
        uint32_t nb_meters;
        uint32_t nb_shared_counters;
    } resource;
};

struct doca_flow_port_cfg {
    uint16_t port_id;
    int type;
    char devargs[256];
};

struct doca_flow_port {
    uint16_t port_id;
    bool is_active;
    uint64_t rx_packets;
    uint64_t tx_packets;
    uint64_t rx_bytes;
    uint64_t tx_bytes;
};

struct doca_flow_match {
    struct {
        uint16_t l3_type;
        uint8_t l4_type_ext;
        struct {
            uint32_t src_ip;
            uint32_t dst_ip;
        } ip4;
        struct {
            struct {
                uint16_t src_port;
                uint16_t dst_port;
            } l4_port;
        } tcp;
    } outer;
};

struct doca_flow_actions {
    uint32_t action_type;
    bool drop;
};

struct doca_flow_fwd {
    uint32_t type;
    uint16_t port_id;
};

struct doca_flow_pipe_cfg {
    struct {
        char name[256];
        uint32_t type;
        bool is_root;
        uint32_t nb_actions;
        uint32_t domain;
    } attr;
    struct doca_flow_port *port;
    struct doca_flow_match *match;
    struct doca_flow_match *match_mask;
    struct doca_flow_actions **actions;
};

struct doca_flow_pipe {
    char name[256];
    uint32_t pipe_id;
    struct doca_flow_port *port;
    uint32_t entry_count;
    bool is_active;
};

struct doca_flow_pipe_entry {
    uint32_t entry_id;
    struct doca_flow_pipe *pipe;
    struct doca_flow_match match;
    struct doca_flow_actions actions;
    uint64_t packets_matched;
    uint64_t bytes_matched;
};

struct doca_flow_query {
    uint64_t total_pkts;
    uint64_t total_bytes;
};

/* Global Simulator State */
static struct {
    bool initialized;
    uint32_t log_level;
    struct doca_dev devices[4];
    uint32_t device_count;
    struct doca_flow_ctx flow_ctx;
    struct doca_flow_port ports[16];
    uint32_t port_count;
    struct doca_flow_pipe pipes[256];
    uint32_t pipe_count;
    struct doca_flow_pipe_entry entries[65536];
    uint32_t entry_count;
    pthread_mutex_t lock;
    bool packet_simulator_running;
    pthread_t packet_sim_thread;
} simulator = {
    .initialized = false,
    .log_level = DOCA_LOG_LEVEL_INFO,
    .device_count = 0,
    .port_count = 0,
    .pipe_count = 0,
    .entry_count = 0,
    .packet_simulator_running = false
};

/* Utility Functions */
static uint16_t rte_cpu_to_be_16(uint16_t val) {
    return htons(val);
}

static void doca_log_print(int level, const char *fmt, ...) {
    if (level < simulator.log_level) return;
    
    const char *level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    time_t now = time(NULL);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    printf("[%s] [DEVEMU] [%s] ", time_str, level_str[level]);
    
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    printf("\n");
}

static const char* doca_error_get_descr(doca_error_t error) {
    switch (error) {
        case DOCA_SUCCESS: return "Success";
        case DOCA_ERROR_NOT_FOUND: return "Not found";
        case DOCA_ERROR_NO_MEMORY: return "No memory";
        case DOCA_ERROR_INVALID_VALUE: return "Invalid value";
        case DOCA_ERROR_INITIALIZATION: return "Initialization error";
        case DOCA_ERROR_TIME_OUT: return "Timeout";
        case DOCA_ERROR_SHUTDOWN: return "Shutdown";
        default: return "Unknown error";
    }
}

/* Packet Simulator Thread */
static void* packet_simulator(void *arg) {
    (void)arg;
    DOCA_LOG_INFO("Packet simulator thread started");
    
    while (simulator.packet_simulator_running) {
        pthread_mutex_lock(&simulator.lock);
        
        /* Simulate packet processing */
        for (uint32_t i = 0; i < simulator.port_count; i++) {
            if (simulator.ports[i].is_active) {
                /* Simulate random packet arrival */
                if (rand() % 100 < 30) {  /* 30% chance of packet */
                    simulator.ports[i].rx_packets++;
                    simulator.ports[i].rx_bytes += 64 + rand() % 1436;  /* Random packet size */
                    simulator.flow_ctx.packets_processed++;
                    
                    /* Check firewall rules */
                    bool dropped = false;
                    for (uint32_t j = 0; j < simulator.entry_count; j++) {
                        if (simulator.entries[j].actions.drop && rand() % 100 < 10) {
                            /* 10% chance this packet matches drop rule */
                            simulator.entries[j].packets_matched++;
                            simulator.entries[j].bytes_matched += 64 + rand() % 1436;
                            simulator.flow_ctx.packets_dropped++;
                            dropped = true;
                            break;
                        }
                    }
                    
                    if (!dropped) {
                        simulator.ports[i].tx_packets++;
                        simulator.ports[i].tx_bytes += 64 + rand() % 1436;
                    }
                }
            }
        }
        
        pthread_mutex_unlock(&simulator.lock);
        usleep(10000);  /* 10ms sleep */
    }
    
    DOCA_LOG_INFO("Packet simulator thread stopped");
    return NULL;
}

/* DOCA Log API Implementation */
doca_error_t doca_log_backend_create_standard(void) {
    simulator.log_level = DOCA_LOG_LEVEL_INFO;
    DOCA_LOG_INFO("DOCA DevEmu Simulator initialized - log backend created");
    return DOCA_SUCCESS;
}

void doca_log_level_set_global(int level) {
    simulator.log_level = level;
    DOCA_LOG_INFO("Global log level set to %d", level);
}

/* DOCA Device API Implementation */
doca_error_t doca_dev_inventory_get(struct doca_dev ***dev_list, uint32_t *nb_devs) {
    if (!dev_list || !nb_devs) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    
    /* Create simulated BlueField-3 devices */
    if (simulator.device_count == 0) {
        /* Simulate BlueField-3 DPU */
        strcpy(simulator.devices[0].name, "BlueField-3 DPU Simulator");
        simulator.devices[0].device_id = 0xBF3000;
        simulator.devices[0].is_bluefield = true;
        simulator.devices[0].port_count = 2;
        simulator.devices[0].capabilities = 0xFFFFFFFF;  /* All capabilities */
        simulator.device_count = 1;
        
        DOCA_LOG_INFO("Created simulated BlueField-3 DPU device");
    }
    
    /* Allocate device list */
    static struct doca_dev *device_ptrs[4];
    for (uint32_t i = 0; i < simulator.device_count; i++) {
        device_ptrs[i] = &simulator.devices[i];
    }
    
    *dev_list = device_ptrs;
    *nb_devs = simulator.device_count;
    
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("Device inventory: %u devices found", *nb_devs);
    return DOCA_SUCCESS;
}

/* DOCA Flow API Implementation */
doca_error_t doca_flow_init(const struct doca_flow_cfg *cfg) {
    if (!cfg) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    
    if (simulator.flow_ctx.initialized) {
        pthread_mutex_unlock(&simulator.lock);
        return DOCA_ERROR_INITIALIZATION;
    }
    
    simulator.flow_ctx.initialized = true;
    simulator.flow_ctx.pipe_count = 0;
    simulator.flow_ctx.entry_count = 0;
    simulator.flow_ctx.packets_processed = 0;
    simulator.flow_ctx.packets_dropped = 0;
    
    /* Start packet simulator */
    simulator.packet_simulator_running = true;
    pthread_create(&simulator.packet_sim_thread, NULL, packet_simulator, NULL);
    
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("DOCA Flow initialized with %u queues, mode: %s", 
                  cfg->pipe_queues, cfg->mode_args);
    DOCA_LOG_INFO("Resources: %u counters, %u meters, %u shared counters",
                  cfg->resource.nb_counters, cfg->resource.nb_meters, 
                  cfg->resource.nb_shared_counters);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_destroy(void) {
    pthread_mutex_lock(&simulator.lock);
    
    if (!simulator.flow_ctx.initialized) {
        pthread_mutex_unlock(&simulator.lock);
        return DOCA_ERROR_INITIALIZATION;
    }
    
    /* Stop packet simulator */
    simulator.packet_simulator_running = false;
    pthread_mutex_unlock(&simulator.lock);
    
    pthread_join(simulator.packet_sim_thread, NULL);
    
    pthread_mutex_lock(&simulator.lock);
    simulator.flow_ctx.initialized = false;
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("DOCA Flow destroyed - processed %lu packets, dropped %lu",
                  simulator.flow_ctx.packets_processed, 
                  simulator.flow_ctx.packets_dropped);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_port_start(const struct doca_flow_port_cfg *cfg, 
                                  struct doca_flow_port **port) {
    if (!cfg || !port) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    
    if (cfg->port_id >= 16) {
        pthread_mutex_unlock(&simulator.lock);
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    struct doca_flow_port *new_port = &simulator.ports[simulator.port_count];
    new_port->port_id = cfg->port_id;
    new_port->is_active = true;
    new_port->rx_packets = 0;
    new_port->tx_packets = 0;
    new_port->rx_bytes = 0;
    new_port->tx_bytes = 0;
    
    *port = new_port;
    simulator.port_count++;
    
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("Started port %u (type=%d, devargs=%s)", 
                  cfg->port_id, cfg->type, cfg->devargs);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_port_stop(struct doca_flow_port *port) {
    if (!port) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    port->is_active = false;
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("Stopped port %u - RX: %lu packets, TX: %lu packets",
                  port->port_id, port->rx_packets, port->tx_packets);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_pipe_create(const struct doca_flow_pipe_cfg *cfg,
                                   const struct doca_flow_fwd *fwd,
                                   const struct doca_flow_fwd *fwd_miss,
                                   struct doca_flow_pipe **pipe) {
    if (!cfg || !pipe) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    
    struct doca_flow_pipe *new_pipe = &simulator.pipes[simulator.pipe_count];
    strcpy(new_pipe->name, cfg->attr.name);
    new_pipe->pipe_id = simulator.pipe_count;
    new_pipe->port = cfg->port;
    new_pipe->entry_count = 0;
    new_pipe->is_active = true;
    
    *pipe = new_pipe;
    simulator.pipe_count++;
    simulator.flow_ctx.pipe_count++;
    
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("Created pipe '%s' (id=%u, type=%u, root=%d, actions=%u)",
                  cfg->attr.name, new_pipe->pipe_id, cfg->attr.type,
                  cfg->attr.is_root, cfg->attr.nb_actions);
    
    if (fwd) {
        DOCA_LOG_DEBUG("Forward action: type=%u, port=%u", fwd->type, fwd->port_id);
    }
    if (fwd_miss) {
        DOCA_LOG_DEBUG("Miss action: type=%u, port=%u", fwd_miss->type, fwd_miss->port_id);
    }
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_pipe_destroy(struct doca_flow_pipe *pipe) {
    if (!pipe) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    pipe->is_active = false;
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("Destroyed pipe '%s' (id=%u)", pipe->name, pipe->pipe_id);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_pipe_add_entry(uint16_t pipe_queue,
                                      struct doca_flow_pipe *pipe,
                                      const struct doca_flow_match *match,
                                      const struct doca_flow_actions *actions,
                                      void *monitor,
                                      void *fwd,
                                      uint32_t flags,
                                      void *usr_ctx,
                                      struct doca_flow_pipe_entry **entry) {
    if (!pipe || !match || !actions || !entry) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    
    struct doca_flow_pipe_entry *new_entry = &simulator.entries[simulator.entry_count];
    new_entry->entry_id = simulator.entry_count;
    new_entry->pipe = pipe;
    memcpy(&new_entry->match, match, sizeof(*match));
    memcpy(&new_entry->actions, actions, sizeof(*actions));
    new_entry->packets_matched = 0;
    new_entry->bytes_matched = 0;
    
    *entry = new_entry;
    simulator.entry_count++;
    simulator.flow_ctx.entry_count++;
    pipe->entry_count++;
    
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_INFO("Added entry %u to pipe '%s'", new_entry->entry_id, pipe->name);
    DOCA_LOG_DEBUG("Match: L3=%04x, L4=%u, TCP dst_port=%u",
                   match->outer.l3_type, match->outer.l4_type_ext,
                   ntohs(match->outer.tcp.l4_port.dst_port));
    DOCA_LOG_DEBUG("Action: type=%u, drop=%d", actions->action_type, actions->drop);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_entries_process(struct doca_flow_port *port,
                                       uint16_t pipe_queue,
                                       uint64_t timeout,
                                       uint32_t max_processed) {
    if (!port) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    /* Simulate processing delay */
    usleep(1000);  /* 1ms processing time */
    
    DOCA_LOG_DEBUG("Processed entries on port %u (queue=%u, timeout=%lu, max=%u)",
                   port->port_id, pipe_queue, timeout, max_processed);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_query_entry(struct doca_flow_pipe_entry *entry,
                                   struct doca_flow_query *query_stats) {
    if (!entry || !query_stats) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    pthread_mutex_lock(&simulator.lock);
    query_stats->total_pkts = entry->packets_matched;
    query_stats->total_bytes = entry->bytes_matched;
    pthread_mutex_unlock(&simulator.lock);
    
    DOCA_LOG_DEBUG("Query entry %u: %lu packets, %lu bytes",
                   entry->entry_id, query_stats->total_pkts, query_stats->total_bytes);
    
    return DOCA_SUCCESS;
}

/* DOCA Context API Implementation */
doca_error_t doca_flow_ctx_create(struct doca_flow_ctx **ctx) {
    if (!ctx) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    *ctx = &simulator.flow_ctx;
    DOCA_LOG_INFO("Created DOCA Flow context");
    
    return DOCA_SUCCESS;
}

doca_error_t doca_flow_ctx_destroy(struct doca_flow_ctx *ctx) {
    if (!ctx) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    DOCA_LOG_INFO("Destroyed DOCA Flow context");
    return DOCA_SUCCESS;
}

/* DOCA DevEmu API Implementation */
doca_error_t doca_devemu_create(struct doca_dev *dev, struct doca_devemu **devemu) {
    if (!dev || !devemu) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    struct doca_devemu *new_devemu = malloc(sizeof(struct doca_devemu));
    if (!new_devemu) {
        return DOCA_ERROR_NO_MEMORY;
    }
    
    new_devemu->device = dev;
    new_devemu->emulation_active = true;
    new_devemu->virtual_functions = 16;  /* Simulate 16 VFs */
    
    *devemu = new_devemu;
    
    DOCA_LOG_INFO("Created DevEmu context for device '%s' with %u VFs",
                  dev->name, new_devemu->virtual_functions);
    
    return DOCA_SUCCESS;
}

doca_error_t doca_devemu_destroy(struct doca_devemu *devemu) {
    if (!devemu) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    DOCA_LOG_INFO("Destroyed DevEmu context");
    free(devemu);
    
    return DOCA_SUCCESS;
}

/* Main Test Function */
int main(int argc, char **argv) {
    doca_error_t result;
    
    printf("========================================\n");
    printf(" NVIDIA DOCA DevEmu Simulator v2.6.0   \n");
    printf("========================================\n");
    printf("Simulating BlueField-3 DPU Environment\n");
    printf("========================================\n\n");
    
    /* Initialize mutex */
    pthread_mutex_init(&simulator.lock, NULL);
    
    /* Initialize logging */
    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS) {
        fprintf(stderr, "Failed to initialize logging\n");
        return 1;
    }
    
    /* Get device inventory */
    struct doca_dev **dev_list;
    uint32_t nb_devs;
    
    result = doca_dev_inventory_get(&dev_list, &nb_devs);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get device inventory: %s", doca_error_get_descr(result));
        return 1;
    }
    
    DOCA_LOG_INFO("Found %u DOCA devices", nb_devs);
    
    /* Initialize DOCA Flow */
    struct doca_flow_cfg flow_cfg = {
        .pipe_queues = 16,
        .mode_args = "vnf,hws,isolated",
        .resource = {
            .nb_counters = 1024,
            .nb_meters = 512,
            .nb_shared_counters = 256
        }
    };
    
    result = doca_flow_init(&flow_cfg);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize DOCA Flow: %s", doca_error_get_descr(result));
        return 1;
    }
    
    /* Start ports */
    struct doca_flow_port *uplink_port, *downlink_port;
    struct doca_flow_port_cfg port_cfg;
    
    port_cfg.port_id = 0;
    port_cfg.type = 1;
    sprintf(port_cfg.devargs, "%d", 0);
    
    result = doca_flow_port_start(&port_cfg, &uplink_port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start uplink port: %s", doca_error_get_descr(result));
        return 1;
    }
    
    port_cfg.port_id = 1;
    sprintf(port_cfg.devargs, "%d", 1);
    
    result = doca_flow_port_start(&port_cfg, &downlink_port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start downlink port: %s", doca_error_get_descr(result));
        return 1;
    }
    
    /* Create firewall pipe */
    struct doca_flow_pipe_cfg pipe_cfg = {0};
    struct doca_flow_match match = {0};
    struct doca_flow_match match_mask = {0};
    struct doca_flow_actions actions = {0};
    struct doca_flow_actions *actions_arr[1];
    struct doca_flow_fwd fwd = {0};
    struct doca_flow_pipe *firewall_pipe;
    
    strcpy(pipe_cfg.attr.name, "N6_FIREWALL_PIPE");
    pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
    pipe_cfg.attr.is_root = true;
    pipe_cfg.attr.nb_actions = 1;
    pipe_cfg.attr.domain = DOCA_FLOW_PIPE_DOMAIN_DEFAULT;
    pipe_cfg.port = uplink_port;
    
    /* Match TCP traffic */
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
    match_mask.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match_mask.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
    match_mask.outer.tcp.l4_port.dst_port = 0xFFFF;
    
    pipe_cfg.match = &match;
    pipe_cfg.match_mask = &match_mask;
    
    actions.action_type = DOCA_FLOW_ACTION_DROP;
    actions_arr[0] = &actions;
    pipe_cfg.actions = actions_arr;
    
    fwd.type = DOCA_FLOW_FWD_PORT;
    fwd.port_id = 1;
    
    result = doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd, &firewall_pipe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create firewall pipe: %s", doca_error_get_descr(result));
        return 1;
    }
    
    /* Add firewall rule to block port 8001 */
    struct doca_flow_pipe_entry *entry;
    match.outer.tcp.l4_port.dst_port = rte_cpu_to_be_16(8001);
    actions.drop = true;
    
    result = doca_flow_pipe_add_entry(0, firewall_pipe, &match, &actions, 
                                      NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &entry);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add firewall entry: %s", doca_error_get_descr(result));
        return 1;
    }
    
    result = doca_flow_entries_process(uplink_port, 0, DEFAULT_TIMEOUT_US, 0);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to process entries: %s", doca_error_get_descr(result));
        return 1;
    }
    
    printf("\n✅ DevEmu Firewall Simulation Active!\n");
    printf("   Blocking TCP port: 8001\n");
    printf("   Processing packets...\n\n");
    
    /* Run simulation for 10 seconds */
    for (int i = 0; i < 10; i++) {
        sleep(1);
        
        /* Query statistics */
        struct doca_flow_query stats;
        doca_flow_query_entry(entry, &stats);
        
        pthread_mutex_lock(&simulator.lock);
        printf("Statistics (t=%ds):\n", i+1);
        printf("  Total processed: %llu packets\n", (unsigned long long)simulator.flow_ctx.packets_processed);
        printf("  Total dropped:   %llu packets\n", (unsigned long long)simulator.flow_ctx.packets_dropped);
        printf("  Rule matches:    %llu packets\n", (unsigned long long)stats.total_pkts);
        printf("  Port 0 RX:       %llu packets\n", (unsigned long long)simulator.ports[0].rx_packets);
        printf("  Port 1 TX:       %llu packets\n", (unsigned long long)simulator.ports[1].tx_packets);
        printf("\n");
        pthread_mutex_unlock(&simulator.lock);
    }
    
    /* Cleanup */
    doca_flow_pipe_destroy(firewall_pipe);
    doca_flow_port_stop(downlink_port);
    doca_flow_port_stop(uplink_port);
    doca_flow_destroy();
    
    pthread_mutex_destroy(&simulator.lock);
    
    printf("========================================\n");
    printf(" DevEmu Simulation Complete\n");
    printf("========================================\n");
    
    return 0;
}