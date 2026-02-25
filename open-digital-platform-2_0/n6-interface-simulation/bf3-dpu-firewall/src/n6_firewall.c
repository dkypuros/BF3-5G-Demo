/**
 * @file n6_firewall.c
 * @brief Production N6 Interface Firewall for NVIDIA BlueField-3 DPU
 * 
 * This application implements a hardware-accelerated firewall on the N6 interface
 * between 5G UPF and Data Network using NVIDIA DOCA DevEmu and Flow APIs.
 * 
 * Real-world deployment:
 * - Runs on NVIDIA BlueField-3 DPU ARM cores
 * - Programs hardware flow tables using DOCA Flow
 * - Provides line-rate packet processing (up to 400Gbps)
 * - Integrates with DOCA DevEmu for device emulation
 * 
 * @author NVIDIA DOCA Team
 * @version 2.6.0
 * @date 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <argp.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>

/* NVIDIA DOCA Headers */
#include <doca_log.h>
#include <doca_error.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_flow.h>
#include <doca_devemu.h>

/* Application Headers */
#include "n6_firewall.h"
#include "n6_config.h"

DOCA_LOG_REGISTER(N6_FIREWALL);

/* Global application context */
static struct n6_firewall_ctx app_ctx = {0};
static volatile bool force_quit = false;

/**
 * Signal handler for graceful shutdown
 */
static void
signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        DOCA_LOG_INFO("Received signal %d, shutting down gracefully", signum);
        force_quit = true;
    }
}

/**
 * Initialize DOCA devices and contexts
 */
static doca_error_t
init_doca_devices(struct n6_firewall_ctx *ctx)
{
    doca_error_t result;
    struct doca_dev **dev_list;
    uint32_t nb_devs;

    DOCA_LOG_INFO("Initializing DOCA devices for BlueField-3 DPU");

    /* Get list of DOCA devices */
    result = doca_dev_inventory_get(&dev_list, &nb_devs);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get DOCA device inventory: %s", 
                     doca_error_get_descr(result));
        return result;
    }

    if (nb_devs == 0) {
        DOCA_LOG_ERR("No DOCA devices found - ensure BlueField-3 DPU is configured");
        return DOCA_ERROR_NOT_FOUND;
    }

    DOCA_LOG_INFO("Found %u DOCA devices", nb_devs);

    /* Use first available device */
    ctx->doca_dev = dev_list[0];
    
    /* Create DOCA Flow context */
    result = doca_flow_ctx_create(&ctx->flow_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create DOCA Flow context: %s",
                     doca_error_get_descr(result));
        return result;
    }

    /* Create DOCA DevEmu context */
    result = doca_devemu_create(ctx->doca_dev, &ctx->devemu_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create DOCA DevEmu context: %s",
                     doca_error_get_descr(result));
        doca_flow_ctx_destroy(ctx->flow_ctx);
        return result;
    }

    DOCA_LOG_INFO("DOCA devices initialized successfully");
    return DOCA_SUCCESS;
}

/**
 * Initialize DOCA Flow for hardware packet processing
 */
static doca_error_t
init_doca_flow(struct n6_firewall_ctx *ctx)
{
    struct doca_flow_cfg flow_cfg = {0};
    struct doca_flow_port_cfg port_cfg = {0};
    doca_error_t result;

    DOCA_LOG_INFO("Initializing DOCA Flow for hardware acceleration");

    /* Configure DOCA Flow */
    flow_cfg.pipe_queues = N6_FIREWALL_QUEUES;
    flow_cfg.mode_args = "vnf,hws,isolated";  /* VNF mode with hardware steering */
    flow_cfg.resource.nb_counters = N6_FIREWALL_COUNTERS;
    flow_cfg.resource.nb_meters = N6_FIREWALL_METERS;
    flow_cfg.resource.nb_shared_counters = N6_FIREWALL_SHARED_COUNTERS;

    result = doca_flow_init(&flow_cfg);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize DOCA Flow: %s",
                     doca_error_get_descr(result));
        return result;
    }

    /* Configure uplink port (toward UPF) */
    port_cfg.port_id = N6_UPLINK_PORT_ID;
    port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
    snprintf(port_cfg.devargs, sizeof(port_cfg.devargs), "%d", N6_UPLINK_PORT_ID);

    result = doca_flow_port_start(&port_cfg, &ctx->uplink_port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start uplink port: %s",
                     doca_error_get_descr(result));
        doca_flow_destroy();
        return result;
    }

    /* Configure downlink port (toward Data Network) */
    port_cfg.port_id = N6_DOWNLINK_PORT_ID;
    snprintf(port_cfg.devargs, sizeof(port_cfg.devargs), "%d", N6_DOWNLINK_PORT_ID);

    result = doca_flow_port_start(&port_cfg, &ctx->downlink_port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start downlink port: %s",
                     doca_error_get_descr(result));
        doca_flow_port_stop(ctx->uplink_port);
        doca_flow_destroy();
        return result;
    }

    DOCA_LOG_INFO("DOCA Flow initialized with uplink port %d and downlink port %d",
                  N6_UPLINK_PORT_ID, N6_DOWNLINK_PORT_ID);

    return DOCA_SUCCESS;
}

/**
 * Create firewall pipe for N6 interface traffic filtering
 */
static doca_error_t
create_firewall_pipe(struct n6_firewall_ctx *ctx)
{
    struct doca_flow_pipe_cfg pipe_cfg = {0};
    struct doca_flow_match match = {0};
    struct doca_flow_match match_mask = {0};
    struct doca_flow_actions actions = {0};
    struct doca_flow_actions *actions_arr[2];
    struct doca_flow_fwd fwd = {0};
    struct doca_flow_fwd fwd_miss = {0};
    doca_error_t result;

    DOCA_LOG_INFO("Creating N6 firewall pipe for traffic filtering");

    /* Configure pipe attributes */
    snprintf(pipe_cfg.attr.name, sizeof(pipe_cfg.attr.name), "N6_FIREWALL_PIPE");
    pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
    pipe_cfg.attr.is_root = true;
    pipe_cfg.attr.nb_actions = 2;  /* Allow and Drop actions */
    pipe_cfg.attr.domain = DOCA_FLOW_PIPE_DOMAIN_DEFAULT;
    pipe_cfg.port = ctx->uplink_port;

    /* Match on IPv4 TCP packets */
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
    match.outer.ip4.src_ip = 0;    /* Wildcard - match any source */
    match.outer.ip4.dst_ip = 0;    /* Wildcard - match any destination */
    match.outer.tcp.l4_port.src_port = 0;  /* Wildcard */
    match.outer.tcp.l4_port.dst_port = 0;  /* Will be set per rule */

    /* Set match mask */
    match_mask.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match_mask.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
    match_mask.outer.tcp.l4_port.dst_port = UINT16_MAX;  /* Exact match on dest port */

    pipe_cfg.match = &match;
    pipe_cfg.match_mask = &match_mask;

    /* Configure actions */
    /* Action 0: Drop */
    actions_arr[0] = &actions;
    actions.action_type = DOCA_FLOW_ACTION_DROP;

    /* Action 1: Forward to next pipe/port */
    actions_arr[1] = &actions;
    actions.action_type = DOCA_FLOW_ACTION_FORWARD;

    pipe_cfg.actions = actions_arr;

    /* Configure forwarding */
    fwd.type = DOCA_FLOW_FWD_PORT;
    fwd.port_id = N6_DOWNLINK_PORT_ID;

    /* Miss action - forward by default */
    fwd_miss.type = DOCA_FLOW_FWD_PORT;
    fwd_miss.port_id = N6_DOWNLINK_PORT_ID;

    result = doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd_miss, &ctx->firewall_pipe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create firewall pipe: %s",
                     doca_error_get_descr(result));
        return result;
    }

    DOCA_LOG_INFO("N6 firewall pipe created successfully");
    return DOCA_SUCCESS;
}

/**
 * Add firewall rule to block specific port
 */
static doca_error_t
add_firewall_rule(struct n6_firewall_ctx *ctx, uint16_t blocked_port)
{
    struct doca_flow_match match = {0};
    struct doca_flow_actions actions = {0};
    struct doca_flow_pipe_entry *entry;
    struct doca_flow_query query_stats = {0};
    doca_error_t result;

    DOCA_LOG_INFO("Adding firewall rule to block TCP port %u", blocked_port);

    /* Set match criteria for this specific rule */
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
    match.outer.tcp.l4_port.dst_port = rte_cpu_to_be_16(blocked_port);

    /* Use DROP action (index 0) */
    actions.action_type = DOCA_FLOW_ACTION_DROP;

    /* Add the entry to the pipe */
    result = doca_flow_pipe_add_entry(0, ctx->firewall_pipe, &match, &actions, 
                                      NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &entry);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add firewall entry for port %u: %s",
                     blocked_port, doca_error_get_descr(result));
        return result;
    }

    /* Store entry reference for statistics */
    ctx->blocked_port_entries[ctx->nb_blocked_ports] = entry;
    ctx->blocked_ports[ctx->nb_blocked_ports] = blocked_port;
    ctx->nb_blocked_ports++;

    /* Process entries to commit to hardware */
    result = doca_flow_entries_process(ctx->uplink_port, 0, DEFAULT_TIMEOUT_US, 0);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to process firewall entries: %s",
                     doca_error_get_descr(result));
        return result;
    }

    /* Query initial stats */
    result = doca_flow_query_entry(entry, &query_stats);
    if (result == DOCA_SUCCESS) {
        DOCA_LOG_INFO("Firewall rule for port %u active - initial stats: %lu packets, %lu bytes",
                      blocked_port, query_stats.total_pkts, query_stats.total_bytes);
    }

    return DOCA_SUCCESS;
}

/**
 * Print firewall statistics
 */
static void
print_firewall_stats(struct n6_firewall_ctx *ctx)
{
    struct doca_flow_query query_stats = {0};
    doca_error_t result;
    uint32_t i;

    printf("\n=== N6 Firewall Statistics ===\n");
    printf("Active Rules: %u\n", ctx->nb_blocked_ports);
    printf("%-10s %-15s %-15s\n", "Port", "Packets", "Bytes");
    printf("----------------------------------------\n");

    for (i = 0; i < ctx->nb_blocked_ports; i++) {
        result = doca_flow_query_entry(ctx->blocked_port_entries[i], &query_stats);
        if (result == DOCA_SUCCESS) {
            printf("%-10u %-15lu %-15lu\n", 
                   ctx->blocked_ports[i],
                   query_stats.total_pkts,
                   query_stats.total_bytes);
        } else {
            printf("%-10u %-15s %-15s\n", 
                   ctx->blocked_ports[i], "ERROR", "ERROR");
        }
    }
    printf("=============================\n\n");
}

/**
 * Main firewall processing loop
 */
static void
firewall_main_loop(struct n6_firewall_ctx *ctx)
{
    uint64_t last_stats_time = 0;
    uint64_t current_time;
    
    DOCA_LOG_INFO("Starting N6 firewall main processing loop");
    DOCA_LOG_INFO("Press Ctrl+C to stop the firewall");

    while (!force_quit) {
        /* Get current timestamp */
        current_time = rte_get_tsc_cycles() / rte_get_tsc_hz();

        /* Print statistics every 10 seconds */
        if (current_time - last_stats_time >= 10) {
            print_firewall_stats(ctx);
            last_stats_time = current_time;
        }

        /* Process any pending DOCA operations */
        doca_flow_entries_process(ctx->uplink_port, 0, 1000, 0);  /* 1ms timeout */
        
        /* Brief sleep to avoid busy polling */
        usleep(100000);  /* 100ms */
    }

    DOCA_LOG_INFO("Firewall main loop terminated");
}

/**
 * Cleanup resources
 */
static void
cleanup_resources(struct n6_firewall_ctx *ctx)
{
    DOCA_LOG_INFO("Cleaning up N6 firewall resources");

    if (ctx->firewall_pipe) {
        doca_flow_pipe_destroy(ctx->firewall_pipe);
    }

    if (ctx->uplink_port) {
        doca_flow_port_stop(ctx->uplink_port);
    }

    if (ctx->downlink_port) {
        doca_flow_port_stop(ctx->downlink_port);
    }

    doca_flow_destroy();

    if (ctx->devemu_ctx) {
        doca_devemu_destroy(ctx->devemu_ctx);
    }

    if (ctx->flow_ctx) {
        doca_flow_ctx_destroy(ctx->flow_ctx);
    }

    DOCA_LOG_INFO("Cleanup completed");
}

/**
 * Argument parsing structure
 */
static struct argp_option options[] = {
    {"port", 'p', "PORT", 0, "TCP port to block (default: 8001)"},
    {"verbose", 'v', 0, 0, "Enable verbose logging"},
    {"config", 'c', "FILE", 0, "Configuration file path"},
    {0}
};

struct arguments {
    uint16_t blocked_port;
    bool verbose;
    char *config_file;
};

static error_t
parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key) {
    case 'p':
        arguments->blocked_port = atoi(arg);
        if (arguments->blocked_port == 0 || arguments->blocked_port > 65535) {
            argp_usage(state);
        }
        break;
    case 'v':
        arguments->verbose = true;
        break;
    case 'c':
        arguments->config_file = arg;
        break;
    case ARGP_KEY_ARG:
        argp_usage(state);
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static char doc[] = "N6 Interface Firewall for NVIDIA BlueField-3 DPU";
static struct argp argp = {options, parse_opt, 0, doc};

/**
 * Main application entry point
 */
int
main(int argc, char **argv)
{
    struct arguments arguments = {
        .blocked_port = N6_DEFAULT_BLOCKED_PORT,
        .verbose = false,
        .config_file = NULL
    };
    doca_error_t result;
    int ret;

    /* Parse command line arguments */
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    /* Initialize logging */
    result = doca_log_backend_create_standard();
    if (result != DOCA_SUCCESS) {
        fprintf(stderr, "Failed to initialize DOCA logging\n");
        return EXIT_FAILURE;
    }

    if (arguments.verbose) {
        doca_log_level_set_global(DOCA_LOG_LEVEL_DEBUG);
    }

    printf("=========================================\n");
    printf("  NVIDIA BlueField-3 N6 Firewall v2.6  \n");
    printf("=========================================\n");
    printf("Configuration:\n");
    printf("  Blocked Port: %u\n", arguments.blocked_port);
    printf("  Verbose Mode: %s\n", arguments.verbose ? "Enabled" : "Disabled");
    printf("  Config File:  %s\n", arguments.config_file ? arguments.config_file : "None");
    printf("=========================================\n\n");

    /* Set up signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Initialize DOCA */
    result = init_doca_devices(&app_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize DOCA devices");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    /* Initialize DOCA Flow */
    result = init_doca_flow(&app_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to initialize DOCA Flow");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    /* Create firewall pipe */
    result = create_firewall_pipe(&app_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create firewall pipe");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    /* Add firewall rule */
    result = add_firewall_rule(&app_ctx, arguments.blocked_port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add firewall rule");
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    DOCA_LOG_INFO("N6 Firewall initialized successfully");
    DOCA_LOG_INFO("Blocking TCP traffic on port %u", arguments.blocked_port);

    /* Run main processing loop */
    firewall_main_loop(&app_ctx);

    ret = EXIT_SUCCESS;

cleanup:
    cleanup_resources(&app_ctx);
    return ret;
}