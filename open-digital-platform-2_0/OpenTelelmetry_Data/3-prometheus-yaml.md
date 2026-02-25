```yaml

# File: /etc/prometheus/prometheus.yml

# Global config
global:
  scrape_interval: 15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
  evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
  # scrape_timeout is set to the global default (10s).

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

# Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

# Scrape configurations
scrape_configs:
  # Prometheus itself
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # AMF metrics
  - job_name: "amf"
    static_configs:
      - targets: ["localhost:9100"]

  # You can add more jobs for other Network Functions here as you instrument them
  # For example:
  # - job_name: "smf"
  #   static_configs:
  #     - targets: ["localhost:9101"]
  
  # - job_name: "upf"
  #   static_configs:
  #     - targets: ["localhost:9102"]

```