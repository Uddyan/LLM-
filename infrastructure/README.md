# Infrastructure

This directory contains infrastructure-as-code and deployment configurations.

## Structure

- **kubernetes/**: Kubernetes manifests and configurations
  - Deployments
  - Services
  - ConfigMaps and Secrets
  - Ingress configurations

- **terraform/**: Infrastructure provisioning
  - Cloud resources (AWS/Azure)
  - VPC and networking
  - Storage and databases
  - GPU instances for training

- **monitoring/**: Observability and monitoring
  - Prometheus configurations
  - Grafana dashboards
  - Logging (ELK stack)
  - LLM-specific metrics

## Technologies

- Kubernetes (EKS/AKS)
- Terraform
- Prometheus + Grafana
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Datadog/New Relic (optional)
