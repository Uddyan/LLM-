# MCP Server Agents

This directory contains the Model Context Protocol (MCP) server agents for automated business operations.

## Agent Structure

Each agent is an MCP server that provides specialized functionality:

### 1. Analytics Agent (`analytics-agent/`)
- Provides data insights to managers
- Generates reports and identifies trends
- Predictive analytics

### 2. Inventory Agent (`inventory-agent/`)
- Tracks inventory across franchises
- Automated reorder triggers
- Supplier management

### 3. Customer Service Agent (`customer-service-agent/`)
- Handles customer inquiries 24/7
- Schedules service appointments
- Processes complaints and refunds

### 4. Franchisee Support Agent (`franchisee-support-agent/`)
- Supports franchise owners with operations
- Provides training resources
- Troubleshooting and best practices

### 5. Training Agent (`training-agent/`)
- Delivers personalized training programs
- Tracks completion and certifications
- Identifies skill gaps

### 6. Scheduling Agent (`scheduling-agent/`)
- Optimizes technician scheduling
- Route optimization
- Emergency dispatch

## Technologies

- FastMCP (Python framework)
- FastAPI
- LangGraph
- AutoGen
