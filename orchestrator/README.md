# Orchestrator

This directory contains the agent orchestration layer that coordinates multiple MCP agents.

## Structure

- **agent-coordinator/**: Multi-agent coordination
  - Query planning and routing
  - Agent selection and invocation
  - Result synthesis
  - Complex workflow management

## Functionality

The orchestrator:
1. Analyzes incoming queries
2. Determines which agents to invoke
3. Executes multi-step plans
4. Synthesizes results from multiple agents
5. Returns unified responses

## Technologies

- LangGraph for complex workflows
- AutoGen for multi-agent systems
- RabbitMQ/SQS for message queuing
- Redis for caching
