# Scheduling Agent

MCP server agent for optimizing technician scheduling.

## Capabilities

- Smart appointment scheduling
- Route optimization
- Workload balancing
- Emergency dispatch
- Availability management

## Tools/Integrations

- Scheduling software (ServiceTitan, Jobber)
- Google Maps API
- Calendar systems (Google Calendar, Outlook)

## MCP Resources

- `resources://schedules/*`
- `resources://technicians/*`
- `resources://appointments/*`

## Example Query

**Dispatcher**: "Schedule HVAC maintenance for customer in zip code 76710"

**Agent**: Finds available tech → Optimizes route → Books appointment → Sends confirmations
