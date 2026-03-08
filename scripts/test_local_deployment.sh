#!/bin/bash
# Test the local ADHD environment deployment
# Usage: ./scripts/test_local_deployment.sh
#
# Prerequisites: Server must be running (./scripts/run_local_server.sh)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8001"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Local ADHD Environment${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Test 1: Health check
echo -e "${BLUE}Test 1: Health Check${NC}"
response=$(curl -s ${BASE_URL}/health)
echo "Response: $response"
if echo "$response" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}\n"
else
    echo -e "${RED}✗ Health check failed${NC}\n"
    exit 1
fi

# Test 2: Root endpoint
echo -e "${BLUE}Test 2: Root Endpoint (/)${NC}"
response=$(curl -s ${BASE_URL}/)
echo "$response" | jq '.'
echo -e "${GREEN}✓ Root endpoint working${NC}\n"

# Test 3: Info endpoint
echo -e "${BLUE}Test 3: Info Endpoint${NC}"
response=$(curl -s ${BASE_URL}/info)
echo "$response" | jq '.'
echo -e "${GREEN}✓ Info endpoint working${NC}\n"

# Test 4: Reset endpoint
echo -e "${BLUE}Test 4: Reset Endpoint${NC}"
response=$(curl -s -X POST ${BASE_URL}/reset)
echo "$response" | jq '.'
scenario=$(echo "$response" | jq -r '.observation.scenario')
echo -e "Scenario: ${YELLOW}$scenario${NC}"
echo -e "${GREEN}✓ Reset endpoint working${NC}\n"

# Test 5: Step with good action (primary tool)
echo -e "${BLUE}Test 5: Step with Good Action (primary tool)${NC}"
response=$(curl -s -X POST ${BASE_URL}/step \
    -H "Content-Type: application/json" \
    -d '{"tool_calls": ["adhd_task_initiation_coach"], "message": "Open the document and type just the title."}')
echo "$response" | jq '.'
reward=$(echo "$response" | jq -r '.reward')
echo -e "Reward: ${YELLOW}$reward${NC}"
if [ "$reward" = "1.0" ]; then
    echo -e "${GREEN}✓ Good action scored 1.0${NC}\n"
else
    echo -e "${RED}✗ Expected reward 1.0, got $reward${NC}\n"
    exit 1
fi

# Test 6: Reset and step with bad action (no tool)
echo -e "${BLUE}Test 6: Step with Bad Action (no tool)${NC}"
curl -s -X POST ${BASE_URL}/reset > /dev/null
response=$(curl -s -X POST ${BASE_URL}/step \
    -H "Content-Type: application/json" \
    -d '{"tool_calls": [], "message": "What do you want to work on?"}')
echo "$response" | jq '.'
reward=$(echo "$response" | jq -r '.reward')
echo -e "Reward: ${YELLOW}$reward${NC}"
if [ "$reward" = "0.0" ]; then
    echo -e "${GREEN}✓ Bad action scored 0.0${NC}\n"
else
    echo -e "${RED}✗ Expected reward 0.0, got $reward${NC}\n"
    exit 1
fi

# Test 7: Reset and step with medium action (secondary tool)
echo -e "${BLUE}Test 7: Step with Medium Action (secondary tool)${NC}"
curl -s -X POST ${BASE_URL}/reset > /dev/null
response=$(curl -s -X POST ${BASE_URL}/step \
    -H "Content-Type: application/json" \
    -d '{"tool_calls": ["set_timer"], "message": "Let me set a 5 minute timer."}')
echo "$response" | jq '.'
reward=$(echo "$response" | jq -r '.reward')
echo -e "Reward: ${YELLOW}$reward${NC}"
if [ "$reward" = "0.5" ]; then
    echo -e "${GREEN}✓ Medium action scored 0.5${NC}\n"
else
    echo -e "${RED}✗ Expected reward 0.5, got $reward${NC}\n"
    exit 1
fi

# Test 8: Run manual test script
echo -e "${BLUE}Test 8: Manual Test Script${NC}"
PYTHONPATH=/workspaces/adhd-coach .venv/bin/python scripts/test_environment_manual.py

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All Tests Passed!${NC}"
echo -e "${GREEN}========================================${NC}\n"
echo -e "${BLUE}Server is running at: ${NC}http://localhost:8001"
echo -e "${BLUE}API docs available at: ${NC}http://localhost:8001/docs\n"
