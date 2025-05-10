#!/bin/bash

# Start metric collection in background
./collect-metrics-medium.sh &
COLLECT_PID=$!

# Wait a moment for collection to start
sleep 2

# Run load test
echo "Starting load test for medium worker..."
k6 run --out json=medium-results.json load-test-medium.js

# Wait for metrics collection to complete
wait $COLLECT_PID
echo "Testing and metrics collection complete."

# Optional: Get resource metrics for all pods
echo "Resource utilization for pods:"
kubectl top pods -n authentication-app -l node=medium > pod-resource-medium.txt

# Get detailed HPA status
kubectl describe hpa authentication-hpa-medium -n authentication-app > hpa-status-medium.txt

echo "All data collected and saved."
