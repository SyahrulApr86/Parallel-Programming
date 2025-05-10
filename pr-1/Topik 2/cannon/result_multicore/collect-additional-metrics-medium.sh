#!/bin/bash

# Output files
OUTPUT_DIR="medium-metrics-$(date +%Y%m%d-%H%M%S)"
mkdir -p $OUTPUT_DIR

# Collect container resource usage
echo "Collecting container resource metrics..."
kubectl top pods -n authentication-app -l node=medium --containers > $OUTPUT_DIR/container-resources.txt

# Get pod logs
echo "Collecting pod logs..."
for pod in $(kubectl get pods -n authentication-app -l node=medium -o jsonpath='{.items[*].metadata.name}'); do
  kubectl logs -n authentication-app $pod > $OUTPUT_DIR/${pod}-logs.txt
done

# Get detailed pod information
echo "Collecting detailed pod information..."
kubectl get pods -n authentication-app -l node=medium -o yaml > $OUTPUT_DIR/pods-detail.yaml

# Get events
echo "Collecting events..."
kubectl get events -n authentication-app --sort-by=.metadata.creationTimestamp > $OUTPUT_DIR/events.txt

# Get service endpoints
echo "Collecting service endpoints..."
kubectl get endpoints -n authentication-app > $OUTPUT_DIR/endpoints.txt

# Get node detailed info
echo "Collecting node information..."
kubectl describe node k3s-worker-medium > $OUTPUT_DIR/node-details.txt

echo "Additional metrics collection complete. Saved to $OUTPUT_DIR/"
