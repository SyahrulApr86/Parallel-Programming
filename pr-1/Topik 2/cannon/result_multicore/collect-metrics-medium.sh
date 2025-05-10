#!/bin/bash

# Nama file output
HPA_FILE="hpa-metrics-medium.csv"
PODS_FILE="pods-metrics-medium.csv"
NODE_FILE="node-metrics-medium.csv"

# Header untuk file CSV
echo "Timestamp,Name,Reference,Targets,MinPods,MaxPods,Replicas" > $HPA_FILE
echo "Timestamp,Name,Ready,Status,Restarts,Age,Node" > $PODS_FILE
echo "Timestamp,Node,CPU(cores),CPU(%),Memory(bytes),Memory(%)" > $NODE_FILE

# Interval dalam detik
INTERVAL=5
DURATION=300 # 5 menit

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))

# Fungsi untuk mendapatkan data
get_metrics() {
  CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
  
  # HPA metrics
  kubectl get hpa authentication-hpa-medium -n authentication-app -o custom-columns="NAME:.metadata.name,REF:.spec.scaleTargetRef.name,TARGETS:.status.currentMetrics[0].resource.current.averageUtilization,MINPODS:.spec.minReplicas,MAXPODS:.spec.maxReplicas,REPLICAS:.status.currentReplicas" --no-headers | awk -v time="$CURRENT_TIME" '{print time","-bash}' >> $HPA_FILE
  
  # Pods metrics
  kubectl get pods -n authentication-app -l node=medium -o custom-columns="NAME:.metadata.name,READY:.status.containerStatuses[0].ready,STATUS:.status.phase,RESTARTS:.status.containerStatuses[0].restartCount,AGE:.metadata.creationTimestamp,NODE:.spec.nodeName" --no-headers | awk -v time="$CURRENT_TIME" '{print time","-bash}' >> $PODS_FILE
  
  # Node metrics
  kubectl top nodes k3s-worker-medium | grep -v "NAME" | awk -v time="$CURRENT_TIME" '{print time","-bash}' >> $NODE_FILE
}

# Pengumpulan data loop
echo "Collecting metrics for $DURATION seconds at $INTERVAL second intervals..."
while [ $(date +%s) -lt $END_TIME ]; do
  get_metrics
  sleep $INTERVAL
done

echo "Metrics collection complete."
echo "HPA metrics saved to: $HPA_FILE"
echo "Pod metrics saved to: $PODS_FILE"
echo "Node metrics saved to: $NODE_FILE"
