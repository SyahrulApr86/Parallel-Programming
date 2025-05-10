#!/bin/bash

# Mengekstrak statistik utama dari data K6
echo "Analyzing K6 results..."
jq -r '.metrics.http_req_duration.values.avg | "Average request duration: " + (. * 1000 | tostring) + " ms"' medium-results.json > medium-summary.txt
jq -r '.metrics.http_reqs.values.count | "Total requests: " + tostring' medium-results.json >> medium-summary.txt
jq -r '.metrics.http_reqs.values.rate | "Request rate: " + tostring + " req/s"' medium-results.json >> medium-summary.txt

# Mengekstrak informasi HPA
echo "Analyzing HPA data..."
echo "HPA scaling events:" >> medium-summary.txt
awk -F, 'NR>1 &&  != prev_replicas {print "Time: "  ", Replicas changed from " prev_replicas " to " ; prev_replicas=}' prev_replicas=1 hpa-metrics-medium.csv >> medium-summary.txt

# Mengekstrak max CPU dan memory usage
echo "Analyzing node metrics..." 
echo "Maximum CPU usage:" >> medium-summary.txt
sort -t, -k3,3 -n node-metrics-medium.csv | tail -1 >> medium-summary.txt
echo "Maximum memory usage:" >> medium-summary.txt
sort -t, -k5,5 -n node-metrics-medium.csv | tail -1 >> medium-summary.txt

echo "Analysis complete. Results saved to medium-summary.txt"
