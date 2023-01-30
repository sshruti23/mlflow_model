#!/bin/bash
host=$1
pat_token=$2
curl -X POST https://$host/api/2.1/jobs/runs/submit \
-H "Authorization:Bearer $pat_token}" \
-H "Content-Type: application/json" \
--data @github_workflows/data/training_single_cluster.json

rm -rf github_workflows/data/request.json