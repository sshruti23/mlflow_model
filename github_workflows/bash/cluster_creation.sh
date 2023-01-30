#!/bin/bash
host=$1
pat_token=$2
output=$(curl -X POST https://$host/api/2.0/clusters/create \
              -H "Authorization:Bearer $pat_token" \
              -H "Content-Type: application/json" \
              --data @github_workflows/data/cluster_creation_request.json)
id=$( jq -r  '.cluster_id' <<< "${output}" )
echo "cluster_id=$id" >> $GITHUB_ENV