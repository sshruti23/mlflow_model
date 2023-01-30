output=$(curl -X POST https://${{ secrets.AWS_DB_HOST }}/api/2.0/clusters/create \
              -H "Authorization:Bearer ${{ secrets.AWS_DB_TOKEN }}" \
              -H "Content-Type: application/json" \
              --data @github_workflows/data/cluster_creation_request.json)
id=$( jq -r  '.cluster_id' <<< "${output}" )
echo "cluster_id=$id" >> $GITHUB_ENV