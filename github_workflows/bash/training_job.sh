curl -X POST https://${{ secrets.AWS_DB_HOST }}/api/2.1/jobs/runs/submit \
-H "Authorization:Bearer ${{ secrets.AWS_DB_TOKEN }}" \
-H "Content-Type: application/json" \
--data @github_workflows/data/training_single_cluster.json

rm -rf github_workflows/data/request.json