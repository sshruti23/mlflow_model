import sys

from jinja2 import Template
import json


def get(databricks_cluster_id: str):
    file = open("github_workflows/data/training_single_cluster.json")
    data = json.load(file)
    template = Template(str(data))
    request_body = template.render(cluster_id=databricks_cluster_id)
    with open("github_workflows/data/request.json", "w+") as outfile:
        outfile.write(request_body.replace("\'", "\""))


if __name__ == '__main__':
    args = sys.argv
    get(args[1])
