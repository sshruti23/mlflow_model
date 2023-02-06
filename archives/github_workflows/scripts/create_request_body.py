import sys

from jinja2 import Template
import json


def get(databricks_cluster_id: str):
    file = open("databricks_workflows/request_body/training_single_cluster.json")
    data = json.load(file)
    template = Template(str(data))

    request_body = template.render(cluster_id=databricks_cluster_id)
    with open("databricks_workflows/request_body/request.json", "w+") as outfile:
        outfile.write(request_body.replace("\'", "\""))


if __name__ == '__main__':
    args = sys.argv
    get(args[1])
