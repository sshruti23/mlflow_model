import argparse
from pkg_resources import resource_string

parser = argparse.ArgumentParser(
    prog="Library Code", description="Main Entry", epilog="Text at the bottom of help"
)

parser.add_argument("-m", "--module")

if __name__ == "__main__":
    args = parser.parse_args()
    print("Module :" + args.module)

    import os
    from os.path import isfile, join

    print("--------")
    cwd = os.getcwd()
    onlyfiles = [os.path.join(cwd, f) for f in os.listdir(cwd) if
                 os.path.isfile(os.path.join(cwd, f))]
    print(onlyfiles)
    print("--------")

    model_module = resource_string(__name__, args.module)
    model_module.start()
