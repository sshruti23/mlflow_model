import argparse
import importlib

parser = argparse.ArgumentParser(
    prog="Library Code", description="Main Entry", epilog="Text at the bottom of help"
)

parser.add_argument("-m", "--module")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.module)

    model_module = importlib.import_module(args.module)
    model_module.start()
