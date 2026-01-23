from argparse import ArgumentParser
from yaml import load, Loader

def get_yaml_config(model_name):
    with open(f"models/{model_name}/config.yaml", "r") as f:
        return load(f, Loader=Loader)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    config = get_yaml_config(args.model)

if __name__ == "__main__":
    main()