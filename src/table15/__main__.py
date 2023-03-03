import sys

from table15.runner import run


def main():
    args = sys.argv[1:]
    if len(args) > 0:
        pipeline_configs_path = args[0]
        output_path = args[1] if len(args) > 1 else './output'
    else:
        pipeline_configs_path = 'src/table15/configs/pipeline_configs/pima.yaml'
        output_path = './output'
    run(pipeline_configs_path=pipeline_configs_path, output_path=output_path)


if __name__ == '__main__':
    main()
