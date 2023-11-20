import configargparse

def get_args():
    parser = configargparse.ArgParser(default_config_files=["parameters.ini"])

    # Add arguments to the configuration parser
    parser.add_argument(
        "-c",
        "--my-config",
        default="parameters.ini",
        is_config_file=True,
        help="Config file path"
    )

    parser.add_argument(
        "--data-folder",
        help="Path to the folder containing all data subfolders",
        type=str
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA training."
    )

    return parser
