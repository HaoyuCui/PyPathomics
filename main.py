import argparse
import os.path
from pathlib import Path
import logging

from src.postprocess import postprocess_files
from src.preprocess import preprocess_files, run_wsi
from src.utils import get_config, print_config

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for PyPathomoics', add_help=True)
    parser.add_argument('--auto_skip', action='store_true', default=True,
                        help='Automatically skip the existing dir')
    parser.add_argument('-f', '--file_mode', action='store_true', help='Input: file, instead of dir')
    parser.add_argument('--seg', type=Path, required=True,
                        help='Nucleus segmentation result dir(file) for the run, support suffix: .json or .dat')
    parser.add_argument('--wsi', type=Path, required=True, help='WSI dir(file) for the run')
    parser.add_argument('--ext', type=str, default='svs', help='WSI file extension, default: svs')
    parser.add_argument('--level', type=int, default=0, help='WSI level to process, default: 0')
    parser.add_argument('--buffer', type=Path, default=None, help='Output buffer for preprocess')
    parser.add_argument('--output', type=Path, default='pypathomics-result.csv',
                        help='Output file path for the run (suffix: xlsx/csv)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    configs = get_config()

    print_config(args, configs)

    if os.path.isdir(args.output):
        args.output = args.output / 'pypathomics-result.csv'

    if os.listdir(args.buffer) != 0:
        logging.warning(f'Buffer directory {args.buffer} is not empty, it may cause conflict. Continue? (y/N)')
        if input().lower() != 'y':
            return

    if not args.file_mode:
        preprocess_files(args, configs)
    else:
        run_wsi(args, configs)

    # Post-process features
    df_feats = postprocess_files(args, configs)

    if args.output.suffix == '.xlsx':
        df_feats.to_excel(args.output, index=False)
    elif args.output.suffix == '.csv':
        df_feats.to_csv(args.output, index=False)

    logging.info(f'Features saved to {args.output}')


if __name__ == '__main__':
    main()
