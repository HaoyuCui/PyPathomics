import argparse
from pathlib import Path
import pandas as pd
import logging
import sys

from src import postprocess, preprocess
from src.utils import get_config, print_config

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configurations for PyPathomoics')
    parser.add_argument('--auto_skip', action='store_true', default=True,
                        help='Automatically skip the existing dir')
    parser.add_argument('--config', type=str, default=None, help='Config file for the run')
    parser.add_argument('-f', '--file_mode', action='store_true', help='Input: file, instead of dir')
    parser.add_argument('--json', type=Path, required=True, help='Json dir(file) for the run')
    parser.add_argument('--wsi', type=Path, required=True, help='WSI dir(file) for the run')
    parser.add_argument('--ext', type=str, default='svs', help='WSI file extension, default: svs')
    parser.add_argument('--level', type=int, default=0, help='WSI level to process, default: 0')
    parser.add_argument('--buffer', type=Path, default=None, help='Output buffer for preprocess')
    parser.add_argument('--output', type=Path, required=True, help='Output dir for the run')
    return parser.parse_args()


def process_files(args, configs):
    process_queue = list(args.json.glob(f'*.json'))
    output_dir = args.output
    ext = args.ext.split('.')[-1]
    logging.info(f'Total {len(process_queue)} files to process.')

    for i, seg_path in enumerate(process_queue):
        slide_name = seg_path.stem
        wsi_path = args.wsi / f"{slide_name}.{ext}"
        output_path = output_dir / f"{slide_name}_Feats_T.csv"

        if args.auto_skip and output_path.exists():
            logging.info(f'Skip {slide_name} as it is already processed.')
            continue

        preprocess.process(seg_path, wsi_path, output_dir, args.level, configs['feature-set'], configs['cell-types'])


def run_wsi(args, configs):
    preprocess.process(args.wsi, args.json, args.output, args.level, configs['feature-set'], configs['cell-types'])


def main():
    args = parse_arguments()
    configs = get_config()

    print_config(args)
    process_queue = list(args.json.glob(f'*.json'))

    if not args.file_mode:
        process_files(args, configs)
    else:
        run_wsi(args, configs)

    # Post-process features

    df_feats_list = []

    for i, slide in enumerate(process_queue):
        logging.info(f'Processing {slide} {i + 1} / {len(process_queue)}')
        extractor = postprocess.FeatureExtractor(slide, args.buffer, feature_list=configs['feature-set'],
                                                 cell_types=configs['cell-types'],
                                                 statistic_types=configs['statistic-types'])
        slide_feats = extractor.extract()
        slide_feats['slide'] = slide
        df_feats_list.append(slide_feats)

    df_feats = pd.concat(df_feats_list, ignore_index=True)
    cols = ['slide'] + [col for col in df_feats.columns if col != 'slide']
    df_feats = df_feats[cols]

    feats_loc = args.output / 'features.xlsx'
    df_feats.to_excel(feats_loc, index=False)


if __name__ == '__main__':
    main()
