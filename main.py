import argparse
import os.path
from pathlib import Path
import pandas as pd
import logging

from src import postprocess, preprocess
from src.utils import get_config, print_config

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configurations for PyPathomoics')
    parser.add_argument('--auto_skip', action='store_true', default=True,
                        help='Automatically skip the existing dir')
    parser.add_argument('-f', '--file_mode', action='store_true', help='Input: file, instead of dir')
    parser.add_argument('--seg', type=Path, required=True,
                        help='Nucleus segmentation result dir(file) for the run, support suffix: .json or .dat')
    parser.add_argument('--wsi', type=Path, required=True, help='WSI dir(file) for the run')
    parser.add_argument('--ext', type=str, default='svs', help='WSI file extension, default: svs')
    parser.add_argument('--level', type=int, default=0, help='WSI level to process, default: 0')
    parser.add_argument('--buffer', type=Path, default=None, help='Output buffer for preprocess')
    parser.add_argument('--output', type=Path, required=True, help='Output path for the run (suffix: xlsx/csv)')
    return parser.parse_args()


def preprocess_files(args, configs):
    process_queue = list(args.seg.glob(f'*.json')) + list(args.seg.glob(f'*.dat'))
    output_dir = args.buffer
    ext = args.ext
    if len(process_queue) > 1 and ext is None:
        logging.warning('No file extension provided, use --ext to specify the extension.')
    elif len(process_queue) == 1 and ext is None:
        ext = process_queue[0].suffix[1:]
    logging.info(f'Total {len(process_queue)} files to process.')

    for i, seg_path in enumerate(process_queue):
        logging.info(f'Phase 1 Preprocessing \t {i + 1} / {len(process_queue)} \t  {seg_path} ')
        slide_name = seg_path.stem
        wsi_path = args.wsi / f"{slide_name}.{ext}"
        output_path = output_dir / f"{slide_name}_Feats_T.csv"

        if args.auto_skip and output_path.exists():
            logging.info(f'Skip {slide_name} as it is already processed.')
            continue

        preprocess.process(seg_path, wsi_path, output_dir, args.level, configs['feature-set'], configs['cell-types'])


def postprocess_files(args, configs):
    process_queue = list(args.seg.glob(f'*.json')) + list(args.seg.glob(f'*.dat'))
    df_feats_list = []
    for i, slide in enumerate(process_queue):
        logging.info(f'Phase 2 Postprocessing \t {i + 1} / {len(process_queue)} \t {slide} ')
        slide = slide.stem
        extractor = postprocess.FeatureExtractor(slide, args.buffer, feature_list=configs['feature-set'],
                                                 cell_types=configs['cell-types'],
                                                 statistic_types=configs['statistic-types'])
        slide_feats = extractor.extract()
        slide_feats['slide'] = slide
        df_feats_list.append(slide_feats)

    df_feats = pd.concat(df_feats_list, ignore_index=True)
    cols = ['slide'] + [col for col in df_feats.columns if col != 'slide']
    return df_feats[cols]


def run_wsi(args, configs):
    logging.info(f'Phase 1 Preprocessing \t 1 / 1 \t {args.seg} ')
    preprocess.process(args.seg, args.wsi, args.buffer, args.level, configs['feature-set'], configs['cell-types'])


def main():
    args = parse_arguments()
    configs = get_config()

    print_config(args)
    assert args.output.suffix in ['.xlsx', '.csv'], 'Output file should be in xlsx or csv format.'
    assert len(os.listdir(args.buffer)) == 0, (f'Buffer directory {args.buffer} should be empty, or it may cause '
                                               f'conflict.')

    if not args.file_mode:
        preprocess_files(args, configs)
    else:
        run_wsi(args, configs)

    # Post-process features
    df_feats = postprocess_files(args, configs)

    output_loc = args.output
    if args.output.suffix == '.xlsx':
        df_feats.to_excel(output_loc, index=False)
    elif args.output.suffix == '.csv':
        df_feats.to_csv(output_loc, index=False)

    logging.info(f'Features saved to {output_loc}')


if __name__ == '__main__':
    main()
