import argparse
import os

import pandas as pd

import postprocess
import preprocess

from utils import get_config

parser = argparse.ArgumentParser(description='Configurations for PyPathomoics')
parser.add_argument('--auto_skip', action='store_true', default=True, help='Automatically skip the existing dir')
parser.add_argument('--config', type=str, default=None, help='Config file for the run')
parser.add_argument('-f', action='store_true', default=False, help='Input: file, instead of dir')
parser.add_argument('--json', type=str, default=None, help='Json dir(file) for the run')
parser.add_argument('--wsi', type=str, default=None, help='WSI dir(file) for the run')
parser.add_argument('--ext', type=str, default='svs', help='WSI file extension, default: svs')
parser.add_argument('--level', type=int, default=0, help='WSI level to process, default: 0')
parser.add_argument('--buffer', type=str, default=None, help='Output buffer for preprocess')
parser.add_argument('--output', type=str, default=None, help='Output dir for the run')
args = parser.parse_args()


def run_wsi(_args):
    preprocess.process(_args.wsi, _args.json, _args.output)


def main():
    if not args.f:
        process_queue = [file for file in os.listdir(args.json) if file.endswith('.json')]
        output_dir = args.output

        print(f'Total {len(process_queue)} files to process')
        for i, json_file in enumerate(process_queue):
            slide_name = json_file.split('.')[0]
            wsi_path = os.path.join(args.wsi, slide_name + args.ext.split('.')[1])
            if args.auto_skip and os.path.exists(os.path.join(output_dir, slide_name + '_Feats_T.csv')):
                print(f'Skip {slide_name} as is already processed')
                continue
            json_path = os.path.join(args.json, json_file)

            # preprocess
            preprocess.process(json_path, wsi_path, output_dir, level=args.level)

    else:
        run_wsi(args)
        process_queue = [args.json]

    # post process
    feature_list = get_config()['feature_list']
    cell_types = get_config()['cell_types']

    df_feats_list = []
    for i, slide in enumerate(process_queue):
        print(f'Processing {slide} {i} / {len(process_queue)}')
        extractor = postprocess.FeatureExtractor(slide, args.buffer, feature_list=feature_list, cell_types=cell_types)
        slide_feats = extractor.extract()
        slide_feats['slide'] = slide
        df_feats_list.append(slide_feats)

    df_feats = pd.concat(df_feats_list, ignore_index=True)
    cols = ['slide'] + [col for col in df_feats.columns if col != 'slide']
    df_feats = df_feats[cols]

    feats_loc = os.path.join(args.output, 'features.xlsx')
    df_feats.to_excel(feats_loc, index=False)


if __name__ == '__main__':
    main()
