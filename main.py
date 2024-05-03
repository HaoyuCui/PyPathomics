import argparse
import os
import preprocess


parser = argparse.ArgumentParser(description='Configurations for PyPathomoics')
parser.add_argument('-b', action='store_true', default=True, help='Run the directory, input as a directory')
parser.add_argument('--auto_skip', action='store_true', default=True, help='Automatically skip the existing dir')
parser.add_argument('--config', type=str, default=None, help='Config file for the run')
parser.add_argument('--json', type=str, default=None, help='Json dir(file) for the run')
parser.add_argument('--wsi', type=str, default=None, help='WSI dir(file) for the run')
parser.add_argument('--ext', type=str, default='svs', help='WSI file extension, default: svs')
parser.add_argument('--output', type=str, default=None, help='Output dir for the run')
args = parser.parse_args()


def run_wsi(_args):
    preprocess.process(_args.wsi, _args.json, _args.output)


def main():
    if args.b:
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
            preprocess.process(json_path, wsi_path, output_dir)
    else:
        run_wsi(args)
