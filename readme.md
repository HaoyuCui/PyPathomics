# PyPathomics

> PyPathomics is an open-source software for gigapixel whole-slide image analysis.

The PyPathomics is under development. The current version is 0.1.0.

## Installation

1. Clone the repository and navigate to the directory

2. Set Up the environment
    ```bash
   conda create -n pypathomics python=3.7
    conda activate pypathomics
    pip install -r requirements.txt
    ```
   
## Options and Usage

### Options: 

#### Options for config.yaml (e.g.)

```yaml
   openslide-home: path\to\openslide-home
   feature-set: ['Morph', 'Texture', 'Triangle']
   cell_types: ['I', 'S', 'T']
```

Where, 
- `openslide-home`: the path to the openslide library
- `feature-set`: the list of feature sets to be extracted, accept: `Morph`, `Texture`, `Triangle`
- `cell_types`: the list of cell types to be extracted, accept: `I`, `S`, `T`



#### Options for main.py


```text
   -f           Input: file, rather than directory [default: False]
   --auto_skip  Automatically skip the existing directory [default: True]
   --config     Path to the configuration file
   --json       *Path to the json dir(file) from Hover-Net
   --wsi        *Path to the WSI dir(file)
   --ext        *Extension of the WSI file, [default: .svs]
   --level      Level of the WSI [default: 0]
   --buffer     *Output buffer for preprocess
   --output     *Output directory for the run
```
Necessary arguments are marked with `*`

### Usage: 
1. run as batch
```bash
python main.py --json /path/to/json_dir --wsi /path/to/wsi_dir --buffer /path/to/buffer --ext .svs --output /path/to/output
```

2. run as single file
```bash
python main.py -f --json /path/to/json_file --wsi /path/to/wsi_file --buffer /path/to/buffer --ext .svs --output /path/to/output
```

