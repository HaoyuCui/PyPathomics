<div align=center>
 <img width="1080" alt="image" src="src/.logo.svg">
</div>

# PyPathomics

<a href="https://github.com/HaoyuCui/PyPathomics?tab=GPL-3.0-1-ov-file"> 
  <img align="right", src="https://img.shields.io/badge/License-GPL-blue.svg">
</a>

<a href="https://imic.nuist.edu.cn/"> 
  <img align="right", src="https://img.shields.io/static/v1?label=Org&message=iMIC&color=green"/>
</a>

PyPathomics is an open-source software for gigapixel whole-slide image analysis. Off-the-shelf and easy-to-use.

Currently under development. This is a simplified version of [[sc_MTOP](https://github.com/fuscc-deep-path/sc_MTOP)] [[Paper](https://www.nature.com/articles/s41467-023-42504-y)]

Support for:
- [x] Hover-Net
[[Repo](https://github.com/vqdang/hover_net)] [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045)] (Graham et al., 2019)

- [x] Cerberus (In Beta)
[[Repo](https://github.com/TissueImageAnalytics/cerberus)] [[Paper](https://doi.org/10.1016/j.media.2022.102685)] (Graham et al., 2023)

## How it works?

<div align=center>
 <img width="960" alt="image" src="https://github.com/user-attachments/assets/11098570-4939-4bb9-9080-7e2dbe3bc68c">
</div>

Get all **cell** and **slide-level** features directly through the cell segmentation file like Hover-Net's `.json` file.

## Installation

 ```bash
conda create -n pypathomics
conda activate pypathomics
pip install -r requirements.txt
 ```

## Options and Usage

### Options:

#### Options for config.yaml

```yaml
   openslide-home: path\to\openslide-home # for Windows only
   feature-set: ['Morph', 'Texture', 'Triangle']
   cell_types: ['I', 'S', 'T']
   statistic-types: ['basic', 'distribution']
```

Where, 
- `openslide-home`: Specifies the path to the OpenSlide library. (for Windows only, leave it empty for Linux)
- `feature-set`:  A list of feature sets to extract. Options: `Morph`, `Texture`, `Triangle`.
- `cell_types`: Types of cells to analyze. Options: `I` (inflammatory), `S`, (stromal), `T` (tumor).
- `statistic-types`: Types of statistics to calculate. Options: `basic`, `distribution`.

| statistics-types | e.g.                                             |
|------------------|--------------------------------------------------|
| basic            | mean, std                                        |
| distribution     | Q25, Q75, median, IQR, range, skewness, kurtosis |


#### Options for main.py

Required Arguments:
```text
    --config     Specify the configuration file path
    --seg        Path to the segmentation directory or file from Hover-Net(.json) or Cerberus(.dat)
    --wsi        Path to the WSI directory or file
    --ext        WSI file extension (default: .svs)
    --buffer     Specify the output buffer dir for preprocessing
    --output     Set the output directory for the analysis
```


Optional Arguments:  
```text
    -f           Run for a single file (default: run for directory) 
    --auto_skip  Skip existing directories automatically (default: True)
    --level      Detail level of the WSI to analyze (default: 0)
```

### Usage:

1. Make sure you run the [Hover-Net's wsi seg](https://github.com/vqdang/hover_net) or [Cerberus' wsi seg](https://github.com/TissueImageAnalytics/cerberus) and get the seg files. 

2. Modify and check the [config.yaml](./config.yaml) before running.

3. Analyze a Directory:
    ```bash
    python main.py --seg /path/to/seg_dir --wsi /path/to/wsi_dir --buffer /path/to/buffer --ext .svs --output /path/to/output.csv
    ```

4. Analyze a Single File
    ```bash
    python main.py -f --seg /path/to/seg_file --wsi /path/to/wsi_file --buffer /path/to/buffer --ext .svs --output /path/to/output.csv
    ```


##### All-cell Info, stored in /path/to/buffer

| Feature       | Description                                            |
|---------------|--------------------------------------------------------|
| **Name**      | Identifier for the cell                                |
| **Centroid**  | Position of the cell's centroid                        |
| **Cell Type** | Information about the cell type                        |

##### Slide Cell Ratio

| Feature       | Description                                            |
|---------------|--------------------------------------------------------|
| **Ratio**     | Reflects the proportion of cells of this type          |

##### Slide Morphological Features

| Feature                                         | Description                                                                   |
|-------------------------------------------------|-------------------------------------------------------------------------------|
| **Area**                                        | Area of the cell, indicating cell size                                        |
| **AreaBbox**                                    | Area of the minimum bounding rectangle around the cell                        |
| **CellEccentricities**                          | Eccentricity of the cell                                                      |
| **Circularity**                                 | Roundness of the cell                                                         |
| **Elongation**                                  | Elongation rate of the cell                                                   |
| **Extent**                                      | Proportion of the cell occupying its bounding rectangle                       |
| **MajorAxisLength** / **Morph_MinorAxisLength** | Lengths of the major and minor axes of the fitted ellipse for the cell        |
| **Perimeter**                                   | Perimeter of the cell boundary                                                |
| **Solidity**                                    | Ratio of the cell area to its convex hull area                                |
| **CurvMean** / **Std** / **Max** / **Min**      | Mean, standard deviation, maximum, and minimum of the cell boundary curvature |

##### Texture Features

| Feature                                         | Description                                                                                                           |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **ASM (Angular Second Moment)**                 | Texture consistency, measuring the similarity between a pixel and its neighbors                                       |
| **Contrast**                                    | Texture contrast, describing the intensity variation in the image                                                     |
| **Correlation**                                 | Texture correlation, measuring the similarity between a pixel and its neighbors                                       |
| **Entropy**                                     | Texture entropy, representing the diversity of information in the image; higher values indicate more complex textures |
| **Homogeneity**                                 | Texture homogeneity, assessing the consistency of the texture                                                         |
| **IntensityMean** / **Std** / **Max** / **Min** | Mean, standard deviation, maximum, and minimum of the texture intensity                                               |

##### Delaunay Triangle Spatial Features

| Feature         | Description                                                                                |
|-----------------|--------------------------------------------------------------------------------------------|
| **Area**        | Area of the Delaunay triangle around the cell                                              |
| **Perimeter**   | Perimeter of the Delaunay triangle around the cell                                         |
| **Angle_Range** | Difference between the maximum and minimum angles of the Delaunay triangle around the cell |


## Citation
    
 ```bibtex
   @software{pypathomics,
     author       = {HY Cui and XX Wang and J Xu and DP Chen},
     title        = {PyPathomics},
     year         = 2024,
     publisher    = {GitHub},
     url          = {https://github.com/HaoyuCui/PyPathomics},
     version      = {1.0},
   }

 ```



