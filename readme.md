<div align=center>
 <img width="1106" alt="image" src="https://github.com/HaoyuCui/PyPathomics/assets/75052311/93badd07-b488-45a4-a97c-78d503d02260">
</div>

# PyPathomics

> PyPathomics is an open-source software for gigapixel whole-slide image analysis.

The PyPathomics is under development. The current version is 0.1.0.

Support for:
- [x] Hover-Net
[[Repo](https://github.com/vqdang/hover_net)] [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub)] (Graham et al., 2019)

- [x] Cerberus
[[Repo](https://github.com/TissueImageAnalytics/cerberus)] [[Paper](https://doi.org/10.1016/j.media.2022.102685)] (Graham et al., 2023)

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
   openslide-home: path\to\openslide-home
   feature-set: ['Morph', 'Texture', 'Triangle']
   cell_types: ['I', 'S', 'T']
   statistic-types: ['basic', 'distribution']
```

Where, 
- `openslide-home`: Specifies the path to the OpenSlide library. (for Windows only, leave it empty for Linux)
- `feature-set`:  A list of feature sets to extract. Options: `Morph`, `Texture`, `Triangle`.
- `cell_types`: Types of cells to analyze. Options: `I` (inflammatory), `S`, (stromal), `T` (tumor).
- `statistic-types`: Types of statistics to calculate. Options: `basic`, `distribution`.


#### Options for main.py

Required Arguments:
```text
    --config     Specify the configuration file path
    --seg       Path to the seg directory or file from Hover-Net(.json) or Cerberus(.dat)
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
    python main.py --seg /path/to/seg_dir --wsi /path/to/wsi_dir --buffer /path/to/buffer --ext .svs --output /path/to/output
    ```

4. Analyze a Single File
    ```bash
    python main.py -f --seg /path/to/seg_file --wsi /path/to/wsi_file --buffer /path/to/buffer --ext .svs --output /path/to/output
    ```


### Feature set:

#### Basic Cell Info

- **Name**: Identifier for the cell
- **Centroid**: Position of the cell's centroid
- **Cell Type**: Information about the cell type
- **Num**: Reflects the number of cells of this type of the slide
- **Ratio**: Reflects the proportion of cells of this type

##### Morphological Features

- **Morph_Area**: Area of the cell, indicating cell size
- **Morph_AreaBbox**: Area of the minimum bounding rectangle around the cell
- **Morph_CellEccentricities**: Eccentricity of the cell
- **Morph_Circularity**: Roundness of the cell
- **Morph_Elongation**: Elongation rate of the cell
- **Morph_Extent**: Proportion of the cell occupying its bounding rectangle
- **Morph_MajorAxisLength/Morph_MinorAxisLength**: Lengths of the major and minor axes of the fitted ellipse for the cell
- **Morph_Perimeter**: Perimeter of the cell boundary
- **Morph_Solidity**: Ratio of the cell area to its convex hull area
- **Morph_CurvMean/Morph_CurvStd/Morph_CurvMax/Morph_CurvMin**: Mean, standard deviation, maximum, and minimum of the cell boundary curvature

##### Texture Features

- **Texture_ASM (Angular Second Moment)**: Texture consistency, measuring the similarity between a pixel and its neighbors.
- **Texture_Contrast**: Texture contrast, describing the intensity variation in the image.
- **Texture_Correlation**: Texture correlation, measuring the similarity between a pixel and its neighbors.
- **Texture_Entropy**: Texture entropy, representing the diversity of information in the image; higher values indicate more complex textures.
- **Texture_Homogeneity**: Texture homogeneity, assessing the consistency of the texture.
- **Texture_IntensityMean/Texture_IntensityStd/Texture_IntensityMax/Texture_IntensityMin**: Mean, standard deviation, maximum, and minimum of the texture intensity

##### Delaunay Triangle Spatial Features

- **Triangle_Area**: Area of the Delaunay triangle around the cell
- **Triangle_Perimeter**: Perimeter of the Delaunay triangle around the cell
- **Triangle_Angle_Range**: Difference between the maximum and minimum angles of the Delaunay triangle around the cell
- **IoT**: Ratio of the Voronoi area of inflammatory cells to tumor cells
- **IoS**: Ratio of the Voronoi area of inflammatory cells to stroma cells
- **SoT**: Ratio of the Voronoi area of stroma cells to tumor cells



