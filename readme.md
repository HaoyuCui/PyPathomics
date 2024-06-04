<div align=center>
 <img width="1080" alt="image" src="src/.logo.svg">
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

| statistics-types | e.g.                                             |
|------------------|--------------------------------------------------|
| basic            | mean, std                                        |
| distribution     | Q25, Q75, median, IQR, range, skewness, kurtosis |


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


#### Basic Cell Info

| Feature       | Description                                            |
|---------------|--------------------------------------------------------|
| **Name**      | Identifier for the cell                                |
| **Centroid**  | Position of the cell's centroid                        |
| **Cell Type** | Information about the cell type                        |
| **Num**       | Reflects the number of cells of this type of the slide |
| **Ratio**     | Reflects the proportion of cells of this type          |

##### Morphological Features

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

| Feature                                        | Description                                                                                                           |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **ASM (Angular Second Moment)**                | Texture consistency, measuring the similarity between a pixel and its neighbors                                       |
| **Contrast**                                   | Texture contrast, describing the intensity variation in the image                                                     |
| **Correlation**                                | Texture correlation, measuring the similarity between a pixel and its neighbors                                       |
| **Entropy**                                    | Texture entropy, representing the diversity of information in the image; higher values indicate more complex textures |
| **Homogeneity**                                | Texture homogeneity, assessing the consistency of the texture                                                         |
| **IntensityMean** / *Std** / **Max** / **Min** | Mean, standard deviation, maximum, and minimum of the texture intensity                                               |

##### Delaunay Triangle Spatial Features

| Feature         | Description                                                                                |
|-----------------|--------------------------------------------------------------------------------------------|
| **Area**        | Area of the Delaunay triangle around the cell                                              |
| **Perimeter**   | Perimeter of the Delaunay triangle around the cell                                         |
| **Angle_Range** | Difference between the maximum and minimum angles of the Delaunay triangle around the cell |




