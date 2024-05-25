# PyPathomics

> PyPathomics is an open-source software for gigapixel whole-slide image analysis.

The PyPathomics is under development. The current version is 0.1.0.

## Installation

 ```bash
conda create -n pypathomics
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
   statistic-types: ['basic', 'distribution']
```

Where, 
- `openslide-home`: Specifies the path to the OpenSlide library.
- `feature-set`:  A list of feature sets to extract. Options: `Morph`, `Texture`, `Triangle`.
- `cell_types`: Types of cells to analyze. Options: `I` (inflammatory), `S`, (stromal), `T` (tumor).
- `statistic-types`: Types of statistics to calculate. Options: `basic`, `distribution`.


#### Options for main.py

Required Arguments:
```text
    --config     Specify the configuration file path
    --json       Path to the json directory or file from Hover-Net
    --wsi        Path to the WSI directory or file
    --ext        WSI file extension (default: .svs)
    --buffer     Specify the output buffer dir for preprocessing
    --output     Set the output directory for the analysis
```


Optional Arguments:  
```text
    -f           Run on a single file (default: directory mode) 
    --auto_skip  Skip existing directories automatically (default: True)
    --level      Detail level of the WSI to analyze (default: 0)
``` 


### Usage: 

1. Make sure you run the Hover-Net and get the json files. Check the config file before run.

2. Analyze a Directory:
    ```bash
    python main.py --json /path/to/json_dir --wsi /path/to/wsi_dir --buffer /path/to/buffer --ext .svs --output /path/to/output
    ```

3. Analyze a Single File
    ```bash
    python main.py -f --json /path/to/json_file --wsi /path/to/wsi_file --buffer /path/to/buffer --ext .svs --output /path/to/output
    ```


### Feature set:

#### Basic Cell Info

- **Name**: Identifier for the cell, used to distinguish different cells, facilitating data management and analysis.
- **Centroid**: Position of the cell's centroid, used to analyze spatial distribution and evaluate cell interactions or aggregation.
- **Cell Type**: Information about the cell type, crucial for type-based classification and disease state identification.
- **Num**: Reflects the number of cells of this type.
- **Ratio**: Reflects the proportion of cells of this type.

##### Morphological Features

- **Morph_Area**: Area of the cell, indicating cell size, useful for differentiating cell types or growth states.
- **Morph_AreaBbox**: Area of the minimum bounding rectangle around the cell, useful for assessing shape regularity.
- **Morph_CellEccentricities**: Eccentricity of the cell, indicating the roundness or ellipticity, used for morphological classification.
- **Morph_Circularity**: Roundness of the cell, assessing how close the shape is to a perfect circle.
- **Morph_Elongation**: Elongation rate of the cell, indicating the degree of cell shape elongation.
- **Morph_Extent**: Proportion of the cell occupying its bounding rectangle, reflecting the compactness of the cell filling.
- **Morph_MajorAxisLength/Morph_MinorAxisLength**: Lengths of the major and minor axes of the fitted ellipse for the cell, describing cell shape.
- **Morph_Perimeter**: Perimeter of the cell boundary, related to cell complexity and shape regularity.
- **Morph_Solidity**: Ratio of the cell area to its convex hull area, assessing the concavity of the cell shape.
- **Morph_CurvMean/Morph_CurvStd/Morph_CurvMax/Morph_CurvMin**: Mean, standard deviation, maximum, and minimum of the cell boundary curvature, reflecting boundary smoothness and regularity.

##### Texture Features

- **Texture_ASM (Angular Second Moment)**: Texture consistency, indicating image smoothness.
- **Texture_Contrast**: Texture contrast, describing the intensity variation in the image.
- **Texture_Correlation**: Texture correlation, measuring the similarity between a pixel and its neighbors.
- **Texture_Entropy**: Texture entropy, representing the diversity of information in the image; higher values indicate more complex textures.
- **Texture_Homogeneity**: Texture homogeneity, assessing the consistency of the texture.
- **Texture_IntensityMean/Texture_IntensityStd/Texture_IntensityMax/Texture_IntensityMin**: Mean, standard deviation, maximum, and minimum of the texture intensity, reflecting brightness characteristics of the image.

##### Delaunay Triangle Spatial Features

- **Triangle_Area**: Area of the Delaunay triangle around the cell, reflecting cell population density; negatively correlated with density.
- **Triangle_Perimeter**: Perimeter of the Delaunay triangle around the cell, indicating cell clustering features; negatively correlated with cell density. Changes in perimeter can indicate cell mobility, useful for studying cell migration and invasiveness.
- **Triangle_Angle_Range**: Difference between the maximum and minimum angles of the Delaunay triangle around the cell, reflecting distribution characteristics; larger differences usually indicate increased distribution non-uniformity.
- **IoT (inflammatory cells' Voronoi area over tumor cells' Voronoi area)**: Ratio of the Voronoi area of inflammatory cells to tumor cells, calculated using the Voronoi diagram. IoT can reveal the infiltration degree of immune cells in the tumor microenvironment, valuable for evaluating the immune response state of the tumor.
- **IoS (inflammatory cells' Voronoi area over stroma cells' Voronoi area)**: Ratio of the Voronoi area of inflammatory cells to stroma cells, calculated using the Voronoi diagram. IoS helps understand the role of stroma cells in tumor growth and metastasis.
- **SoT (stroma cells' Voronoi area over tumor cells' Voronoi area)**: Ratio of the Voronoi area of stroma cells to tumor cells, reflecting the interaction intensity between tumor cells and stroma cells.



