# Tree biomass evaluation using LiDAR

> This is the repository for the code and data used in the paper **Advancing Fine Branch Biomass Estimation with Lidar and Structural Models** ([preprint](https://hal.archives-ouvertes.fr/hal-03838571)).

The data (+code) is available on a dedicated Zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7038482.svg)](https://doi.org/10.5281/zenodo.7038482). If you need to reproduce the results, you can download the data and code from the Zenodo repository and execute the scripts presented below.

## Introduction

Light detection and ranging (lidar) is commonly used in forestry for different applications, such as the evaluation of tree biomass. There are various methods to evaluate the biomass of a tree from lidar data, with accuracy varying based on point-cloud quality.

Low-precision lidar data is used to get the tree height and crown dimensions that are used to approximate the tree biomass using allometries, while middle-precision to high precision data are used to estimate the tree biomass directly from a reconstruction of the tree structure for the trunk and occasionally the structural branches (*e.g.* see applications of TreeQSM, Plantscan3d or AMAPScan). 

Despite high-resolution data, estimating the biomass of entire trees, including smaller branches, remains challenging. Lidar data can be imprecise due to the laser footprint that is high compared to the diameter of finer structures, or more simply due to wind or occlusions. While the resulting point cloud provides topology and length information for most structures, accurately estimating diameters is still a challenge for volume and biomass calculations.

**Project Objective:** This project aims to develop a novel method for evaluating tree biomass, encompassing most of the tree structure down to the smaller branches. The key focus is on re-estimating diameters.

**Experimental Setup:** We conducted an experiment in an agroforestry system, measuring two branches per tree across three trees. Our measurements included lidar acquisitions and manual assessments of branch topology, dimensions, biomass, and fresh and dry wood density.

**Volume and Biomass Estimation:** 

- Plantscan3d: The estimation of the volume and biomass from lidar data is first done using Plantscan3d, an OpenAlea software used to estimate tree topology and dimensions. The diameters are estimated at each node using the mean-distance algorithm that uses the lidar point-cloud directly. This technique is known to generate a high error rate, especially for smaller structures.

- Pipe model theory: A second estimation of the volumes and biomass is done using the pipe model theory (PMT), a well-known method in the field of forestry. The PMT is used to compute the cross-section of the structures based on the cross-section of their bearer. Applying this method recursively, all cross-sections are then estimated, and the volume of the structures can then be computed.

- Structural model (ours): We propose a new method extending the "structural model" approach proposed by Hu et al. (2021). It relies on a multilinear statistical model that leverages not only the tree skeleton but also allometric relationships between the cross-sectional area and other structural features such as the position along the axis or the total sub-tree length.

The manual measurements are used for two purposes:

- as a database to calibrate the structural model
- as a reference to assess the estimation of the volume and biomass from the different methods cited before

<img src="./2-results/2-plots/Figure_8-visualisation_2d_fine.png" width="500" alt="3D reconstruction of a branch and a tree" />

*Figure 1. Co-registered LiDAR point-cloud (left panels) and 3D reconstruction using the structural model (right panels) of a sampled branch (top panels) and a whole tree (bottom panels).*

## Project steps

This project is divided into several steps:

1. Compute the tree/branch skeleton from the point cloud
2. Check the measured length from the LiDAR data is close to the manual measurements. This is crucial as lengths are considered well estimated by LiDAR;
3. Check the integrity of the manual measurements by evaluating if the biomass estimated from manual dimensions measurements (lengths and diameters) at segment scale and an average fresh wood density are close to the reference biomass measured using a scale on the field. This step helps us check if the volumes/biomass estimated using manual measurements can be used as a reference for evaluating the different methods to estimate them from LiDAR point-clouds;
4. Find the variables explaining the cross-section at segment scale and build the structural model;
5. Evaluate the different methods for the estimation of the tree biomass

## Structure

The project is structured into folders for the data (`0-data`), the code (`1-code`) and the outputs (`2-results`).

### Data

All data related to the project are stored in the `0-data` folder. You can find an extensive description of the folders and files structure in the [README file](0-data/README.md) at its root.

### Code

The code related to the project is stored in the `1-code` folder. All computations are made using the Julia programming language and the [MultiScaleTreeGraph.jl](https://vezy.github.io/MultiScaleTreeGraph.jl/dev/) package. The computations are made in pure Julia scripts or [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks for interactivity.

To open a notebook, add Pluto to your Julia environment and import it. To do so, open Julia, and type the following:

```julia
using Pkg; Pkg.add("Pluto")
using Pluto
```

Then you can open the notebooks with the commands given below.

- Step 1: The first step is done in two Julia scripts. Open the following scripts in *e.g.* VS Code, and execute the code from the script to repeat the analysis.
  - the first script (`1.1-compute_field_mtg_data.jl`) computes new variables into the manually measured MTG, and export the results as CSV and MTG files in `0-data/1.2-mtg_manual_measurement_corrected_enriched`;
  - the second script (`1.2-mtg_plantscan3d_to_segments.jl`) transforms the MTGs from Plantscan3d into the same format as used in the manually-measured MTGs;

- Step 2: Checking estimated length from LiDAR:

```julia
Pluto.run(notebook = "1-code/2-step_3_check_LiDAR_length.jl")
```

- Step 3: Checking manual measurements integrity:

```julia
Pluto.run(notebook = "1-code/3-check_manual_measurement.jl")
```

- Step 4: fit models using all variables that can be computed from LiDAR. To execute the notebook, copy and paste the following command into the Julia REPL:

```julia
Pluto.run(notebook = "1-code/4-model_cross_section.jl")
```

- Step 5: analysis.

  - Axis scale:

    ```julia
    Pluto.run(notebook = "1-code/5.1-axis_scale.jl")
    ```

  - Branch scale:

    ```julia
    Pluto.run(notebook = "1-code/5.2-branch_scale.jl")
    ```

  - Tree scale:

    ```julia
    Pluto.run(notebook = "1-code/5.3-tree_scale.jl")
    ```

  - Visualization in 2D (notebook can be slow, there is a faster script for this in `1-code/5.4-visualization_2d_script.jl`):

    ```julia
    Pluto.run(notebook = "1-code/5.4-visualize_2d.jl")
    ```

  - Visualization in 3D:

    ```julia
    Pluto.run(notebook = "1-code/5.5-visualize_3d.jl")
    ```

  - Visualization of the topology:

    ```julia
    Pluto.run(notebook = "1-code/5.6-visualize_topology.jl")
    ```

    - Comparison of the methods:

    ```julia
    Pluto.run(notebook = "1-code/5.7-compare_methods.jl")
    ```

    - Vizualisation to describe the structural model methodology:

    ```julia
    Pluto.run(notebook = "1-code/5.8-viz_methodology.jl")
    ```

### Results

The results are all stored in the `2-results` folder. They are divided into three sub-categories:

- output data (`1-data`): computed data, final results of computations and analyses
- figures (`2-plots`): most interesting plots resulting from analyses
- reports (`3-reports`): outputs from the `Pluto.jl` notebooks.

These files inside these folders are generated by the scripts above.