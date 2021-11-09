# Tree biomass evaluation using LiDAR

## Introduction

Light detection and ranging (LiDAR) is commonly used in forestry for different applications, among which the evaluation of the trees biomass. Different methods exist to evalute the biomass of a tree from LiDAR data, with varying accuracy, mainly depending on the quality of the data acquisition.

Low-precision LiDAR data is used to get the tree height and crown dimensions that are used to approximate the tree biomass using allometries, while middle-precision to high precision data are used to estimate the tree biomass directly from a reconstruction of the tree structure for the trunk and occasionally the structural branches (*e.g.* see appications of TreeQSM, plantscan3d or AMAPScan). But usually even high resolution data is not sufficient to estimate the biomass of trees as a whole including smaller branches. This is because LiDAR data is often messy at that scale, due to either the laser footprint that is too high compared to the footprint of the structure of interest, or more simply due to wind or occlusions. The resulting point cloud is often good enough to estimate the topology and the length of almost all the structures, but not enough to estimate their diameters properly, which is crucial to compute the volumes, and then the biomass.

This project is an attempt at finding a new method for the evaluation of the tree biomass including most of the tree structure up to the smaller branches by re-estimating the diameters.

For this purpose we set up an experiment on an agroforestry system where we measured two branches per tree on three trees. The measurements included a LiDAR acquisition, and manual measurements on the branches for their topology, dimensions, biomass and a sampling of their fresh and dry wood density.

The estimation of the volume and biomass from LiDAR data is first done using plantscan3d, an OpenAlea software used to estimate tree topology and dimensions. The diameters are estimated at each node using the mean-distance algorithm that uses the LiDAR point-cloud directly. This technique is known to generate a high error rate, especially for smaller strucures.

A second estimation of the volumes and biomass is done using the pipe model algorithm, a well-known method in the field of forestry. The pipe model is used to compute the cross-section of the structures based on the cross-section of their bearer. Applying this method sequentially, all cross-sections are then estimated, and the volume of the structures can then be computed.

Then, we propose a new method, combining botanical variables and statitics to compute the cross-section of the structures. The variables must be computable from the information we get from the LiDAR reconstruction, namely the topology and lengths of the structures.

The manual measurements are used for two purposes:

- as a database to fit the statistical model of the third method, and
- as a reference to assess the estimation of the volume and biomass from the different methods cited before

## Project steps

This project is divided into several steps:

1. Check the measured lenght from the LiDAR data is close to the manual measurements. This is crucial as length are considered well estimated by LiDAR;
2. Check the integrity of the manual measurements by evaluating if the biomass estimated from manual dimensions measurements (length and diameters) at segment scale and an average fresh wood density are close to the reference biomass measured using a scale on the field. This step helps us check if the volumes/biomass estimated using manual measurements can be used as a reference for evaluating the different methods to estimate them from LiDAR point-clouds;
3. Find the variables explaining the cross-section at segment scale. This step is done to fit a statistical model onto the manual measurements dataset;
4. Evaluate the different methods for the evaluation of the tree biomass

## Structure

The project is structured into folders for the data (`0-data`), the code (`1-code`) and the outputs (`2-results`).

### Data

All data related to the project are stored in the `0-data` folder. You can find an extensive description of the folders and files structure in the [README file](0-data/README.md) at its root.

### Code

The code related to the project is stored in the `1-code` folder. All computations are made using the Julia programming language and the [MultiScaleTreeGraph.jl](https://vezy.github.io/MultiScaleTreeGraph.jl/dev/) package. The computations are made in the Julia scripts as follows:

- `1-compute_field_mtg_data.jl`: compute new variables into the manually measured MTG, and export the results as CSV and MTG files in `0-data/1.2-mtg_manual_measurement_corrected_enriched`;
- `2-model_diameter.jl`: fit a model using all variables that can be computed from LiDAR;
- `3-mtg_plantscan3d_to_segments.jl`: put the MTGs from plantscan3d into the same format as used in the manually-measured MTGs;
- `4.0-compute_volume.jl`: compute the structures volumes and biomass, and save the results into `2-results/1-data/df_stats_branch.csv` and `2-results/1-data/df_manual.csv`;

The following Julia scripts are also [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks. It is best to open them using Pluto for a best format.

To open these notebooks, simply add Pluto to your environment and import it:

```julia
using Pkg; Pkg.add("Pluto")
using Pluto
```

Then you can open the notebooks using these commands (copy-paste it in the Julia REPL):

- Step 1: Checking estimated length from LiDAR:

```julia
Pluto.run(notebook = "1-code/4.1-step_1_check_LiDAR_length.jl")
```

- Step 2: Checking manual measurements integrity:

```julia
Pluto.run(notebook = "1-code/4.2-step_2_check_manual_measurement.jl")
```

### Results

The results are all stored in the `2-results` folder. They are divided into three sub-categories:

- output data (`1-data`): computed data, final results of computations and analyses
- figures (`2-plots`): most interesting plots resulting from analyses
- reports (`3-reports`): outputs from the `Pluto.jl` notebooks.
