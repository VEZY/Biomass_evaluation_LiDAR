# Aim: transform mtg from plantscan3d LiDAR reconstruction to match the same format than the
# one used in the mtg from field measurements, i.e. with Axis and segments
# Author: A. Bonnet & R. Vezy
# Date: 10/05/2021


# Script set-up -----------------------------------------------------------

# using Pkg; Pkg.add(url = "https://github.com/VEZY/MTG.jl", rev = "master")

includet("1-code/functions.jl")
using .BiomassFromLiDAR

# Raw MTGs:
segmentize_mtgs(
    joinpath("0-data", "4-mtg_lidar_plantscan3d", "1-raw_output"),
    joinpath("0-data", "4-mtg_lidar_plantscan3d", "3-raw_output_segmentized")
)

# Corrected MTGs:
segmentize_mtgs(
    joinpath("0-data", "4-mtg_lidar_plantscan3d", "2-manually_corrected"),
    joinpath("0-data", "4-mtg_lidar_plantscan3d", "4-corrected_segmentized")
)
