using MTG
using CSV
using Plots
using DataFrames
using StatsPlots
using Statistics
includet("1-code/functions.jl")
using .BiomassFromLiDAR

# Declaring the paths to the files directories:
dir_path_lidar = joinpath("0-data", "4-mtg_lidar_plantscan3d", "3-raw_output_segmentized")
dir_path_manual = joinpath("0-data", "2-mtg_manual_measurement_corrected_enriched")

# Importing the measurements of wood density
df_dens1 = CSV.read("0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-2-branches-juin.csv", DataFrame, normalizenames = true)
df_dens2 = CSV.read("0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-4-branches-avril.csv", DataFrame, normalizenames = true)
df_density = vcat(df_dens1, df_dens2)

select!(
    df_density,
    :branches,
    [:dry_weight_g, :parafilm_volume_deducted_cm3] => ((x, y) -> x ./ y) => :dry_density,
    [:fresh_weight_g, :parafilm_volume_deducted_cm3] => ((x, y) -> x ./ y) => :fresh_density
)

df_density = groupby(df_density, :branches)
df_density = combine(df_density, :dry_density => mean, :fresh_density => mean, renamecols = false)

# Defining the branch:
branch = "tree11h"

joinpath(dir_path_lidar, branch * ".mtg")

mtg_files =
    filter(
        x -> splitext(basename(x))[2] in [".mtg"],
        readdir(dir_path_lidar)
    )

branches = first.(splitext.(mtg_files))

df_branch = DataFrame(:branch => String[], :variable => String[],  :model => String[], :error => Float64[], :error_norm => Float64[])

for i in branches
    println("Computing branch $i")
    df = compare_model_branch(i, dir_path_lidar, dir_path_manual, df_density)
    df[!,:branch] .= i
    df_branch = vcat(df_branch, df)
end


bar(df_branch.branch, df_branch.error)

@df df_branch scatter(
    :branch,
    :error,
    group = :model
)


@df df_branch bar(
    :branch,
    :error,
    group = :model
)
