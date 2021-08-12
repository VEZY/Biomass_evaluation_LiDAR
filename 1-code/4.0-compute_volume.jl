using MTG
using CSV
# using Plots
using DataFrames
using StatsPlots
using Statistics
using BrowseTables
using Plots
# using CairoMakie
# using AlgebraOfGraphics
using Revise
includet("1-code/functions.jl")
using Main.BiomassFromLiDAR

# Declaring the paths to the files directories:
dir_path_lidar = joinpath("0-data", "3-mtg_lidar_plantscan3d", "4-corrected_segmentized")
dir_path_lidar_raw = joinpath("0-data", "3-mtg_lidar_plantscan3d", "3-raw_output_segmentized")
dir_path_manual = joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched")

# Importing the measurements of wood density
df_dens1 = CSV.read("0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-2-branches-juin.csv", DataFrame, normalizenames = true)
df_dens2 = CSV.read("0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-4-branches-avril.csv", DataFrame, normalizenames = true)
df_density = vcat(df_dens1, df_dens2)

select!(
    df_density,
    :branches,
    [:dry_weight_g, :volume_without_parafilm_cm3] => ((x, y) -> x ./ y) => :dry_density,
    :conventional_method_density => :fresh_density
    # [:fresh_weight_g, :parafilm_volume_deducted_cm3] => ((x, y) -> x ./ y) => :fresh_density
)

df_density = groupby(df_density, :branches)
df_density = combine(df_density, :dry_density => mean, :fresh_density => mean, renamecols = false)

mtg_files =
    filter(
        x -> splitext(basename(x))[2] in [".mtg"],
        readdir(dir_path_lidar)
    )

branches = first.(splitext.(mtg_files))

df_stats_branch = DataFrame(
    :branch => String[],
    :variable => String[],
    :measurement => Float64[],
    :prediction => Float64[],
    :model => String[],
    :error => Float64[],
    :error_norm => Float64[]
    )

df_manual = DataFrame(
    :branch => String[],
    :id => Int[],
    :symbol => String[],
    :scale => Int[],
    :index => Int[],
    :parent_id => Int[],
    :link => Float64[],
    :mass_g => Float64[],
    :fresh_mass => Float64[]
    )

for i in branches
    println("Computing branch $i")
    (mtg_manual, mtg_lidar_ps3d, mtg_lidar_ps3d_raw, mtg_lidar_model) =
        compute_volume_model(i, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, df_density)
    df = volume_stats(mtg_manual, mtg_lidar_ps3d, mtg_lidar_ps3d_raw, mtg_lidar_model, df_density)
    df[!,:branch] .= i
    df_stats_branch = vcat(df_stats_branch, df)

    df = DataFrame(mtg_manual, [:mass_g, :fresh_mass])
    df[!,:branch] .= i
    df_manual = vcat(df_manual, df[:,Not(:tree)])
end

CSV.write("2-results/1-data/df_stats_branch.csv", df_stats_branch)
CSV.write("2-results/1-data/df_manual.csv", df_manual)
