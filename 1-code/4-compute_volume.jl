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
includet("functions.jl")
using Main.BiomassFromLiDAR

# Declaring the paths to the files directories:
dir_path_lidar = joinpath("0-data", "3-mtg_lidar_plantscan3d", "4-corrected_segmentized")
dir_path_manual = joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched")

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

df_branch = DataFrame(
    :branch => String[],
    :variable => String[],
    :measurement => Float64[],
    :prediction => Float64[],
    :model => String[],
    :error => Float64[],
    :error_norm => Float64[]
    )

for i in branches
    println("Computing branch $i")
    df = compare_model_branch(i, dir_path_lidar, dir_path_manual, df_density)
    df[!,:branch] .= i
    df_branch = vcat(df_branch, df)
end

bar(df_branch.branch, df_branch.error, group = df_branch.model .* df_branch.variable)

BrowseTables.open_html_table(df_branch)

gdf_branch = groupby(df_branch, [:variable, :model])

stats =
combine(
    gdf_branch,
    [:measurement, :prediction] => RMSE => :RMSE,
    [:measurement, :prediction] => EF => :EF
)

stats_length = filter(x -> x.variable == "length", stats)
stats_volume = filter(x -> x.variable == "volume", stats)
stats_biomass = filter(x -> x.variable == "biomass", stats)

# using AlgebraOfGraphics:
# xy =
#     data(df_branch) *
#     mapping(:measurement, :prediction, layout = :variable, color = :model) +
#     data(df_branch) *
#     mapping(:measurement, :measurement, layout = :variable) *
#     visual(Lines)
# draw(xy, axis = (aspect = 1, xscale = , yscale = identity))

p_length = @df filter(x -> x.variable == "length", df_branch) scatter(
    :measurement,
    :prediction,
    group = :model,
    label = hcat(
        "plantscan3d, RMSE: $(stats_length.RMSE[1]), EF: $(stats_length.EF[1])",
        "Stat. mod. ⌀<20mm, RMSE: $(stats_length.RMSE[2]), EF: $(stats_length.EF[2])"
        ),
    yguide = "LiDAR length (m)",
    xguide = "Manual length (m)",
    xlims = (0.0, 100.0),
    ylims = (0.0, 100.0),
    legend = :bottomright
)
Plots.abline!(1,0, line = :dash, label = "identity")

p_vol = @df filter(x -> x.variable == "volume", df_branch) scatter(
    :measurement,
    :prediction,
    group = :model,
    label = hcat(
        "plantscan3d, RMSE: $(stats_volume.RMSE[1]), EF: $(stats_volume.EF[1])",
        "Stat. mod. ⌀<20mm, RMSE: $(stats_volume.RMSE[2]), EF: $(stats_volume.EF[2])"
        ),
    yguide = "Predicted volume (m³)",
    xguide = "Measured volume (m³)",
    xlims = (0.0, 0.04),
    ylims = (0.0, 0.04),
    legend = :bottomright
)
Plots.abline!(1,0, line = :dash, label = "identity")

p_biomass = @df filter(x -> x.variable == "biomass", df_branch) scatter(
    :measurement,
    :prediction,
    group = :model,
        label = hcat(
        "plantscan3d, RMSE: $(stats_biomass.RMSE[1]), EF: $(stats_biomass.EF[1])",
        "Stat. mod. ⌀<20mm, RMSE: $(stats_biomass.RMSE[2]), EF: $(stats_biomass.EF[2])"
        ),
    yguide = "Predicted biomass (kg)",
    xguide = "Measured biomass (kg)",
    xlims = (0.0, 40),
    ylims = (0.0, 40),
    legend = :bottomright
)
Plots.abline!(1,0, line = :dash, label = "identity")

l = @layout [a ; b c]
plot(p_length, p_vol, p_biomass, layout = l)
