using MTG
using CSV
using DataFrames
includet("1-code/functions.jl")
using .BiomassFromLiDAR

# Importing the measurements of wood density
df_dens1 = CSV.read("0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-2-branches-juin.csv", DataFrame)
df_dens2 = CSV.read("0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-4-branches-avril.csv", DataFrame)
df_density = vcat(df_dens1, df_dens2)

dir_path = joinpath("0-data", "4-mtg_lidar_plantscan3d", "3-raw_output_segmentized")
branch = joinpath(dir_path, "tree11h" * ".mtg")
mtg = read_mtg(branch)

@mutate_mtg!(mtg, diameter = node[:radius] * 2 * 1000, symbol = "S") # diameter in mm
@mutate_mtg!(mtg, length = node[:length] * 1000, symbol = "S") # in mm too

@mutate_mtg!(mtg, volume = compute_volume(node), symbol = "S") # volume in mm3
@mutate_mtg!(mtg, volume = compute_volume_axis(node), symbol = "A") # Axis volume in mm3

df = DataFrame(mtg, :volume)
tot_vol = sum(filter(x -> x.symbol == "A", df).volume) * 1e-9 # Total volume in m3
dry_biomass = tot_vol * 0.49 # mass in kg
fresh_biomass = tot_vol * 0.905 # fresh biomass in kg

true_biomass = 6.98
