using MultiScaleTreeGraph
using CSV
# using Plots
using DataFrames
using Statistics
using BrowseTables
using Revise
includet("./functions.jl")
using .BiomassFromLiDAR

# Declaring the paths to the files directories:
dir_path_lidar = joinpath("0-data", "3-mtg_lidar_plantscan3d", "5-corrected_segmentized_id")
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
        x -> splitext(basename(x))[2] in [".xlsx"],
        readdir(dir_path_lidar)
    )

branches = first.(splitext.(mtg_files))

df_stats_branch = DataFrame(
    :branch => String[],
    :variable => String[],
    :measurement => Float64[],
    :prediction => Float64[],
    :model => String[]
)

df_all = DataFrame(
    :origin => String[],
    :branch => String[],
    :id => Int[],
    :symbol => String[],
    :scale => Int[],
    :index => Int[],
    :parent_id => Int[],
    :link => Float64[],
    :mass_g => Float64[],
    :fresh_mass => Float64[],
    :volume => Float64[],
    :length => Float64[],
    :cross_section => Float64[],
    :id_cor => Int[]
    )

for i in branches
    println("Computing branch $i")
    (mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model) =
        compute_volume_model(i, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, df_density)
    df = volume_stats(mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model, df_density)
    df[!,:branch] .= i
    df_stats_branch = vcat(df_stats_branch, df)

    # Manual measurement:
    df_manual = DataFrame(mtg_manual, [:mass_g, :cross_section, :length_gap_filled, :fresh_mass, :volume_gf, :id_cor])
    df_manual[!,:branch] .= i
    rename!(df_manual, Dict(:volume_gf => "volume", :length_gap_filled => "length"))
    df_manual[!,:origin] .= "measurement"

    # plantscan3d, as-is:
    df_ps3d_raw = DataFrame(mtg_lidar_ps3d_raw, [:fresh_mass, :volume_ps3d])
    rename!(df_ps3d_raw, Dict(:volume_ps3d => "volume"))
    df_ps3d_raw[!,:branch] .= i
    # df_ps3d_raw[!,:id_cor] .= missing
    df_ps3d_raw[!,:origin] .= "plantscan3d, raw"

    # statistical model:
    df_stat_mod = DataFrame(
        mtg_lidar_model,
        [
            :length_sim,
            :cross_section_ps3d,
            :cross_section_pipe,
            :cross_section_pipe_50,
            :cross_section_stat_mod_50,
            :cross_section_stat_mod,
            :fresh_mass,
            :fresh_mass_50,
            :fresh_mass_ps3d,
            :fresh_mass_pipe_mod,
            :fresh_mass_pipe_mod_50,
            :volume_stat_mod,
            :volume_stat_mod_50,
            :volume_ps3d,
            :volume_pipe_mod,
            :volume_pipe_mod_50,
            :id_cor
        ]
    )

    df_stat_mod_biomass = select(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :id_cor,
            :fresh_mass => "stat. mod.",
            :fresh_mass_50 => "stat. mod. ⌀<50",
            :fresh_mass_ps3d => "plantscan3d",
            :fresh_mass_pipe_mod => "Pipe model",
            :fresh_mass_pipe_mod_50 => "Pipe mod. ⌀<50"
    )

    df_stat_mod_biomass = stack(
            df_stat_mod_biomass,
            ["stat. mod.", "stat. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor],
            variable_name = :origin,
            value_name = :fresh_mass
        )


    df_stat_mod_cs = select(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :id_cor,
            :cross_section_stat_mod => "stat. mod.",
            :cross_section_stat_mod_50 => "stat. mod. ⌀<50",
            :cross_section_ps3d => "plantscan3d",
            :cross_section_pipe => "Pipe model",
            :cross_section_pipe_50 => "Pipe mod. ⌀<50"
    )

    df_stat_mod_cs = stack(
            df_stat_mod_cs,
            ["stat. mod.", "stat. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor],
            variable_name = :origin,
            value_name = :cross_section
        )

    select!(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :length_sim => :length, :id_cor,
            :volume_stat_mod => "stat. mod.",
            :volume_stat_mod_50 => "stat. mod. ⌀<50",
            :volume_ps3d => "plantscan3d",
            :volume_pipe_mod => "Pipe model",
            :volume_pipe_mod_50 => "Pipe mod. ⌀<50"
    )

    df_stat_mod = stack(
            df_stat_mod,
            ["stat. mod.", "stat. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor, :length],
            variable_name = :origin,
            value_name = :volume
        )

    df_stat_mod = leftjoin(df_stat_mod, df_stat_mod_biomass[:,[:origin, :id, :fresh_mass]], on = [:origin,:id])
    df_stat_mod = leftjoin(df_stat_mod, df_stat_mod_cs[:,[:origin, :id, :cross_section]], on = [:origin,:id])
    df_stat_mod[!,:branch] .= i

    df_all = vcat(
        df_all,
        df_manual[:,Not(:tree)],
        df_ps3d_raw[:,Not(:tree)],
        df_stat_mod,
        cols = :union
        )
end

# sum(filter(x -> x.origin == "measurement" && x.branch == "tree11h" && x.scale == 3, df_all).length) / 1000
CSV.write("2-results/1-data/df_stats_branch.csv", df_stats_branch)
CSV.write("2-results/1-data/df_all.csv", df_all)

# using Plots

# df_manual = filter(x -> x.origin == "measurement", df_all)
# scatter(df_manual.mass_g, df_manual.fresh_mass)


# df_manual_fresh_biomass =
#     select(filter(x -> x.origin == "measurement" && x.id_cor !== missing, df_all),
#     :branch, :id_cor,
#     :fresh_mass => :fresh_mass_meas,
#     :volume => :volume_meas)

# df_compare = leftjoin(
#     df_manual_fresh_biomass,
#     filter(x -> x.origin != "measurement" && x.id_cor !== missing, df_all),
#     on = [:branch, :id_cor]
#     )


# scatter(df_compare.fresh_mass_meas, df_compare.fresh_mass)
# scatter(df_compare.volume_meas, df_compare.volume)
