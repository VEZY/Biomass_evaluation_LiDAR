using MultiScaleTreeGraph
# using CairoMakie
using GLMakie
using GeometryBasics
using CSV
using DataFrames
using ColorSchemes
using Revise
includet("./functions.jl")
using .BiomassFromLiDAR

LiDAR_directory = "0-data/2-lidar_processing/2-grouped_point_clouds/2-branches"
MTG_directory = "0-data/3-mtg_lidar_plantscan3d/2-manually_corrected"
LiDAR_trees_dir = "0-data/2-lidar_processing/2-grouped_point_clouds/1-trees"
MTG_trees_dir = "0-data/3-mtg_lidar_plantscan3d/7-tree_scale"
symbol = "N"

LiDAR_files =
    filter(
        x -> endswith(x, ".txt"), # all MTGs
        readdir(LiDAR_directory)
    )
MTG_files =
    filter(
        x -> endswith(x, ".mtg"), # all MTGs
        readdir(MTG_directory)
    )
branches_MTG = sort!(replace.(MTG_files, ".mtg" => ""))
branches_LiDAR = sort!([match(r"tree[0-9]{1,2}[a-z]", i).match for i in LiDAR_files])


branch = "tree11h"
tree = "12"

LiDAR = CSV.read(joinpath(LiDAR_directory, branch * ".txt"), DataFrame, header=["x", "y", "z", "reflectance", "other"]);
LiDAR_files_tree =
    filter(
        x -> endswith(x, ".txt"), # all MTGs
        readdir(LiDAR_trees_dir)
    )

MTG_files_tree =
    filter(
        x -> endswith(x, ".mtg"), # all MTGs
        readdir(MTG_trees_dir)
    )

trees_LiDAR = sort!([match(r"[0-9]{1,2}", i).match for i in LiDAR_files_tree])
trees_MTG = sort!([match(r"[0-9]{1,2}", i).match for i in MTG_files_tree])
LiDAR_tree = CSV.read(joinpath(LiDAR_trees_dir, "all_scans_tree_" * tree * ".txt"), DataFrame, header=["x", "y", "z", "reflectance", "other"]);

mtg = read_mtg(joinpath(MTG_directory, branch * ".mtg"))
structural_model!(mtg, 0.5, 0.5, 1999.0)
branching_order!(mtg) # Do it again, because the structural model does it in basipetal way
transform!(
    mtg,
    # Make the circle for the structural model:
    (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_sm, symbol=symbol)) => :cyl_sm,
    symbol=symbol
)

# Computing min/max values for colouring:
z_values = descendants(mtg, :ZZ, ignore_nothing=true)
z_min = minimum(z_values)
z_max = maximum(z_values);

mtg_tree = read_mtg(joinpath(MTG_trees_dir, "all_scans_tree_" * tree * ".mtg"))
# structural_model!(mtg_tree, 0.5, 0.5, 128524.0)
structural_model!(mtg_tree, 0.5, 0.5, 240000.0)
branching_order!(mtg_tree) # Do it again, because the structural model does it in basipetal way
transform!(
    mtg_tree,
    (node -> cylinder_from_radius(node, [:XX, :ZZ, :YY], radius=:radius_sm, symbol=symbol)) => :cyl_sm,
    symbol=symbol
)
z_values_tree = descendants!(mtg_tree, :ZZ, ignore_nothing=true)
z_min_tree = minimum(z_values_tree)
z_max_tree = maximum(z_values_tree)
point_color = [get(ColorSchemes.viridis, (i - z_min_tree) / (z_max_tree - z_min_tree)) for i in LiDAR_tree[:, 3]]

begin
    # Plot the LiDAR point cloud
    fig = Figure()
    # ax1 = Axis3(fig[1,1], elevation = 0.0π)
    ax1 = Axis(fig[1, 1])
    hidedecorations!(ax1)
    # hidespines!(ax1)
    ax1.title = "LiDAR point cloud"
    ax2 = Axis(fig[1, 2])
    hidedecorations!(ax2)
    # hidespines!(ax2)
    ax2.title = "3D reconstruction"

    ax3 = Axis(fig[2, 1])
    hidedecorations!(ax3)
    # hidespines!(ax3)
    ax4 = Axis(fig[2, 2])
    hidedecorations!(ax4)
    # hidespines!(ax4)

    scatter!(ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=LiDAR[:, 3], markersize=0.3)
    traverse!(mtg, symbol="N", filter_fun=node -> node[:cyl_sm] !== nothing) do node
        mesh!(ax2, node[:cyl_sm], color=get(ColorSchemes.viridis, (node[:ZZ] - z_min) / (z_max - z_min)))
    end

    scatter!(ax3, LiDAR_tree[:, 1], LiDAR_tree[:, 3], LiDAR_tree[:, 2], color=LiDAR_tree[:, 3], markersize=0.3)
    traverse!(mtg_tree, symbol="N", filter_fun=node -> node[:cyl_sm] !== nothing) do node
        cols = [get(ColorSchemes.viridis, (i[2] - z_min_tree) / (z_max_tree - z_min_tree)) for i in GeometryBasics.coordinates(node[:cyl_sm])]
        mesh!(ax4, node[:cyl_sm], color=cols)
    end
    # rowgap!(fig.layout, 10)
    # colgap!(fig.layout, 0.5)
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))
    resize_to_layout!(fig)
    fig
end
save("2-results/2-plots/step_5_visualisation_2d_GLMakie.png", fig, px_per_unit=3)
