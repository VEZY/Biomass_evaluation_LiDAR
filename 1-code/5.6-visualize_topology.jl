using MultiScaleTreeGraph
using CairoMakie
using GeometryBasics
using CSV
using DataFrames
using ColorSchemes

includet("./functions.jl")
using .BiomassFromLiDAR

# MTG_directory = "0-data/3-mtg_lidar_plantscan3d/1-raw_output"
MTG_directory = "0-data/3-mtg_lidar_plantscan3d/2-manually_corrected"

# List all mtg files in the folder:
MTG_files =
    filter(
        x -> endswith(x, ".mtg"), # all MTGs
        readdir(MTG_directory)
    )

# Read all MTGs and compute the cylinders of all branches:
symbol = "N"
mtgs = []
max_order = Int[]
xy_bottom = []
branches_MTG = String[]
for i in MTG_files
    mtg = read_mtg(joinpath(MTG_directory, i))
    transform!(mtg, (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], symbol=symbol)) => :cyl, symbol=symbol)

    branching_order!(mtg)
    push!(max_order, maximum(traverse(mtg, node -> node[:branching_order])))
    push!(mtgs, mtg)
    x_bottom = minimum(traverse(mtg, node -> node[:XX], symbol=symbol))
    y_bottom = minimum(traverse(mtg, node -> node[:YY], symbol=symbol))
    push!(xy_bottom, [x_bottom, y_bottom])
    branch_name = replace(i, ".mtg" => "")
    push!(branches_MTG, string(branch_name[5:end-1], "-", branch_name[end]))
end

max_order = maximum(max_order)

# Plot the LiDAR point cloud for all branches at once:
begin
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1)
    hidedecorations!(ax)
    for (i, mtg) in enumerate(mtgs)
        if i < 4
            ax = Axis(fig[1, i], aspect=1)
        else
            ax = Axis(fig[2, i-3], aspect=1)
        end
        traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
            mesh!(
                ax,
                node[:cyl],
                color=get(
                    (ColorSchemes.seaborn_rocket_gradient),
                    node[:branching_order] / max_order
                )
            )
        end

        errorbars!(ax, [xy_bottom[i][1]], [xy_bottom[i][2]], 0.1, whiskerwidth=5, direction=:x, color="black")
        text!(
            ax,
            [xy_bottom[i][1]], [xy_bottom[i][2]],
            text="10 cm",
            align=(:center, :bottom),
            fontsize=9,
            offset=(2, 3)
        )
        hidedecorations!(ax)
        ax.title = "$(branches_MTG[i])"
    end
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)
    # resize_to_layout!(fig)

    Colorbar(
        fig[1:2, 4],
        limits=(0, max_order),
        colormap=cgrad(:seaborn_rocket_gradient, max_order, categorical=true),
        size=25,
        label="Branching order"
    )
    fig
end

save("2-results/2-plots/step_5_visualisation_topology.png", fig, px_per_unit=3)

# Visualize the topology of one part of a branch:
begin
    fig2 = Figure()
    ax = Axis(fig2[1, 1], aspect=1)
    hidedecorations!(ax)
    traverse!(mtgs[1], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        mesh!(ax, node[:cyl], color=get((ColorSchemes.RdBu_8), node[:branching_order] / max_order[1]))
    end
    fig2
end
