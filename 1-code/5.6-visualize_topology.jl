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
    # Make the cylinder:
    transform!(
        mtg,
        (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], symbol=symbol)) => :cyl,
        (node -> circle_from_radius(node, [:XX, :YY, :ZZ], symbol=symbol, radius_factor=0.0001)) => :circle,
        symbol=symbol
    )
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
LiDAR_directory = "0-data/2-lidar_processing/2-grouped_point_clouds/2-branches"

begin
    branch_index = 1
    LiDAR_subsampling_factor = 50
    lidar_file = filter(x -> occursin("tree", basename(x)), readdir(LiDAR_directory, join=true))[branch_index]
    LiDAR = CSV.read(lidar_file, DataFrame, header=["x", "y", "z", "reflectance", "other"])[1:LiDAR_subsampling_factor:end, :]
    fig2 = Figure(size=(800, 800))
    # ax = Axis(fig2[1, 1], aspect=1)
    g1 = fig2[1, 1] = GridLayout()
    g2 = fig2[2, 1] = GridLayout()
    g3 = fig2[3, 1] = GridLayout()
    g4 = fig2[4, 1] = GridLayout()

    ax1 = Axis(g1[1, 1], aspect=1, title="Point cloud")
    ax2 = Axis(g2[1, 1], aspect=1, title="Skeleton")
    ax3a = Axis(g3[1, 1], aspect=1, title="Point cloud", titlesize=12)
    ax3b = Axis(g3[1, 2], aspect=1, title="PMT", titlesize=12)
    ax3c = Axis(g3[1, 3], aspect=1, title="SM (Ours)", titlesize=12) # Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_pX_p + \epsilon

    ax4 = Axis(g4[1, 1], aspect=1, title="Volume")

    hidedecorations!(ax1; grid=false, minorgrid=false)
    hidedecorations!(ax2; grid=false, minorgrid=false)
    hidedecorations!(ax3a; grid=false, minorgrid=false)
    hidedecorations!(ax3b; grid=false, minorgrid=false)
    hidedecorations!(ax3c; grid=false, minorgrid=false)
    hidedecorations!(ax4; grid=false, minorgrid=false)

    # Draw the LiDAR point cloud:
    # scatter!(ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=LiDAR[:, 4], markersize=0.3)
    scatter!(ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=LiDAR[:, 4], markersize=2)

    # Draw the skeleton (lines):
    traverse!(mtgs[branch_index], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        node_start = ancestors(node, [:XX, :YY, :ZZ], recursivity_level=2, symbol=symbol)
        if length(node_start) != 0
            lines!(
                ax2,
                [node_start[1][1], node[:XX]],
                [node_start[1][2], node[:YY]],
                [node_start[1][3], node[:ZZ]],
                color=get(
                    (ColorSchemes.seaborn_rocket_gradient),
                    node[:branching_order] / max_order
                ),
                linewidth=0.8
            )
        end
    end

    # Skeleton + the circles based on the cross-sectional area at each node
    traverse!(mtgs[branch_index], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        mesh!(
            ax3, node[:circle],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    # Draw the 3d reconstruction:
    traverse!(mtgs[branch_index], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        mesh!(
            ax4, node[:cyl],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    Label(g3[1, 2, Top()], "Diameter",
        fontsize=16,
        font=:bold,
        padding=(0, 0, 20, 0),
        # halign=:right
    )

    colgap!(fig2.layout, 0)
    rowgap!(fig2.layout, 0)
    # resize_to_layout!(fig2)
    fig2
end


let
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1, title="Cross-sectional area")

    traverse!(mtgs[branch_index], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        mesh!(ax, node[:circle], color=:slategrey)
    end
    fig
end

node = mtgs[branch_index][1][1]
xyz_attr = [:XX, :YY, :ZZ]
node_start = ancestors(node, xyz_attr, recursivity_level=2, symbol=symbol)
cyl_csa = Cylinder(
    Point3((node_start[1][1], node_start[1][2], node_start[1][3])),
    Point3((node[xyz_attr[1]], node[xyz_attr[2]], node[xyz_attr[3]])),
    node[:radius] # radius in meter
)

# Make a cylinder of length 1 and radius taken from the node using the previous cylinder direction and origin:
top = cyl_csa.extremity
new_end_point = top + direction(cyl_csa) * 0.01  # Scale the direction vector to the desired length

# Create a new cylinder with the same origin, new end point, and the same radius
new_cyl = Cylinder(top, new_end_point, radius(cyl_csa))