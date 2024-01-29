using MultiScaleTreeGraph
using CairoMakie
using GeometryBasics
using CSV
using DataFrames
using ColorSchemes
using Colors
using Revise
includet("./functions.jl")
using .BiomassFromLiDAR

LiDAR = CSV.read("0-data/4-method_visualization/tree11h_extract.asc", DataFrame, header=["x", "y", "z", "R", "G", "B"])
point_color = CSV.read("0-data/4-method_visualization/tree11h_extract.txt", DataFrame)[:, 4]
symbol = "N"
xyz_attr = [:XX, :YY, :ZZ]
max_order = 5
mtg = read_mtg("0-data/4-method_visualization/tree11h_extract_point_cloud.mtg")
structural_model!(mtg, 0.5, 0.5, π * (0.008^2.0))
branching_order!(mtg) # Do it again, because the structural model does it in basipetal way

# Make the circle for the structural model:
transform!(
    mtg,
    (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_pipe, symbol=symbol)) => :cyl_pipe,
    (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_sm, symbol=symbol)) => :cyl_sm,
    (node -> circle_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_pipe, symbol=symbol, radius_factor=0.0001)) => :circle_pipe,
    (node -> circle_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_sm, symbol=symbol, radius_factor=0.0001)) => :circle_sm,
    symbol=symbol
)

# Make the figure:
begin
    fig = Figure(size=(800, 1200))
    g = fig[1:4, 1:2] = GridLayout()
    ax1 = Axis(g[1, 1], aspect=1, title="Point cloud")
    ax2a = Axis(g[2, 1], aspect=1, title="Skeleton")
    ax3 = Axis(g[3, 1], aspect=1, title="Diameters")
    ax4 = Axis(g[4, 1], aspect=1, title="Volume")

    # Left:
    ax1b = Axis(g[1, 2], aspect=1, title="Destructive measurement")
    ax2b = Axis(g[2, 2], aspect=1, title="Modelling")

    hidedecorations!(ax1)
    hidedecorations!(ax1b)
    hidedecorations!(ax2a)
    hidedecorations!(ax2b)
    hidedecorations!(ax3)
    hidedecorations!(ax4)

    # Draw the LiDAR point cloud:
    scatter!(ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2)

    # Draw the skeleton (lines):
    scatter!(ax2a, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=0.1)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        draw_skeleton!(ax2a, node, xyz_attr, symbol=symbol, linewidth=2)
    end

    # Diameter based on the cross-sectional area at each node
    scatter!(ax3, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=0.1)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:circle_sm] !== nothing) do node
        draw_skeleton!(ax3, node, xyz_attr, symbol=symbol, linewidth=0.8)
        mesh!(
            ax3, node[:circle_sm],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    # Draw the 3d reconstruction:
    scatter!(ax4, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=0.1)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        mesh!(
            ax4, node[:cyl_sm],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end


    # Draw the field "measurement":
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        mesh!(
            ax1b, node[:cyl_sm],
            color="#BA8C63"
        )
    end

    # Get the coordinates of the segments origin:
    segment_positions = descendants(mtg, [:XX, :YY, :ZZ], symbol=symbol, filter_fun=BiomassFromLiDAR.is_seg, type=Vector{Float64})

    #! maybe use the point in the middle?
    scatter!(ax1b, Point3.(segment_positions), color=:red, markersize=10)

    #! add arrows to show measured lenghts

    #! add the general multilinear model equation in the "Modelling" pane

    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    # resize_to_layout!(fig)
    fig
end

# Save the figure:
save("2-results/2-plots/Fig.0-viz_methodology.png", fig, px_per_unit=3)
