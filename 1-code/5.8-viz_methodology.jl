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
    alphapoints = 0.5
    fig = Figure(size=(800, 1200))
    g = fig[1:4, 1:2] = GridLayout()
    ax1 = Axis(g[1, 1], aspect=1, title="1. Point cloud", titlealign=:left)
    ax1b = Axis(g[1, 2], aspect=1, title="2. Destructive measurements", titlealign=:left)
    ax2a = Axis(g[2, 1], aspect=1, title="3. Skeleton", titlealign=:left)
    ax2b = Axis(g[2, 2], aspect=1, title="4. Modelling", xlabel="Measured CSA (mm²)", ylabel="Predicted CSA (mm²)", titlealign=:left)
    ax3 = Axis(g[3, 1:2], aspect=1, title="5. Skeleton + Diameters", titlealign=:left)
    ax4 = Axis(g[4, 1:2], aspect=1, title="6. Volume", titlealign=:left)


    hidedecorations!(ax1)
    hidedecorations!(ax1b)
    hidedecorations!(ax2a)
    # hidedecorations!(ax2b)
    hidedecorations!(ax3)
    hidedecorations!(ax4)
    hidespines!(ax1)
    hidespines!(ax1b)
    hidespines!(ax2a)
    hidespines!(ax2b)
    hidespines!(ax3)
    hidespines!(ax4)

    #! Draw th full branch LiDAR point cloud and zoom-in for this part of the branch.
    # Draw the LiDAR point cloud:
    scatter!(ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2)

    # Draw the skeleton (lines):
    scatter!(ax2a, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=alphapoints)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        draw_skeleton!(ax2a, node, xyz_attr, symbol=symbol, linewidth=2)
    end

    # Diameter based on the cross-sectional area at each node
    scatter!(ax3, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=alphapoints)
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
    scatter!(ax4, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=alphapoints)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        mesh!(
            ax4, node[:cyl_sm],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end


    # Get the nodes in each segment:
    nodes_in_segment = Vector{Vector{Node}}(Node[])
    segment_id = [1]
    traverse!(mtg, symbol=symbol) do node
        length(nodes_in_segment) != segment_id[1] && push!(nodes_in_segment, Node[])
        push!(nodes_in_segment[segment_id[1]], node)

        if BiomassFromLiDAR.is_seg(node)
            segment_id[1] += 1
        end
    end

    # Draw the "length" measurements
    arrows_positions = Vector{Tuple{Point3{Float64},Point3{Float64}}}()
    for node_vec in nodes_in_segment
        parent_node = parent(node_vec[1])
        isroot(parent_node) && (parent_node = node_vec[1])

        last_node = node_vec[end]

        push!(
            arrows_positions,
            (
                Point3(parent_node[:XX], parent_node[:YY], parent_node[:ZZ]),
                Point3(last_node[:XX], last_node[:YY], last_node[:ZZ])
            )
        )
    end

    # col_arrows = RGBA(139 / 255, 188 / 255, 182 / 255, 1.0)
    col_arrows = :black
    arrows!(
        ax1b,
        [p[1] + Point3(0.01, -0.01, 0.0) for p in arrows_positions],
        [p[2] - p[1] for p in arrows_positions],
        arrowsize=0.0040, color=col_arrows,
        arrowheadsize=10,
        linewidth=0.0010,
        visible=true
        # label="Length"
    )

    # for i in 1:length(arrows_positions)
    #     ps = arrows_positions[i]
    #     p = ps[1] + 0.5 * ps[2] #- Point3(0.01, -0.01, 0.0)
    #     text!(ax1b, p, text=string(i), justification=:center, color=col_arrows
    #     )
    # end

    # Draw the field "measurement":
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        mesh!(
            ax1b, node[:cyl_sm],
            # color="#BA8C63"
            color=:antiquewhite3
            # color=RGB(202 / 255, 200 / 255, 201 / 255)
        )
    end

    # Get the node in the middle of the segment
    segment_positions = Vector{Point3}()
    for node_vec in nodes_in_segment
        middle_node = node_vec[div(length(node_vec), 2)+1]
        # Take the point in the middle of the segment between the node in the middle of the segment and its parent node:
        parent_node = parent(middle_node)
        i = 0.5 * (middle_node[:XX] + parent_node[:XX]), 0.5 * (middle_node[:YY] + parent_node[:YY]), 0.5 * (middle_node[:ZZ] + parent_node[:ZZ])

        push!(segment_positions, i)
    end

    col = RGB(250 / 255, 178 / 255, 101 / 255)
    scatter!(ax1b, segment_positions, color=col, markersize=10, label="Diameter")

    # Legend of the "Destructive measurement" pane:
    Legend(
        g[1, 2],
        [
            MarkerElement(color=col, marker=:circle, markersize=10, strokecolor=:black),
            MarkerElement(color=col_arrows, marker='➞', markersize=10, strokecolor=:black)
        ],
        ["Diameter", "Length"],
        halign=:left,
        valign=:top,
        # orientation=:horizontal,
        tellheight=false,
        tellwidth=false,
        padding=(60.0f0, 6.0f0, 6.0f0, 6.0f0),
        framevisible=false,
    )

    # General equation of the multilinear model:
    text!(
        ax2b,
        [0], [4000],
        text=L"CSA = \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n + \epsilon",
        color=:black, fontsize=15, valign=:center, halign=:center
    )
    # Add a scatter plot that shows the relationship between the cross-sectional area and other variables:
    # The scatter plot is not real, it is just a representation of the relationship between the cross-sectional area and other variables.

    # Draw points that look like a linear relationship for the scatter plot:
    # Generate some fake data
    x = 1:10:4000
    # Add some random noise to y:
    y = x .+ randn(length(x)) .* 150
    y[y.<=0] .= 0
    lines!(ax2b, x, x, color=:slategrey, linewidth=1, alpha=0.5)
    scatter!(ax2b, x, y, color=:black, markersize=2,)

    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    fig
end

# Save the figure:
save("2-results/2-plots/Fig.0-viz_methodology.png", fig, px_per_unit=5)
