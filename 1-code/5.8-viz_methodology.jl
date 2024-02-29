using MultiScaleTreeGraph
# using CairoMakie
using GLMakie
using GeometryBasics
using CSV
using DataFrames
using ColorSchemes
using Colors
using Revise
includet("./functions.jl")
using .BiomassFromLiDAR

GLMakie.activate!()

# LiDAR = CSV.read("0-data/4-method_visualization/tree11h_extract.asc", DataFrame, header=["x", "y", "z", "R", "G", "B"])
LiDAR = CSV.read("0-data/4-method_visualization/tree11h_extract.txt", DataFrame, header=["x", "y", "z", "reflectance", "other"], skipto=2)
select!(LiDAR, [:x, :y, :z] => ((x, y, z) -> Point3.(x, y, z)) => :point, :reflectance)
LiDAR_fullbranch = CSV.read("0-data/2-lidar_processing/2-grouped_point_clouds/2-branches/tree11h.txt", DataFrame, header=["x", "y", "z", "reflectance", "other"])
select!(LiDAR_fullbranch, [:x, :y, :z] => ((x, y, z) -> Point3.(x, y, z)) => :point, :reflectance)
LiDAR_fulltree = CSV.read("0-data/2-lidar_processing/2-grouped_point_clouds/1-trees/all_scans_tree_11.txt", DataFrame, header=["x", "y", "z", "reflectance", "other"])
select!(LiDAR_fulltree, [:x, :y, :z] => ((x, y, z) -> Point3.(x, y, z)) => :point, :reflectance)
# Three different colors for the point cloud. slategrey for the full tree, red for the points that are from the full branch and 
# green for the branch extraction.

LiDAR_fulltree_points = [row.point for row in eachrow(LiDAR_fulltree)]
LiDAR_fullbranch_points = [row.point for row in eachrow(LiDAR_fullbranch)]
LiDAR_points = [row.point for row in eachrow(LiDAR)]

# Create a Dict where the keys are points and the values are colors
point_colors = Dict{typeof(LiDAR_fulltree_points[1]),Tuple{Symbol,Float64}}()
marker_sizes = Dict{typeof(LiDAR_fulltree_points[1]),Float64}()
# Assign colors to points based on their origin
for point in LiDAR_fulltree_points
    point_colors[point] = (:slategrey, 0.3)  # Default color
    marker_sizes[point] = 0.01
end
for point in LiDAR_fullbranch_points
    point_colors[point] = (:green, 1.0)  # Points from the branch
    marker_sizes[point] = 1.0
end
for point in LiDAR_points
    point_colors[point] = (:red, 1.0)  # Points from the extract
    marker_sizes[point] = 1.0
end

# Now you can create the color array for the full tree
LiDAR_fulltree_colors = [point_colors[point] for point in LiDAR_fulltree_points]
LiDAR_fulltree_markers = [marker_sizes[point] for point in LiDAR_fulltree_points]

begin
    f = Figure()
    l = LScene(f[1, 1], show_axis=true)
    # ax = Axis3(f[1, 1])
    scatter!(l, LiDAR_fulltree[:, 1], color=LiDAR_fulltree_colors, markersize=LiDAR_fulltree_markers)
    # scatter!(l, LiDAR_fullbranch[:, 1], LiDAR_fullbranch[:, 2], LiDAR_fullbranch[:, 3], color=:red, markersize=2)
    # scatter!(l, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=:green, markersize=10)
    # rotate_cam!(l.scene, deg2rad(45.0), deg2rad(0.0), 0.0)
    # rotate_cam!(l.scene, deg2rad(45.0), deg2rad(130.0), 0.0)
    # rotate_cam!(l.scene, (deg2rad(0.0), deg2rad(0.0), deg2rad(180.0)))
    # rotate_cam!(l.scene, (deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)))
    # zoom!(l.scene, cameracontrols(l.scene), 3)
    # update_cam!(l.scene, cameracontrols(l.scene))
    # zoom!(l.scene, 0.3)
    f
end

symbol = "N"
xyz_attr = [:XX, :YY, :ZZ]
max_order = 5
mtg = read_mtg("0-data/4-method_visualization/tree11h_extract_pmt.mtg")
structural_model!(mtg, 0.5, 0.5, π * (0.008^2.0))
branching_order!(mtg) # Do it again, because the structural model does it in basipetal way


reference_point = Point3(11.91279984, -9.03291035, 4.90229988)
new_reference_point = Point3(-0.155291, -0.0926051, -0.0124261)

transform!(
    mtg,
    # Translate the branch extract to the previous reference point (plantscan3d changed the reference of the coordinates):
    :XX => (x -> x + reference_point[1] - new_reference_point[1]) => :XX,
    :YY => (y -> y + reference_point[2] - new_reference_point[2]) => :YY,
    :ZZ => (z -> z + reference_point[3] - new_reference_point[3]) => :ZZ,
    # Make the circle for the structural model:
    (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_pipe, symbol=symbol)) => :cyl_pipe,
    (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_sm, symbol=symbol)) => :cyl_sm,
    (node -> circle_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_pipe, symbol=symbol, radius_factor=0.001)) => :circle_pipe,
    (node -> circle_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_sm, symbol=symbol, radius_factor=0.001)) => :circle_sm,
    symbol=symbol
)

# Make the figure:
begin
    # set_theme!(backgroundcolor=:white)
    alphapoints = 0.5
    fig = Figure(size=(900, 800))
    fig.scene.backgroundcolor[] = RGBAf(1, 1, 1, 1)
    g1 = fig[1, 2:3] = GridLayout()
    g2 = fig[2:3, 1:4] = GridLayout()
    g1ax1 = Axis(g1[1, 1], aspect=1, title="S1. Destructive measurements", titlealign=:left)
    g1ax2 = Axis(g1[1, 2], aspect=1, title="S2. Modelling", xlabel="Measured CSA (mm²)", ylabel="Predicted CSA (mm²)", titlealign=:left)
    # g2ax1 = Axis3(g2[1, 1], aspect=(1, 1, 1), title="1. Point cloud", titlealign=:left, elevation=deg2rad(0.0), azimuth=deg2rad(90.0))
    g2ax1 = Axis3(g2[1, 1], aspect=(1, 1, 1), title="1. Point cloud", titlealign=:left, elevation=deg2rad(0.0), azimuth=deg2rad(200.0))
    g2ax1.tellheight = false
    # g2ax1LScene = LScene(g2[1, 1])#, title="1. Point cloud", titlealign=:left)
    # g2ax1 = g2ax1LScene.scene[OldAxis]
    # g2ax1[:title] = "1. Point cloud"
    # g2ax1[:titlealign] = :left
    g2ax2 = Axis(g2[1, 2], aspect=1, title="2. Skeleton", titlealign=:left)
    g2ax3 = Axis(g2[1, 3], aspect=1, title="3. Skeleton + Diameters", titlealign=:left)
    g2ax4 = Axis(g2[1, 4], aspect=1, title="4. Volume", titlealign=:left)

    hidedecorations!(g1ax1)
    # hidedecorations!(g1ax2)
    hidedecorations!(g2ax1)
    hidedecorations!(g2ax2)
    hidedecorations!(g2ax3)
    hidedecorations!(g2ax4)
    hidespines!(g1ax1)
    # hidespines!(g1ax2)
    hidespines!(g2ax1)
    hidespines!(g2ax2)
    hidespines!(g2ax3)
    hidespines!(g2ax4)

    # Draw the LiDAR point cloud:
    # scatter!(g2ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=LiDAR[:, 4], markersize=2)
    # scatter!(g2ax1, LiDAR_fullbranch[:, 1], LiDAR_fullbranch[:, 2], LiDAR_fullbranch[:, 3], color=LiDAR_fullbranch[:, 4], markersize=2)
    # l = LScene(f[1, 1], show_axis=true)
    scatter!(g2ax1, LiDAR_fulltree[:, 1], color=LiDAR_fulltree_colors, markersize=LiDAR_fulltree_markers)
    zoom!(g2ax1.scene, Makie.Camera3D(g2ax1.scene), 0.5)
    translate_cam!(g2ax1.scene, (0.5, 0.0, 0.0))
    update_cam!(g2ax1.scene, cameracontrols(g2ax1.scene))

    # inset_ax = Axis(g2[1, 1], aspect=1, width=Relative(0.5), height=Relative(0.5), halign=1.0, valign=1.0, backgroundcolor=(:white, 0.2))
    # hidedecorations!(inset_ax)
    # translate!(inset_ax.scene, 0, 0, 10)
    # # rotate_cam!(inset_ax.scene, Camera3D(inset_ax.scene), (deg2rad(0.0), deg2rad(0.0), deg2rad(180.0)))
    # scatter!(inset_ax, LiDAR_fullbranch[:, 1], color=:green, markersize=1)

    inset_ax2 = Axis(g2[1, 1], aspect=1, width=Relative(0.40), height=Relative(0.40), halign=1.0, valign=1.0, backgroundcolor=(:white, 0.2))
    hidedecorations!(inset_ax2)
    translate!(inset_ax2.scene, 0, 0, 10)
    # rotate_cam!(inset_ax.scene, Camera3D(inset_ax.scene), (deg2rad(0.0), deg2rad(0.0), deg2rad(180.0)))
    scatter!(inset_ax2, LiDAR[:, 1], color=:red, markersize=1)

    # Draw the skeleton (lines):
    scatter!(g2ax2, LiDAR[:, 1], color=LiDAR[:, 2], markersize=2, alpha=alphapoints)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        col = get((ColorSchemes.seaborn_rocket_gradient), node[:branching_order] / max_order)
        draw_skeleton!(g2ax2, node, xyz_attr, symbol=symbol, linewidth=4, color=col)
    end

    # Diameter based on the cross-sectional area at each node
    scatter!(g2ax3, LiDAR[:, 1], color=LiDAR[:, 2], markersize=2, alpha=alphapoints)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:circle_sm] !== nothing) do node
        draw_skeleton!(g2ax3, node, xyz_attr, symbol=symbol, linewidth=2)
        mesh!(g2ax3, node[:circle_sm], color=:red)
    end

    # Draw the 3d reconstruction:
    scatter!(g2ax4, LiDAR[:, 1], color=LiDAR[:, 2], markersize=2, alpha=alphapoints)
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        mesh!(
            g2ax4, node[:cyl_sm],
            color=get((ColorSchemes.seaborn_rocket_gradient), node[:branching_order] / max_order),
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
    # arrows_positions = Vector{Tuple{Point3{Float64},Point3{Float64}}}()
    col_arrows = :black
    for node_vec in nodes_in_segment
        parent_node = parent(node_vec[1])
        isroot(parent_node) && (parent_node = node_vec[1])

        last_node = node_vec[end]

        for i in 1:length(node_vec)
            node = node_vec[i]
            parent_node = parent(node)
            isroot(parent_node) && (parent_node = node)

            if i != length(node_vec)
                lines!(
                    g1ax1,
                    [
                        Point3(parent_node[:XX], parent_node[:YY], parent_node[:ZZ]) + Point3(0.01, -0.01, 0.0),
                        Point3(node[:XX], node[:YY], node[:ZZ]) + Point3(0.01, -0.01, 0.0)
                    ],
                    color=col_arrows,
                    # linewidth=4,
                    linewidth=2,
                )
            else
                arrows!(
                    g1ax1,
                    [Point3(parent_node[:XX], parent_node[:YY], parent_node[:ZZ]) + Point3(0.01, -0.01, 0.0)],
                    [Point3(node[:XX], node[:YY], node[:ZZ]) - Point3(parent_node[:XX], parent_node[:YY], parent_node[:ZZ])],
                    arrowsize=0.0070,
                    color=col_arrows,
                    linewidth=0.0020,
                    visible=true
                    # label="Length"
                )
            end
        end
    end

    # Draw the field "measurement":
    traverse!(mtg, symbol=symbol, filter_fun=node -> node[:cyl_sm] !== nothing) do node
        mesh!(
            g1ax1, node[:cyl_sm],
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
        i = 0.5 * (middle_node[:XX] + parent_node[:XX]), 0.5 * (middle_node[:YY] + parent_node[:YY]), 0.5 * (middle_node[:ZZ] + parent_node[:ZZ]) + 0.1

        push!(segment_positions, i)
    end

    col = RGB(250 / 255, 178 / 255, 101 / 255)
    scatter!(g1ax1, segment_positions, color=col, markersize=10, label="Diameter")

    # Legend of the "Destructive measurement" pane:
    Legend(
        g1[1, 1],
        [
            MarkerElement(color=col, marker=:circle, markersize=10, strokecolor=:black),
            MarkerElement(color=col_arrows, marker='➞', markersize=10, strokecolor=:black)
        ],
        ["Diameters", "Lengths"],
        halign=:left,
        valign=:bottom,
        orientation=:horizontal,
        tellheight=false,
        tellwidth=false,
        framevisible=false,
    )

    # General equation of the multilinear model:
    text!(
        g1ax2,
        [0], [4000],
        text=L"CSA = \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n + \epsilon",
        color=:black, fontsize=15,
        align=(:left, :center),
        # valign=:center, halign=:center
    )

    # Add a scatter plot that shows the relationship between the cross-sectional area and other variables:
    # The scatter plot is not real, it is just a representation of the relationship between the cross-sectional area and other variables.
    # Draw points that look like a linear relationship for the scatter plot:
    # Generate some fake data
    x = 1:10:4000
    # Add some random noise to y:
    y = x .+ randn(length(x)) .* 150
    y[y.<=0] .= 0
    lines!(g1ax2, x, x, color=:slategrey, linewidth=1, alpha=0.5)
    scatter!(g1ax2, x, y, color=:black, markersize=2,)

    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    fig
end
# g2ax1.scene.
# g2ax1.scene.center = false
# fig.scene.center = false
update_cam!(g2ax1.scene, cameracontrols(g2ax1.scene))
# fig

# Save the figure:
save("2-results/2-plots/Figure_3-viz_methodology.png", fig, px_per_unit=5, update=false)
