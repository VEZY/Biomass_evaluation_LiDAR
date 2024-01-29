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
MTG_files = [
    "tree11h_extract_point_cloud.mtg",
    "tree11h_extract_pmt.mtg",
    "tree11h_extract_pmt.mtg" # This one is repeated but it will be overwritten for our model
]

symbol = "N"
xyz_attr = [:XX, :YY, :ZZ]
mtgs = []
max_order = Int[]

for i in MTG_files
    mtg = read_mtg(joinpath("0-data/4-method_visualization", i))
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
    branch_name = replace(i, ".mtg" => "")
end


structural_model!(mtgs[3], 0.5, 0.5, π * (0.008^2.0))
branching_order!(mtgs[3]) # Do it again, because the structural model does it in basipetal way

# Make the circle for the structural model:
transform!(
    mtgs[3],
    (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_pipe, symbol=symbol)) => :cyl_pipe,
    (node -> cylinder_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_sm, symbol=symbol)) => :cyl_sm,
    (node -> circle_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_pipe, symbol=symbol, radius_factor=0.0001)) => :circle_pipe,
    (node -> circle_from_radius(node, [:XX, :YY, :ZZ], radius=:radius_sm, symbol=symbol, radius_factor=0.0001)) => :circle_sm,
    symbol=symbol
)

# max_order = maximum(max_order)
max_order = 5

# Make the figure:
begin
    fig = Figure(size=(800, 800))
    g1 = fig[1, 1] = GridLayout()
    # g2 = fig[2, 1] = GridLayout()
    g3 = fig[2, 1] = GridLayout()
    g4 = fig[3, 1] = GridLayout()

    ax1 = Axis(g1[1, 1], aspect=1, title="Point cloud")
    ax2 = Axis(g1[1, 2], aspect=1, title="Skeleton")
    ax3a = Axis(g3[1, 1], aspect=1, title="Point cloud")
    ax3b = Axis(g3[1, 2], aspect=1, title="PMT", titlesize=12)
    ax3c = Axis(g3[1, 3], aspect=1, title="SM (Ours)", titlesize=12) # Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_pX_p + \epsilon

    # ax4 = Axis(g4[1, 1], aspect=1, title="Volume")
    ax4a = Axis(g4[1, 1], aspect=1, title="Point cloud", titlesize=12)
    ax4b = Axis(g4[1, 2], aspect=1, title="PMT", titlesize=12)
    ax4c = Axis(g4[1, 3], aspect=1, title="SM (Ours)", titlesize=12)


    hidedecorations!(ax1; grid=false, minorgrid=false)
    hidedecorations!(ax2; grid=false, minorgrid=false)
    hidedecorations!(ax3a; grid=false, minorgrid=false)
    hidedecorations!(ax3b; grid=false, minorgrid=false)
    hidedecorations!(ax3c; grid=false, minorgrid=false)
    hidedecorations!(ax4a; grid=false, minorgrid=false)
    hidedecorations!(ax4b; grid=false, minorgrid=false)
    hidedecorations!(ax4c; grid=false, minorgrid=false)

    # Draw the LiDAR point cloud:
    # scatter!(ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=LiDAR[:, 4], markersize=2)
    scatter!(ax1, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2)

    # Draw the skeleton (lines):
    scatter!(ax2, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=0.1)
    traverse!(mtgs[1], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        draw_skeleton!(ax2, node, xyz_attr, symbol=symbol, linewidth=2)
    end

    # Diameter based on the point cloud:
    scatter!(ax3a, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=0.1)
    traverse!(mtgs[1], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        draw_skeleton!(ax3a, node, xyz_attr, symbol=symbol, linewidth=0.8)
        mesh!(
            ax3a, node[:circle],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    # Diameter based on the pipe model theory:
    scatter!(ax3b, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=0.1)
    traverse!(mtgs[3], symbol=symbol, filter_fun=node -> node[:circle_pipe] !== nothing) do node
        draw_skeleton!(ax3b, node, xyz_attr, symbol=symbol, linewidth=0.8)
        mesh!(
            ax3b, node[:circle_pipe],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    # Diameter based on the cross-sectional area at each node
    scatter!(ax3c, LiDAR[:, 1], LiDAR[:, 2], LiDAR[:, 3], color=point_color, markersize=2, alpha=0.1)
    traverse!(mtgs[3], symbol=symbol, filter_fun=node -> node[:circle_sm] !== nothing) do node
        draw_skeleton!(ax3c, node, xyz_attr, symbol=symbol, linewidth=0.8)
        mesh!(
            ax3c, node[:circle_sm],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    Label(g3[1, 2, Top()], "Diameter",
        fontsize=16,
        font=:bold,
        padding=(0, 0, 20, 50),
    )

    Label(g4[1, 2, Top()], "Volume",
        fontsize=16,
        font=:bold,
        padding=(0, 0, 20, 50),
    )

    # Draw the 3d reconstruction:
    traverse!(mtgs[1], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        mesh!(
            ax4a, node[:cyl],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    traverse!(mtgs[3], symbol=symbol, filter_fun=node -> node[:cyl] !== nothing) do node
        # mesh!(
        #     ax4a, node[:cyl],
        #     color=get(
        #         (ColorSchemes.seaborn_rocket_gradient),
        #         node[:branching_order] / max_order
        #     )
        # )

        mesh!(
            ax4b, node[:cyl_pipe],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )

        mesh!(
            ax4c, node[:cyl_sm],
            color=get(
                (ColorSchemes.seaborn_rocket_gradient),
                node[:branching_order] / max_order
            )
        )
    end

    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    # resize_to_layout!(fig)
    fig
end