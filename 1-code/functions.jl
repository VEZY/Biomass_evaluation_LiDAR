module BiomassFromLiDAR

using MultiScaleTreeGraph
using Statistics: mean
using CSV
using DataFrames

export compute_all_mtg_data
export bind_csv_files
export segmentize_mtgs
export compute_volume
export NRMSE, RMSE, EF, nRMSE
export structural_model!

function structural_model!(mtg_tree, fresh_density, dry_density, first_cross_section=nothing)
    if (:radius in names(mtg_tree))
        transform!(mtg_tree, :radius => (x -> x * 2) => :diameter, symbol="N") # diameter in m
    end

    # Step 1: computes the length of each node:
    transform!(mtg_tree, compute_length_coord => :length, symbol="N") # length is in meters

    @mutate_mtg!(
        mtg_tree,
        pathlength_subtree = sum(filter(x -> x !== nothing, descendants!(node, :length, symbol="N", self=true))),
        symbol = "N",
        filter_fun = x -> x[:length] !== nothing
    )

    # Identify which node is a segment root:
    transform!(mtg_tree, is_seg => :is_segment, symbol="N")
    transform!(mtg_tree, segment_index_on_axis => :segment_index_on_axis, symbol="N")

    @mutate_mtg!(
        mtg_tree,
        segment_subtree = length(descendants!(node, :length, symbol="N", self=true, filter_fun=is_seg)),
        number_leaves = nleaves!(node),
        symbol = "N"
    )

    branching_order!(mtg_tree, ascend=false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Use the first cross-section for the first value to apply the pipe-model:
    if first_cross_section === nothing
        first_cross_section = π * ((descendants(mtg_tree, :diameter, ignore_nothing=true, recursivity_level=5)[1] / 2.0)^2)
    end

    @mutate_mtg!(mtg_tree, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(
        mtg_tree,
        (
            cross_section=first_cross_section,
            cross_section_pipe=first_cross_section,
            cross_section_sm=first_cross_section
        )
    )

    # Compute the cross-section using the structural model:
    @mutate_mtg!(mtg_tree, cross_section_sm = cross_section_stat_mod_all(node, symbol="N"), symbol = "N")

    # Compute the diameters:
    transform!(mtg_tree, :cross_section_pipe => (x -> sqrt(x / π) * 2.0) => :diameter_pipe, symbol="N")
    transform!(mtg_tree, :cross_section_sm => (x -> sqrt(x / π) * 2.0 / 1000.0) => :diameter_sm, symbol="N")

    # Compute the radius
    transform!(mtg_tree, :diameter_pipe => (x -> x / 2) => :radius_pipe, symbol="N")
    transform!(mtg_tree, :diameter_sm => (x -> x / 2) => :radius_sm, symbol="N")

    # Recompute the volume:
    compute_volume_stats(x, var) = x[var] * x[:length]

    @mutate_mtg!(mtg_tree, volume_sm = compute_volume_stats(node, :cross_section_sm), symbol = "N") # volume in mm3
    @mutate_mtg!(mtg_tree, volume_pipe_mod = compute_volume_stats(node, :cross_section_pipe), symbol = "N") # volume in mm3

    # And the biomass:
    @mutate_mtg!(mtg_tree, fresh_mass = node[:volume_sm] * fresh_density * 1e-3, symbol = "N") # in g
    @mutate_mtg!(mtg_tree, dry_mass = node[:volume_sm] * dry_density * 1e-3, symbol = "N") # in g

    @mutate_mtg!(mtg_tree, fresh_mass_pipe_mod = node[:volume_pipe_mod] * fresh_density * 1e-3, symbol = "N") # in g
    @mutate_mtg!(mtg_tree, dry_mass_pipe_mod = node[:volume_pipe_mod] * dry_density * 1e-3, symbol = "N") # in g

    # Clean-up the cached variables:
    clean_cache!(mtg_tree)
end

function compute_cross_section_all(x, var=:cross_section)
    if x.MTG.symbol == "A"
        desc_cross_section = descendants(x, var, symbol="S", recursivity_level=1)
        if length(desc_cross_section) > 0
            return desc_cross_section[1]
        else
            @warn "$(x.name) has no descendants with a value for $var."
        end
    else
        x[var]
    end
end

function compute_volume_subtree(x)
    volume_descendants = filter(x -> x !== nothing, descendants!(x, :volume, symbol="S", self=true))
    length(volume_descendants) > 0 ? sum(volume_descendants) : nothing
end

function compute_cross_section(x)
    if x[:diameter] !== nothing
        π * ((x[:diameter] / 2.0)^2)
    end
end

function compute_cross_section_children(x)
    cross_section_child = filter(x -> x !== nothing, descendants!(x, :cross_section, symbol="S", recursivity_level=1))

    return length(cross_section_child) > 0 ? sum(cross_section_child) : nothing
end

function compute_cross_section_leaves(x)
    cross_section_leaves = filter(x -> x !== nothing, descendants!(x, :cross_section; filter_fun=isleaf))

    return length(cross_section_leaves) > 0 ? sum(cross_section_leaves) : nothing
end

function compute_volume(x)
    if x[:diameter] !== nothing && x[:length] !== nothing
        π * ((x[:diameter] / 2.0)^2) * x[:length]
    end
end

function compute_diameter(x)
    if x[:diameter] === nothing
        diams = [x[:diameter_50_1], x[:diameter_50_2], x[:diameter_70_1], x[:diameter_70_2]]
        filter!(x -> x !== nothing, diams)
        if length(diams) > 0
            return mean(diams)
        else
            return nothing
        end
    else
        return x[:diameter]
    end
end

function get_axis_length(x)
    axis_length = ancestors(x, :axis_length, symbol="A", recursivity_level=1)
    if length(axis_length) > 0
        axis_length[1]
    else
        nothing
    end
end


function compute_length(x)
    if x[:length] === nothing
        x[:length_mm]
    else
        return x[:length] * 10.0
    end
end

function compute_axis_length(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", link=("/", "<"), all=false))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end

function compute_dry_w(x)
    if x[:dry_weight_p1] !== nothing
        x[:dry_weight_p1]
    elseif x[:dry_weight_p2] !== nothing
        x[:dry_weight_p2]
    end
end

function compute_density(x)
    if x[:fresh_density] !== nothing
        x[:fresh_density]
    elseif x[:dry_weight] !== nothing
        x[:dry_weight] / x[:volume_bh]
    end
end

function compute_subtree_length!(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", self=true))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end

function compute_data_mtg!(mtg)

    @mutate_mtg!(mtg, length = compute_length(node), symbol = "S")
    @mutate_mtg!(mtg, dry_weight = compute_dry_w(node), symbol = "S")
    @mutate_mtg!(mtg, density = compute_density(node), symbol = "S")

    @mutate_mtg!(
        mtg,
        pathlength_subtree = sum(filter(x -> x !== nothing, descendants!(node, :length, symbol="S", self=true))),
        symbol = "S",
        filter_fun = x -> x[:length] !== nothing
    )

    @mutate_mtg!(
        mtg,
        segment_subtree = length(descendants!(node, :length, symbol="S", self=true)),
        number_leaves = nleaves!(node),
        symbol = "S"
    )

    branching_order!(mtg, ascend=false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Compute the index of each segment on the axis in a basipetal way (from tip to base)
    @mutate_mtg!(
        mtg,
        n_segments = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)),
        symbol = "A"
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    @mutate_mtg!(
        mtg,
        n_segments_axis = ancestors(node, :n_segments, symbol="A")[1],
        segment_index_on_axis = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)) + 1,
        symbol = "S"
    )

    # Compute the total length of the axis in mm:
    @mutate_mtg!(
        mtg,
        axis_length = compute_axis_length(node),
        symbol = "A"
    )

    # Associate the axis length to each segment:
    @mutate_mtg!(mtg, axis_length = get_axis_length(node), symbol = "S")

    # New branches (>10, e.g. tree12l, the ones from A. Bonnet) diameters are measured twice
    #  on the same point at mid-segment, or even at two points (30% and 70% of the segment
    # length) when the segment length is > 30 cm
    @mutate_mtg!(mtg, diameter = compute_diameter(node), symbol = "S") # diameter of the segment in mm

    @mutate_mtg!(mtg, volume = compute_volume(node), symbol = "S") # volume of the segment in mm3

    @mutate_mtg!(mtg, cross_section = compute_cross_section(node), symbol = "S") # area of segment cross section in mm2
    @mutate_mtg!(mtg, cross_section_children = compute_cross_section_children(node), symbol = "S") # area of segment cross section in mm2

    # Cross section of the terminal nodes for each node
    @mutate_mtg!(mtg, cross_section_leaves = compute_cross_section_leaves(node), symbol = "S")


    # Volume of wood the section bears (all the sub-tree):
    @mutate_mtg!(mtg, volume_subtree = compute_volume_subtree(node), symbol = "S")

    # How many leaves the sibling of the node has:
    @mutate_mtg!(mtg, nleaves_siblings = sum(nleaves_siblings!(node)))

    # How many leaves the node has in proportion to its siblings + itself:
    @mutate_mtg!(mtg, nleaf_proportion_siblings = node[:number_leaves] / (node[:nleaves_siblings] + node[:number_leaves]), symbol = "S")

    first_cross_section = filter(x -> x !== nothing, descendants(mtg, :cross_section, recursivity_level=5))[1]
    @mutate_mtg!(mtg, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(mtg, (cross_section=first_cross_section,))
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    @mutate_mtg!(mtg, cross_section_all = compute_cross_section_all(node))

    # Use the pipe model, but only on nodes with a cross_section <= 314 (≈20mm diameter)
    @mutate_mtg!(mtg, cross_section_pipe_50 = pipe_model!(node, :cross_section_all, 314, allow_missing=true))

    # Clean-up the cached variables:
    clean_cache!(mtg)

    return mtg
end

function compute_all_mtg_data(mtg_file, new_mtg_file, csv_file)
    # Import the mtg file:
    mtg = read_mtg(mtg_file)

    # Compute extra data:
    compute_data_mtg!(mtg)

    # write the resulting mtg to disk:
    write_mtg(new_mtg_file, mtg)

    # And the resulting DataFrame to a csv file:
    df =
        DataFrame(
            mtg,
            [
                :density, :length, :diameter, :axis_length, :branching_order,
                :segment_index_on_axis, :mass_g, :volume, :volume_subtree, :cross_section,
                :cross_section_children, :cross_section_leaves, :n_segments_axis,
                :number_leaves, :pathlength_subtree, :segment_subtree,
                :cross_section_pipe, :cross_section_pipe_50, :nleaf_proportion_siblings,
                :nleaves_siblings, :cross_section_all, :comment, :id_cor
            ])

    CSV.write(csv_file, df[:, Not(:tree)])
end




###############################################
# Functions used in 2-model_diameter.jl
###############################################

function bind_csv_files(csv_files)
    dfs = []
    for i in csv_files
        df_i = CSV.read(i, DataFrame)
        df_i[:, :branch] .= splitext(basename(i))[1]

        transform!(
            df_i,
            :branch => ByRow(x -> x[5:end-1]) => :tree
        )

        rename!(
            df_i,
            :branch => :unique_branch
        )

        transform!(
            df_i,
            :unique_branch => ByRow(x -> x[end]) => :branch
        )
        push!(dfs, df_i)
    end

    df = dfs[1]
    for i in 2:length(dfs)
        df = vcat(df, dfs[i])
    end

    return df
end



###############################################
# Functions used in 3-mtg_plantscan3d_to_segments
###############################################

function segmentize_mtgs(in_folder, out_folder)
    # Listing the mtg files in the folder:
    mtg_files = filter(x -> splitext(basename(x))[2] in [".mtg"], readdir(in_folder))

    # Modifying the format of the MTG to match the one from the field, i.e. with segments and axis instead of nodes
    for i in mtg_files
        segmentize_mtg(
            joinpath(in_folder, i),
            joinpath(out_folder, i),
        )
    end
end

"""
    segmentize_mtg(in_file, out_file)

Transform the input mtg from plantscan3d into an mtg with segments and axis. Segments are
nodes describing the portion of the branch between two branching points. Axis is the
upper-scale grouping following segments, *i.e.* segments with a "/" or "<" link.
"""
function segmentize_mtg(in_file, out_file)
    # in_folder = joinpath("0-data", "3-mtg_lidar_plantscan3d", "1-raw_output")
    # out_folder = joinpath("0-data", "3-mtg_lidar_plantscan3d", "3-raw_output_segmentized")
    # out_file = joinpath(out_folder, mtg_files[1])
    # in_file = joinpath(in_folder, mtg_files[1])
    mtg = read_mtg(in_file)

    # Compute internode length and then cumulate the lenghts when deleting.

    # Step 1: computes the length of each node:
    transform!(mtg, compute_length_coord => :length_node, scale=2) # length is in meters

    # Step 2: cumulate the length of all nodes in a segment for each segment node:
    transform!(mtg, cumul_length_segment => :length, scale=2, filter_fun=is_seg)
    # And add a lenght of 0 for the first segment:
    mtg[1][:length] = 0.0

    # Step 3: delete nodes to make the mtg as the field measurements: with nodes only at in_filtering points
    mtg = delete_nodes!(mtg, filter_fun=is_segment!, scale=(1, 2))

    # Insert a new scale: the Axis.
    # First step we put all nodes at scale 3 instead of 2:
    transform!(mtg, (node -> node.MTG.scale = 3), scale=2)
    mtg.attributes[:scales] = scales(mtg)
    # 2nd step, we add axis nodes (scale 2) branching each time there's a branching node:
    template = MutableNodeMTG("+", "A", 0, 2)
    insert_parents!(mtg, template, scale=3, link="+")
    # And before the first node decomposing the plant:
    insert_parents!(mtg, NodeMTG("/", "A", 1, 2), scale=3, link="/", all=false)

    # 3d step, we change the branching nodes links to decomposition:
    transform!(mtg, (node -> node.MTG.link = "/"), scale=3, link="+")

    # Fourth step, we rename the nodes symbol into segments "S":
    transform!(mtg, (node -> node.MTG.symbol = "S"), symbol="N")

    # And the plant symbol as the plant name:
    symbol_from_file = splitext(replace(basename(out_file), "_" => ""))[1]

    # If the file name ends with a number we need to add something to not mistake it with an index
    if match(r"[0-9]+$", symbol_from_file) !== nothing
        symbol_from_file *= "whole"
    end

    mtg.MTG.symbol = symbol_from_file

    # Updating the symbols in the root node:
    mtg.attributes[:symbols] = symbols(mtg)

    # Last step, we add the index as in the field, *i.e.* the axis nodes are indexed following
    # their topological order, and the segments are indexed following their position on the axis:
    transform!(mtg, (x -> 1) => :index, symbol="A")
    transform!(mtg, (node -> A_indexing(node)) => :index, symbol="A")
    transform!(mtg, (node -> node.MTG.index = node[:index]), symbol="A")
    # NB: this is done in 3 steps because the index is not a node attribute at first (it
    # is in the MTG field). So we first initialize the index to 1, then we update it with the
    # indexing function (A_indexing), then we update the index from the MTG field with the
    # index attribute.

    # Remove the index from the nodes attributes (only temporary):
    traverse!(mtg, node -> pop!(node, :index))

    # Set back the root node with no indexing:
    mtg.MTG.index = -9999

    transform!(mtg, (node -> node.MTG.index = S_indexing(node)), scale=3)

    # Delete the old length of the nodes (length_node) from the attributes:
    transform!(mtg, (x -> x[:length_node] === nothing ? nothing : pop!(x.attributes, :length_node)))

    # Write MTG back to file:
    write_mtg(out_file, mtg)
end


"""
    compute_length_coord(node)

Compute node length as the distance between itself and its parent.
"""
function compute_length_coord(node; x=:XX, y=:YY, z=:ZZ)
    if !isroot(node.parent)
        sqrt(
            (node.parent[x] - node[x])^2 +
            (node.parent[y] - node[y])^2 +
            (node.parent[z] - node[z])^2
        )
    else
        0.0
    end
end

"""
    is_seg(x)

Tests if a node is a segment node. A segment node is a node:
    - at a branching position, *i.e.*, a parent of more than one children node
    - a leaf.
"""
is_seg(x) = isleaf(x) || (!isroot(x) && (length(children(x)) > 1 || x[1].MTG.link == "+"))

"""
    segment_index_on_axis(node, symbol="N")

Compute the index of a segment node on the axis. The computation is basipetal, starting from tip to base. 
The index is the position of the segment on the axis.
"""
function segment_index_on_axis(node, symbol="N")
    isleaf(node) ? 1 : sum(descendants!(node, :is_segment, symbol=symbol, link=("/", "<"), all=false, type=Bool)) + 1
end

"""
    cumul_length_segment(node)

Cumulates the lengths of segments inside a segment. Only does it if the node is considered
as a segment, else returns 0.
"""
function cumul_length_segment(node, length_name=:length_node)
    if is_seg(node)

        length_ancestors =
            [
                node[length_name],
                ancestors(
                    node,
                    length_name,
                    filter_fun=x -> !is_seg(x),
                    scale=2,
                    all=false)...
            ]
        # NB: we don't use self = true because it would trigger a stop due to all = false
        filter!(x -> x !== nothing, length_ancestors)

        sum(length_ancestors) * 1000.0
    else
        0.0
    end
end

function A_indexing(node)
    parent_index = ancestors(
        node, :index,
        symbol="A",
        recursivity_level=1,
        type=Union{Int64,Nothing}
    )

    if length(parent_index) == 0
        return 1
    elseif node.MTG.link == "+"
        return parent_index[1] + 1
    else
        return parent_index[1]
    end
end


function S_indexing(node)
    if isroot(node)
        return 0
    else
        node.MTG.link == "/" ? 1 : node.parent.MTG.index + 1
    end
end

function cross_section_stat_mod_all(node; symbol="N")
    max(
        0.0,
        0.520508 * node[:cross_section_pipe] + 0.0153365 * node[:pathlength_subtree] +
        6.38394 * node[:branching_order] + 10.9389 * node[:segment_index_on_axis] - 10.137 * node[:number_leaves] +
        4.46843 * node[:segment_subtree]
    )
end
end


function cylinder_from_radius(node::MultiScaleTreeGraph.Node, xyz_attr=[:XX, :YY, :ZZ]; radius=:radius, symbol="S")
    node_start = ancestors(node, xyz_attr, recursivity_level=2, symbol=symbol)
    if length(node_start) != 0
        Cylinder(
            Point3((node_start[1][1], node_start[1][2], node_start[1][3] .+ 0.01)),
            Point3((node[xyz_attr[1]], node[xyz_attr[2]], node[xyz_attr[3]] .+ 0.01)),
            node[radius] # radius in meter
        )
    end
end


function circle_from_radius(node::MultiScaleTreeGraph.Node, xyz_attr=[:XX, :YY, :ZZ]; radius=:radius, symbol="S", radius_factor=1.0)
    node_start = ancestors(node, xyz_attr, recursivity_level=2, symbol=symbol)
    if length(node_start) != 0
        template_cyl = cylinder_from_radius(node, xyz_attr, radius=radius, symbol=symbol)
        top = extremity(template_cyl)
        new_end_point = top + direction(template_cyl) * radius_factor # Scale the direction vector to the desired length

        # Create a new cylinder with the same end point, but adding a new end point that gives the width of the circle:
        new_cyl = Cylinder(top, new_end_point, Makie.radius(template_cyl))
    end
end

function draw_skeleton!(axis, node::MultiScaleTreeGraph.Node, xyz_attr=[:XX, :YY, :ZZ]; symbol="S", color=:slategrey, linewidth)
    node_start = ancestors(node, xyz_attr, recursivity_level=2, symbol=symbol)
    if length(node_start) != 0
        lines!(
            axis,
            [node_start[1][1], node[:XX]],
            [node_start[1][2], node[:YY]],
            [node_start[1][3], node[:ZZ]] .+ 0.01,
            color=color,
            linewidth=linewidth
        )
    end
end