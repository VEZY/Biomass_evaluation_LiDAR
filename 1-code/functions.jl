module BiomassFromLiDAR

using MTG
using Statistics:mean
using CSV
using DataFrames

export compute_all_mtg_data
export bind_csv_files
export segmentize_mtgs
export compute_volume, compute_var_axis
export NRMSE, RMSE, EF
export compute_volume_model, volume_stats

###############################################
# Functions used in 1-compute_field_mtg_data.jl
###############################################

function compute_data_mtg(mtg)

    @mutate_mtg!(mtg, length = compute_length(node), symbol = "S")
    @mutate_mtg!(mtg, dry_weight = compute_dry_w(node), symbol = "S")
    @mutate_mtg!(mtg, density = compute_density(node), symbol = "S")

    @mutate_mtg!(
        mtg,
        pathlength_subtree = sum(filter(x -> x !== nothing, descendants!(node, :length, symbol = "S", self = true))),
        symbol = "S",
        filter_fun = x -> x[:length] !== nothing
    )

    @mutate_mtg!(
        mtg,
        segment_subtree = length(descendants!(node, :length, symbol = "S", self = true)),
        number_leaves = nleaves!(node),
        symbol = "S"
    )

    branching_order!(mtg, ascend = false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Compute the index of each segment on the axis in a basipetal way (from tip to base)
    @mutate_mtg!(
        mtg,
        n_segments = length(descendants!(node, :length, symbol = "S", link = ("/", "<"), all = false)),
        symbol = "A"
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    @mutate_mtg!(
        mtg,
        n_segments_axis = ancestors(node, :n_segments, symbol = "A")[1],
        segment_index_on_axis = length(descendants!(node, :length, symbol = "S", link = ("/", "<"), all = false)) + 1,
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

    # @mutate_mtg!(mtg, volume = compute_var_axis(node), symbol = "A") # volume of the axis in mm3

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

    first_cross_section = filter(x -> x !== nothing, descendants(mtg, :cross_section, recursivity_level = 5))[1]
    @mutate_mtg!(mtg, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(mtg, (cross_section = first_cross_section,))
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    @mutate_mtg!(mtg, cross_section_all = compute_cross_section_all(node))

    # Use the pipe model, but only on nodes with a cross_section <= 314 (≈20mm diameter)
    @mutate_mtg!(mtg, cross_section_pipe_20 = pipe_model!(node, :cross_section_all, 314, allow_missing = true))

    # Clean-up the cached variables:
    clean_cache!(mtg)

    return mtg
end

function compute_cross_section_all(x)
    if x.MTG.symbol == "A"
        desc_cross_section = descendants(x, :cross_section, symbol = "S", recursivity_level = 1)
        if length(desc_cross_section) > 0
            return desc_cross_section[1]
        else
            @warn "$(x.name) has no descendants with a value for cross_section."
        end
    else
        x[:cross_section]
    end
end


function compute_volume_subtree(x)
    volume_descendants = filter(x -> x !== nothing, descendants!(x, :volume, symbol = "S", self = true))
    length(volume_descendants) > 0 ? sum(volume_descendants) : nothing
end

function compute_cross_section(x)
    if x[:diameter] !== nothing
        π * ((x[:diameter] / 2.0)^2)
    end
end

function compute_cross_section_children(x)
    cross_section_child = filter(x -> x !== nothing, descendants!(x, :cross_section, symbol = "S", recursivity_level = 1))

    return length(cross_section_child) > 0 ? sum(cross_section_child) : nothing
end

function compute_cross_section_leaves(x)
    cross_section_leaves = filter(x -> x !== nothing, descendants!(x, :cross_section; filter_fun = isleaf))

    return length(cross_section_leaves) > 0 ? sum(cross_section_leaves) : nothing
    end

function compute_volume(x)
    if x[:diameter] !== nothing && x[:length] !== nothing
        π * ((x[:diameter] / 2.0)^2) * x[:length]
    end
end

"""
    compute_var_axis(x, vol_col = :volume)

Sum a variable over an axis alone, excluding the axis it bears itself.
"""
function compute_var_axis(x, vol_col = :volume)
    sum(descendants!(x, vol_col, symbol = "S", link = ("/", "<"), all = false))
end

"""
    compute_A1_axis_from_start(x, vol_col = :volume; id_cor_start)

Compute the sum of a variable over the axis starting from node that has `id_cor_start` value.
"""
function compute_A1_axis_from_start(x, vol_col = :volume; id_cor_start)
    length_gf_A1 = descendants!(x, vol_col, symbol = "S", link = ("/", "<"), all = false)
    id_cor_A1 = descendants!(x, :id_cor, symbol = "S", link = ("/", "<"), all = false)
    sum(length_gf_A1[findfirst(x -> x == id_cor_start, id_cor_A1):end])
end



function compute_var_axis_A2(x, vol_col = :volume)
    sum(descendants!(x, vol_col, symbol = "S"))
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
    axis_length = ancestors(x, :axis_length, symbol = "A", recursivity_level = 1)
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
        return x[:length] * 10.
    end
end

function compute_axis_length(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol = "S", link = ("/", "<"), all = false))
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
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol = "S", self = true))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end


function compute_all_mtg_data(mtg_file, new_mtg_file, csv_file)
    # Import the mtg file:
    mtg = read_mtg(mtg_file)

    # Compute extra data:
    compute_data_mtg(mtg)

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
            :cross_section_pipe, :cross_section_pipe_20, :nleaf_proportion_siblings,
            :nleaves_siblings, :cross_section_all, :comment, :id_cor
        ])

    CSV.write(csv_file, df[:,Not(:tree)])
end




###############################################
# Functions used in 2-model_diameter.jl
###############################################

function bind_csv_files(csv_files)
    dfs = []
    for i in csv_files
        df_i = CSV.read(i, DataFrame)
        df_i[:,:branch] .= splitext(basename(i))[1]

        transform!(
            df_i,
            :branch => ByRow(x -> x[5:end - 1]) => :tree
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
    mtg = read_mtg(in_file)

    # Compute internode length and then cumulate the lenghts when deleting.

    # Step 1: computes the length of each node:
    @mutate_mtg!(mtg, length_node = compute_length_coord(node), scale = 2) # length is in meters

    # Step 3: cumulate the length of all nodes in a segment for each segment node:
    @mutate_mtg!(mtg, length = cumul_length_segment(node), scale = 2,  filter_fun = is_seg)
    # And add a lenght of 0 for the first segment:
    mtg[1][:length] = 0.0

    # Step 4: delete nodes to make the mtg as the field measurements: with nodes only at in_filtering points
    mtg = delete_nodes!(mtg, filter_fun = is_segment!, scale = (1, 2))

    # Insert a new scale: the Axis.
    # First step we put all nodes at scale 3 instead of 2:
    @mutate_mtg!(mtg, node.MTG.scale = 3, scale = 2)

    # 2nd step, we add axis nodes (scale 2) branching each time there's a branching node:
    template = MutableNodeMTG("+", "A", 0, 2)
    insert_nodes!(mtg, template, scale = 3, link = "+")
    # And before the first node decomposing the plant:
    insert_nodes!(mtg, MutableNodeMTG("/", "A", 0, 2), scale = 3, link = "/", all = false)

    # 3d step, we change the branching nodes links to decomposition:
    @mutate_mtg!(mtg, node.MTG.link = "/", scale = 3, link = "+")

    # Fourth step, we rename the nodes symbol into segments "S":
    @mutate_mtg!(mtg, node.MTG.symbol = "S", symbol = "N")
    # And the plant symbol as the plant name:
    # mtg.MTG.symbol = replace(basename(in_file), ".mtg" => "")
    mtg.MTG.symbol = splitext(basename(out_file))[1]
    # Last step, we add the index as in the field, *i.e.* the axis nodes are indexed following
    # their topological order, and the segments are indexed following their position on the axis:

    @mutate_mtg!(mtg,node.MTG.index = A_indexing(node))

    # Set back the root node with no indexing:
    mtg.MTG.index = nothing

    @mutate_mtg!(mtg, node.MTG.index = S_indexing(node), scale = 3)

    # Delete the old length of the nodes (length_node) from the attributes:
    traverse!(mtg, x -> x[:length_node] === nothing ? nothing : pop!(x.attributes, :length_node))

    # Write MTG back to file:
    write_mtg(out_file, mtg)
end


"""
    compute_length_coord(node)

Compute node length as the distance between itself and its parent.
"""
function compute_length_coord(node)
    if !isroot(node.parent)
        sqrt(
            (node.parent[:XX] - node[:XX])^2 +
            (node.parent[:YY] - node[:YY])^2 +
(node.parent[:ZZ] - node[:ZZ])^2
        )
    else
        0.0
    end
end

"""
    is_seg(x)

Is a node also a segment node ? A segment node is a node at a branching position, or at the
first (or last) position in the tree.
"""
is_seg(x) = isleaf(x) || (!isroot(x) && (length(x.children) > 1 || x[1].MTG.link == "+"))

"""
    cumul_length_segment(node)

Cumulates the lengths of segments inside a segment. Only does it if the node is considered
as a segment, else returns 0.
"""
function cumul_length_segment(node)
    if is_seg(node)

        length_ancestors =
        [
            node[:length_node],
            ancestors(
                node,
                :length_node,
                filter_fun = x -> !is_seg(x),
                scale = 2,
                all = false)...
        ]
        # NB: we don't use self = true because it would trigger a stop due to all = false
        filter!(x -> x !== nothing, length_ancestors)


        sum(length_ancestors) * 1000.
    else
        0.0
    end
end

    function A_indexing(node)
if isroot(node)
        return 1
    else
        node.MTG.link == "+" ? node.parent.MTG.index + 1 : node.parent.MTG.index
    end
end


function S_indexing(node)
    if isroot(node)
        return 0
    else
        node.MTG.link == "/" ? 1 : node.parent.MTG.index + 1
    end
end

"""
    NRMSE(obs,sim)

Returns the Normalized Root Mean Squared Error between observations `obs` and simulations `sim`.
Normalization is performed using division by observations range (max-min).
Output: Float/Particles
"""
function NRMSE(obs, sim)
    return sqrt(sum((obs .- sim).^2) / length(obs)) / (findmax(obs)[1] - findmin(obs)[1])
end

"""
    RMSE(obs,sim)

Returns the Root Mean Squared Error between observations `obs` and simulations `sim`.
The closer to 0 the better.
"""
function RMSE(obs, sim, digits = 2)
    return round(sqrt(sum((obs .- sim).^2) / length(obs)), digits = digits)
end


"""
    EF(obs,sim)

Returns the Efficiency Factor between observations `obs` and simulations `sim` using NSE (Nash-Sutcliffe efficiency) model.
More information can be found at https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient.
The closer to 1 the better.
"""
function EF(obs, sim, digits = 2)
    SSres = sum((obs - sim).^2)
    SStot = sum((obs .- mean(obs)).^2)
    return round(1 - SSres / SStot, digits = digits)
end


function compute_data_mtg_lidar!(mtg, fresh_density, dry_density)

    @mutate_mtg!(mtg, diameter = node[:radius] * 2 * 1000, symbol = "S") # diameter in mm


    @mutate_mtg!(
        mtg,
        pathlength_subtree = sum(filter(x -> x !== nothing, descendants!(node, :length, symbol = "S", self = true))),
        symbol = "S",
        filter_fun = x -> x[:length] !== nothing
    )

    @mutate_mtg!(
        mtg,
        segment_subtree = length(descendants!(node, :length, symbol = "S", self = true)),
        number_leaves = nleaves!(node),
        symbol = "S"
    )

    branching_order!(mtg, ascend = false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Compute the index of each segment on the axis in a basipetal way (from tip to base)
    @mutate_mtg!(
        mtg,
        n_segments = length(descendants!(node, :length, symbol = "S", link = ("/", "<"), all = false)),
        symbol = "A"
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    @mutate_mtg!(
        mtg,
        n_segments_axis = ancestors(node, :n_segments, symbol = "A")[1],
        segment_index_on_axis = length(descendants!(node, :length, symbol = "S", link = ("/", "<"), all = false)) + 1,
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

    first_cross_section = filter(x -> x !== nothing, descendants(mtg, :cross_section, recursivity_level = 5))[1]
    @mutate_mtg!(mtg, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(mtg, (cross_section = first_cross_section,))
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    @mutate_mtg!(mtg, cross_section_all = compute_cross_section_all(node))

    # Use the pipe model, but only on nodes with a cross_section <= 314 (≈20mm diameter)
    @mutate_mtg!(mtg, cross_section_pipe_20 = pipe_model!(node, :cross_section_all, 314, allow_missing = true))

    @mutate_mtg!(mtg, cross_section_stat_mod = cross_section_stat_mod(node), symbol = "S")

    # Compute the A2 lengths to match measurements =total length of all segments they bear:
    @mutate_mtg!(mtg, length_sim = compute_var_axis_A2(node, :length), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    # A1 length in mm (just itself, excluding A2 length):
    mtg[1][:length_sim] = compute_A1_axis_from_start(mtg[1], :length, id_cor_start = 0)

    # Recompute the volume:
    compute_volume_stats(x, var) = x[var] * x[:length]

    @mutate_mtg!(mtg, volume_ps3d = compute_volume_stats(node, :cross_section), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg, volume_stat_mod = compute_volume_stats(node, :cross_section_stat_mod), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod = compute_volume_stats(node, :cross_section_pipe), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod_20 = compute_volume_stats(node, :cross_section_pipe_20), symbol = "S") # volume in mm3

    # Compute the A2 volume, which is the volume of all segments they hold
    @mutate_mtg!(mtg, volume_ps3d = compute_var_axis_A2(node, :volume_ps3d), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    @mutate_mtg!(mtg, volume_stat_mod = compute_var_axis_A2(node, :volume_stat_mod), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod = compute_var_axis_A2(node, :volume_pipe_mod), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod_20 = compute_var_axis_A2(node, :volume_pipe_mod_20), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3

    # A1 volume in mm3 (just itself, excluding A2 volumes:
    mtg[1][:volume_ps3d] = compute_A1_axis_from_start(mtg[1], :volume_ps3d, id_cor_start = 0)
    mtg[1][:volume_stat_mod] = compute_A1_axis_from_start(mtg[1], :volume_stat_mod, id_cor_start = 0)
    mtg[1][:volume_pipe_mod] = compute_A1_axis_from_start(mtg[1], :volume_pipe_mod, id_cor_start = 0)
    mtg[1][:volume_pipe_mod_20] = compute_A1_axis_from_start(mtg[1], :volume_pipe_mod_20, id_cor_start = 0)

    # Branch-scale volume, the sum of A1 and all the A2:
    mtg[:volume_ps3d] = sum(descendants!(mtg, :volume_ps3d, symbol = "A", filter_fun = filter_A1_A2))
    mtg[:volume_stat_mod] = sum(descendants!(mtg, :volume_stat_mod, symbol = "A", filter_fun = filter_A1_A2))
    mtg[:volume_pipe_mod] = sum(descendants!(mtg, :volume_pipe_mod, symbol = "A", filter_fun = filter_A1_A2))
    mtg[:volume_pipe_mod_20] = sum(descendants!(mtg, :volume_pipe_mod_20, symbol = "A", filter_fun = filter_A1_A2))

    # And the biomass:
    @mutate_mtg!(mtg, fresh_mass_ps3d = node[:volume_ps3d] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass_ps3d = node[:volume_ps3d] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    @mutate_mtg!(mtg, fresh_mass = node[:volume_stat_mod] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass = node[:volume_stat_mod] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    @mutate_mtg!(mtg, fresh_mass_pipe_mod = node[:volume_pipe_mod] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass_pipe_mod = node[:volume_pipe_mod] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    @mutate_mtg!(mtg, fresh_mass_pipe_mod_20 = node[:volume_pipe_mod_20] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass_pipe_mod_20 = node[:volume_pipe_mod_20] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    # Clean-up the cached variables:
    clean_cache!(mtg)

    return nothing
end

function cross_section_stat_mod(x)
    0.217432 * x[:cross_section_pipe_20] + 0.0226391 * x[:pathlength_subtree] + 19.2056 * x[:branching_order] +
    6.99042 * x[:segment_index_on_axis] - 10.0844 * x[:number_leaves] + 3.61329 * x[:segment_subtree] +
    0.9353 * x[:n_segments_axis] - 6.03946 * x[:nleaf_proportion_siblings]
end

function compute_volume_model(branch, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, df_density)

    # Compute the average density:
    dry_density = filter(x -> x.branches == branch, df_density).dry_density[1]
    fresh_density = filter(x -> x.branches == branch, df_density).fresh_density[1]

    # Importing the mtg from the manual measurement data:
    mtg_manual = read_mtg(joinpath(dir_path_manual, branch * ".mtg"))

    # Gap-filling the measured values of the cross-section using the pipe-model (some segments were not measured):
    @mutate_mtg!(mtg_manual, cross_section_gap_filled = pipe_model!(node, :cross_section, -1, allow_missing = true))

    # Gap-filling the Length by putting 0 (if not measured, probably broken):
    gap_fill_length(x) = x[:length] === nothing ? 0 : x[:length]
    @mutate_mtg!(mtg_manual, length_gap_filled = gap_fill_length(node))

    # Compute the A2 length, which is the total length of all segments they bear:
    @mutate_mtg!(mtg_manual, length_meas = compute_var_axis_A2(node, :length_gap_filled), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    # A1 length in mm (just itself, excluding A2 length and segments not present in the LiDAR measurement):
    mtg_manual[1][:length_meas] = compute_A1_axis_from_start(mtg_manual[1], :length_gap_filled, id_cor_start = 0)

    # Recompute the volume:
    compute_volume_gapfilled(x) = x[:cross_section_gap_filled] * x[:length_gap_filled]
    @mutate_mtg!(mtg_manual, volume_gf = compute_volume_gapfilled(node), symbol = "S") # volume of the segment in mm3

    # Compute the A2 volume, which is the volume of all segments they hold
    @mutate_mtg!(mtg_manual, volume_gf = compute_var_axis_A2(node, :volume_gf), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3

    # A1 volume in mm3 (just itself, excluding A2 volumes, but also excluding the first segment because we don't know:
    mtg_manual[1][:volume_gf] = compute_A1_axis_from_start(mtg_manual[1], :volume_gf, id_cor_start = 0)

    # NB: the first matching segment is identified with a value of 0 in the `id_cor` column.

    # Branch-scale volume, the sum of A1 and all the A2:
    mtg_manual[:volume_gf] =
    sum(
        descendants!(
            mtg_manual,
            :volume_gf,
            symbol = "A",
            filter_fun = filter_A1_A2
        )
    )

    # fresh_density = mtg_manual.attributes[:mass_g] / (mtg_manual.attributes[:volume_gf] * 1e-3)
    # println("Density = $fresh_density")

    @mutate_mtg!(mtg_manual, fresh_mass = node[:volume_gf] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg_manual, dry_mass = node[:volume_gf] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    # Compute the mass of A1 using A1 = tot_mass - ∑A2:
    mass_A2 = descendants!(mtg_manual, :mass_g, symbol = "A", filter_fun = x -> x.MTG.index == 2)
    id_cor_A10 = findfirst(x -> x == 0, descendants!(mtg_manual[1], :id_cor, symbol = "S", link = ("/", "<"), all = false))
    mass_A2 = mass_A2[id_cor_A10:end]
    # NB: the A2 axis that are not found in the LiDAR data are removed from the computation (before id_cor = 0)

    # But compute it only for branches where all A2 where measured:
    if !any(mass_A2 .=== nothing)
        println("All A2 measured (ᵔᴥᵔ)")
        mtg_manual[1][:mass_g] = mtg_manual[:mass_g] - sum(mass_A2)
    end

    # Importing the mtg from the LiDAR data (plantscan3d, not corrected):
    mtg_lidar_ps3d_raw = read_mtg(joinpath(dir_path_lidar_raw, branch * ".mtg"))

    # and computing the volumes and biomass:
    @mutate_mtg!(mtg_lidar_ps3d_raw, diameter = node[:radius] * 2 * 1000, symbol = "S") # diameter in mm
    @mutate_mtg!(mtg_lidar_ps3d_raw, volume = compute_volume(node), symbol = "S") # volume in mm3
    # Compute the A2 volume, which is the volume of all segments they hold
    @mutate_mtg!(mtg_lidar_ps3d_raw, volume = compute_var_axis_A2(node, :volume), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3

    # A1 volume in mm3 (just itself, excluding A2 volumes:
    mtg_lidar_ps3d_raw[1][:volume] = compute_var_axis(mtg_lidar_ps3d_raw[1], :volume)

    # Branch-scale volume, the sum of A1 and all the A2:
    mtg_lidar_ps3d_raw[:volume] =
    sum(
        descendants!(
            mtg_lidar_ps3d_raw,
            :volume,
            symbol = "A",
            filter_fun = filter_A1_A2
        )
    )

    @mutate_mtg!(mtg_lidar_ps3d_raw, fresh_mass = node[:volume] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg_lidar_ps3d_raw, dry_mass = node[:volume] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    # Importing the mtg from the LiDAR, and compute the volume using different methods:
    mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".xlsx"))

    compute_data_mtg_lidar!(mtg_lidar_model, fresh_density, dry_density)

    return (mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model)
end

filter_A1_A2(x) = x.MTG.symbol == "A" && (x.MTG.index == 1 || x.MTG.index == 2)
filter_A1_A2_S(x) = x.MTG.symbol == "S" || filter_A1_A2(x)


function volume_stats(mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model, df_density)
    df_lidar_raw = DataFrame(mtg_lidar_ps3d_raw, [:volume, :length, :diameter])
    df_lidar_model = DataFrame(mtg_lidar_model, [:volume_ps3d, :volume_stat_mod, :volume_pipe_mod, :volume_pipe_mod_20, :length, :cross_section_stat_mod])
    df_manual = DataFrame(mtg_manual, [:volume_gf, :length_gap_filled, :cross_section_gap_filled])

    # Getting the densities:
    dry_density = filter(x -> x.branches == mtg_lidar_model.MTG.symbol, df_density).dry_density[1]
    fresh_density = filter(x -> x.branches == mtg_lidar_model.MTG.symbol, df_density).fresh_density[1]

    tot_lenght_lidar = sum(filter(x -> x.symbol == "S", df_lidar_model).length) / 1000 # length in m
    tot_lenght_lidar_raw = sum(filter(x -> x.symbol == "S", df_lidar_raw).length) / 1000 # length in m
    tot_lenght_manual = sum(filter(x -> x.symbol == "S", df_manual).length_gap_filled) / 1000
    # mean(filter(x -> x.symbol == "S", df_lidar).diameter)
    # mean(sqrt.(filter(x -> x.symbol == "S", df_lidar_model).cross_section_stat_mod ./ π) * 2)
    # mean(sqrt.(filter(x -> x.symbol == "S", df_manual).cross_section_gap_filled ./ π) * 2)
    length_error_pltscan3d =  tot_lenght_lidar - tot_lenght_manual
    length_norm_error_pltscan3d = tot_lenght_lidar / tot_lenght_manual
    length_error_pltscan3d_raw =  tot_lenght_lidar_raw - tot_lenght_manual
    length_norm_error_pltscan3d_raw = tot_lenght_lidar_raw / tot_lenght_manual

    tot_vol_lidar = filter(x -> x.scale == 1, df_lidar_model).volume_ps3d[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_raw = filter(x -> x.scale == 1, df_lidar_raw).volume[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_stat_mod = filter(x -> x.scale == 1, df_lidar_model).volume_stat_mod[1] * 1e-9 # Total volume in m3
    tot_vol_manual = filter(x -> x.scale == 1, df_manual).volume_gf[1] * 1e-9 # Total volume in m3

    volume_error_pltscan3d =  tot_vol_lidar - tot_vol_manual
    volume_norm_error_pltscan3d = tot_vol_lidar / tot_vol_manual
    volume_error_pltscan3d_raw =  tot_vol_lidar_raw - tot_vol_manual
    volume_norm_error_pltscan3d_raw = tot_vol_lidar_raw / tot_vol_manual

    volume_error_stat_model =  tot_vol_lidar_stat_mod - tot_vol_manual
    volume_norm_error_stat_model = tot_vol_lidar_stat_mod / tot_vol_manual

    # Biomass:

    # The fresh density is either taken as the average measured density at the lab or the one
    # computed from the dimension measurements and the whole branch biomass:
    actual_fresh_density = mtg_manual.attributes[:mass_g] / (tot_vol_manual * 1e6)

    dry_biomass_lidar = tot_vol_lidar * dry_density * 1000 # mass in kg
    fresh_biomass_lidar = tot_vol_lidar * fresh_density * 1000 # fresh biomass in kg
    fresh_biomass_actual_lidar = tot_vol_lidar * actual_fresh_density * 1000 # fresh biomass in kg

    dry_biomass_lidar_raw = tot_vol_lidar_raw * dry_density * 1000 # mass in kg
    fresh_biomass_lidar_raw = tot_vol_lidar_raw * fresh_density * 1000 # fresh biomass in kg
    fresh_biomass_actual_lidar_raw = tot_vol_lidar_raw * actual_fresh_density * 1000 # fresh biomass in kg

    dry_biomass_lidar_stat_mod = tot_vol_lidar_stat_mod * dry_density * 1000 # mass in kg
    fresh_biomass_lidar_stat_mod = tot_vol_lidar_stat_mod * fresh_density * 1000 # fresh biomass in kg
    # Using the density re-computed using the volume manual measurement:
    fresh_biomass_actual_stat_mod = tot_vol_lidar_stat_mod * actual_fresh_density * 1000 # fresh biomass in kg

    dry_biomass_manual = tot_vol_manual * dry_density * 1000 # mass in kg
    fresh_biomass_manual = tot_vol_manual * fresh_density * 1000 # fresh biomass in kg

    true_fresh_biomass = mtg_manual.attributes[:mass_g] / 1000

    biomass_error_pltscan3d = true_fresh_biomass - fresh_biomass_actual_lidar
    biomass_norm_error_pltscan3d = fresh_biomass_actual_lidar / true_fresh_biomass

    biomass_error_pltscan3d_raw = true_fresh_biomass - fresh_biomass_actual_lidar_raw
    biomass_norm_error_pltscan3d_raw = fresh_biomass_actual_lidar_raw / true_fresh_biomass

    biomass_error_stat_model = true_fresh_biomass - fresh_biomass_actual_stat_mod
    biomass_norm_error_stat_model = fresh_biomass_actual_stat_mod / true_fresh_biomass

    DataFrame(
        variable = ["length", "length", "length", "volume", "volume", "volume", "biomass", "biomass", "biomass"],
        model = ["plantscan3d cor.", "plantscan3d raw", "stat. model", "plantscan3d cor.", "plantscan3d raw", "stat. model", "plantscan3d cor.", "plantscan3d raw", "stat. model"],
        measurement = [tot_lenght_manual,tot_lenght_manual,tot_lenght_manual,tot_vol_manual,tot_vol_manual,tot_vol_manual,true_fresh_biomass,true_fresh_biomass,true_fresh_biomass],
        prediction = [tot_lenght_lidar,tot_lenght_lidar_raw,tot_lenght_lidar,tot_vol_lidar,tot_vol_lidar_raw,tot_vol_lidar_stat_mod,fresh_biomass_lidar,fresh_biomass_lidar_raw,fresh_biomass_lidar_stat_mod],
        error = [length_error_pltscan3d,length_error_pltscan3d_raw,length_error_pltscan3d,volume_error_pltscan3d, volume_error_pltscan3d_raw, volume_error_stat_model, biomass_error_pltscan3d, biomass_error_pltscan3d_raw, biomass_error_stat_model],
        error_norm = [length_norm_error_pltscan3d,length_norm_error_pltscan3d_raw,length_norm_error_pltscan3d,volume_norm_error_pltscan3d, volume_norm_error_pltscan3d_raw,volume_norm_error_stat_model, biomass_norm_error_pltscan3d, biomass_norm_error_pltscan3d_raw,biomass_norm_error_stat_model]
    )
end

end
