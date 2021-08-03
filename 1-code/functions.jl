using MTG

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

Is a node also a segment node ? A segment node is a node at a branhing position, or at the
first (or last) position in the tree.
"""
is_seg(x) = isleaf(x) || (!isroot(x) && length(x.children) > 1)

"""
    cumul_length_segment(node)

Cumulates the lengths og segments inside a segment. Only does it if the node is considered
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

    # @mutate_mtg!(mtg, volume = compute_volume_axis(node), symbol = "A") # volume of the axis in mm3

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


function compute_volume_axis(x)
    volume_descendants = descendants!(x, :volume, symbol = "S", link = ("/", "<"), all = false)
    sum(volume_descendants)
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
            :cross_section_children, :cross_section_leaves,
            :number_leaves, :pathlength_subtree, :segment_subtree,
            :cross_section_pipe, :cross_section_pipe_20, :nleaf_proportion_siblings,
            :nleaves_siblings, :cross_section_all, :comment
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
