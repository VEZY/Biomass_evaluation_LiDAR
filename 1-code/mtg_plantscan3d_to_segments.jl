# Aim: transform mtg from plantscan3d lidar reconstruction to match field measurements
# Author: A. Bonnet & R. Vezy
# Date: 10/05/2021


# Script set-up -----------------------------------------------------------

# using Pkg; Pkg.add(url = "https://github.com/VEZY/MTG.jl", rev = "master")

using MTG
include("1-code/functions.jl")

# Import the mtg ----------------------------------------------------------

branch = "0-data/3-mtg-lidar/mtg-sans-corec/A1BH-sans-corec.mtg"
mtg = read_mtg(branch)

# Compute internode length and then cumulate the lenghts when deleting.

# Step 1: computes the length of each node:
@mutate_mtg!(mtg, length_node = compute_length_coord(node), scale = 2) # length is in meters

# Step 3: cumulate the length of all nodes in a segment for each segment node:
@mutate_mtg!(mtg, length = cumul_length_segment(node), scale = 2,  filter_fun = is_seg)

# Step 4: delete nodes to make the mtg as the field measurements: with nodes only at branching points
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
# mtg.MTG.symbol = replace(basename(branch), ".mtg" => "")
mtg.MTG.symbol = "tree11h"
# Last step, we add the index as in the field, *i.e.* the axis nodes are indexed following
# their topological order, and the segments are indexed following their position on the axis:
function A_indexing(node)
    if isroot(node)
        return 0
    else
        node.MTG.link == "+" ? node.parent.MTG.index + 1 : node.parent.MTG.index
    end
end

@mutate_mtg!(mtg,node.MTG.index = A_indexing(node))

# Set bag the root node with no indexing:
mtg.MTG.index = nothing

    function S_indexing(node)
    if isroot(node)
        return 0
    else
        node.MTG.link == "/" ? 1 : node.parent.MTG.index + 1
    end
end

@mutate_mtg!(mtg, node.MTG.index = S_indexing(node), scale = 3)

# Delete the old length of the nodes (length_node) from the attributes:
traverse!(mtg, x -> x[:length_node] === nothing ? nothing : pop!(x.attributes, :length_node))

# Write MTG back to file:
write_mtg("0-data/4-mtg-lidar-segments/mtg-segments-sans-corec/tree11h.mtg", mtg)
