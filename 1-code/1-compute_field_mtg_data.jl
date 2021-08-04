# Aim: Compute new variables in the MTG and export the results in a CSV and a new mtg.
# Author: A. Bonnet & M. Millan and R. Vezy
# Date of creation: 22/07/2021

# Imports

using Revise
includet("1-code/functions.jl")
using .BiomassFromLiDAR

# Listing the mtg files in xlsx/xlsm format:

mtg_files =
    filter(
        x -> splitext(basename(x))[2] in [".xlsx",".xlsm"],
        readdir(joinpath("0-data", "1-mtg_manual_measurement_corrected"), join = true)
    )

# Computing new variables for each mtg and saving the results in "0-data/5-enriched_manual_mtg":

for i in mtg_files
    compute_all_mtg_data(
        i,
        joinpath("0-data", "2-mtg_manual_measurement_corrected_enriched", splitext(basename(i))[1] * ".mtg"),
        joinpath("0-data", "2-mtg_manual_measurement_corrected_enriched", splitext(basename(i))[1] * ".csv"),
    )
end


# mtg_file = "0-data\\1-xlsx\\tree11l.xlsm"
# mtg = read_mtg(mtg_file)
# node = get_node(mtg, "node_13")
# compute_data_mtg(mtg)
# node.MTG
# node.attributes[:axis_length]
# ancestors(node, :axis_length, symbol = "A", recursivity_level = 1)
# pipe_model!(node, :cross_section_all, 314, allow_missing = false)
# desc_cross_section = descendants(node, :cross_section, symbol = "S", recursivity_level = 1)
