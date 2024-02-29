# Aim: Compute new variables in the MTG and export the results in a CSV and a new mtg.
# Author: A. Bonnet & M. Millan & R. Vezy
# Date of creation: 22/07/2021

# Imports
using MultiScaleTreeGraph
include("./functions.jl")
using .BiomassFromLiDAR

# Listing the mtg files in xlsx/xlsm format:

mtg_files =
    filter(
        x -> splitext(basename(x))[2] in [".xlsx", ".xlsm"],
        readdir(joinpath("0-data", "1.1-mtg_manual_measurement_corrected_id"), join=true)
    )

# Computing new variables for each mtg and saving the results in "0-data/5-enriched_manual_mtg":

for i in mtg_files
    println("Computing branch $(splitext(basename(i))[1])")
    compute_all_mtg_data(
        i,
        joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched", splitext(basename(i))[1] * ".mtg"),
        joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched", splitext(basename(i))[1] * ".csv"),
    )
end
