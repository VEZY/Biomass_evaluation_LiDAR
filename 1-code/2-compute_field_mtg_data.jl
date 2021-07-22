# Aim: Compute new variables in the MTG and export the results in a CSV and a new mtg.
# Author: A. Bonnet & M. Millan and R. Vezy
# Date of creation: 22/07/2021

# Imports

using MTG
using Statistics
using CSV
using DataFrames
using MTG:parent
include("1-code/functions.jl")

# Listing the mtg files:

mtg_files = filter(x -> endswith(x, ".mtg"), readdir("0-data\\2-mtg", join = true))


# Computing new variables for each mtg and saving the results in "0-data/5-enriched_manual_mtg":

for i in mtg_files
    compute_all_mtg_data(
        i,
        joinpath("0-data", "5-enriched_manual_mtg", basename(i)),
        joinpath("0-data", "5-enriched_manual_mtg", splitext(basename(i))[1] * ".csv"),
    )
end
