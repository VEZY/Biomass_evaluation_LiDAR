# Aim: Use the data computed in 2-compute_all_mtg_data.jl to model the diameter of any node
# in the mtg. Note: we use the cross-section instead of the diameter because it is more
# linearly related to the other variables.
# Author: A. Bonnet & M. Millan and R. Vezy
# Date of creation: 23/07/2021

# Imports
using CSV
using DataFrames
# using StatsModels
using Plots
using GLM
using Statistics
using StatsPlots
using MLBase

# Listing the csv files:

csv_files = filter(x -> endswith(x, ".csv"), readdir("0-data\\5-enriched_manual_mtg\\", join = true))


# Computing new variables for each mtg and saving the results in "0-data/5-enriched_manual_mtg":

dfs = []
for i in csv_files
    df_i = CSV.read(i, DataFrame)
    df_i[:,:branch] .= splitext(basename(i))[1]
    transform!(
        df_i,
        :branch => x -> x[5:end - 1] => :tree,
        :branch => :unique_branch
    )

    transform!(
        df_i,
        :unique_branch => x -> x[end] => :branch
    )
    push!(dfs, df_i)
end

df = dfs[1]
for i in 2:length(dfs)
    df = vcat(df, dfs[i])
end

# We don't need the rows with missing values for the cross_section because this is the variable
# we want to model. Removing them:
dropmissing!(df, :cross_section)

# The formula used for the general model:
# formula = cross_section ~ number_leaves + pathlength_subtree +
#   segment_index_on_axis + axis_length + segment_subtree
# formula = cross_section ~ number_leaves + pathlength_subtree + segment_index_on_axis
formula = @formula(cross_section ~ pathlength_subtree * topological_order)
min_diam = 20

ols = lm(formula, df)


scatter(df[!,:cross_section], df[!,:pathlength_subtree], label = "Measured")
scatter(df[!,:cross_section], df[!,:number_leaves], label = "Measured")
scatter(df[!,:cross_section], df[!,:segment_index_on_axis], label = "Measured")
scatter(df[!,:diameter], df[!,:topological_order], label = "Measured")

histogram(df[!,:diameter])
