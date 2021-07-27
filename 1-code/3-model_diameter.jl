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
using Plots:abline!
using GLM
using Statistics
using StatsPlots
using MLBase
using BrowseTables
# using TerminalPager

# Listing the csv files:

csv_files = filter(x -> endswith(x, ".csv"), readdir("0-data\\5-enriched_manual_mtg\\", join = true))


# Computing new variables for each mtg and saving the results in "0-data/5-enriched_manual_mtg":

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

# We don't need the rows with missing values for the cross_section because this is the variable
# we want to model. Removing them:
dropmissing!(df, :cross_section)
# df |> TerminalPager.pager

# The formula used for the general model:
# formula = cross_section ~ number_leaves + pathlength_subtree +
#   segment_index_on_axis + axis_length + segment_subtree
# formula = cross_section ~ number_leaves + pathlength_subtree + segment_index_on_axis
formula = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree + topological_order + segment_index_on_axis)
min_diam = 20

ols = lm(formula, df)

cross_section ~ 0 + 0.0186699 * cross_section_pipe + 0.0427697 * pathlength_subtree + 4.27239 * topological_order + 0.513183 * segment_index_on_axis


scatter(df[!,:cross_section], df[!,:pathlength_subtree], label = "")
ylabel!("Total path length of the subtree (mm)")
xlabel!("Measured cross section (mm²)")

scatter(df[!,:cross_section], df[!,:number_leaves], label = "Measured")
ylabel!("Total number of terminal nodes (#)")
xlabel!("Measured cross section (mm²)")

scatter(df[!,:cross_section], df[!,:cross_section_pipe], label = "")
ylabel!("Pipe model's cross section (mm²)")
xlabel!("Measured cross section (mm²)")
Plots.abline!(1,0, line = :dash, label = "identity")

histogram(df[!,:diameter])



filter(x -> x.cross_section_pipe < 20, df)
filter(row -> row.sex == "male", df)


BrowseTables.open_html_table(filter(x -> x.cross_section_pipe < 50 && x.cross_section > 500, df))

BrowseTables.open_html_table(filter(x -> x.id == 381 && x.tree == "12", df))

# d	symbol	scale	index	parent_id	link    diameter    nleaf_proportion_siblings	unique_branch
# 23	S	3	22	381	<	45.0    0.9246231155778895	tree12h
