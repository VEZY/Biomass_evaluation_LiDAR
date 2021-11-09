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
# using StatsPlots
using MLBase
using BrowseTables
using TerminalPager
using Revise

includet("1-code/functions.jl") # Do not execute with alt+Enter, but rather with ctrl+Enter
using .BiomassFromLiDAR

# plotlyjs()

# Listing the csv files:
csv_files =
    filter(
        x -> endswith(x, ".csv"), # all MTGs
        # x -> endswith(x, r"tree[1,3].\.csv"), # train only on 2020 MTGs
        readdir(joinpath("0-data", "1.2-mtg_manual_measurement_corrected_enriched"), join = true)
    )

# Binding the csv files and adding the branch / tree origin:
df = bind_csv_files(csv_files)

# We don't need the rows with missing values for the cross_section because this is the variable
# we want to model. Removing them:
dropmissing!(df, :cross_section)

# We don't want the segments that were broken either:
filter!(x -> ismissing(x.comment) || !(x.comment in ["casse", "CASSE", "AVORTE", ]), df)
# df |> TerminalPager.pager

# dataframe with only the segments with a cross-section < 1963.5 mm² ≈ 50 mm diameter
min_diam = 50
min_cross_section = π * ((min_diam / 2.0)^2)
df_inf50 = filter(x -> x.cross_section < min_cross_section, df)


# The formula used for the general model:
# formula = @formula(cross_section ~ 0 + cross_section_pipe_50 + pathlength_subtree + branching_order + segment_index_on_axis + number_leaves + axis_length + segment_subtree)
# NB: using this formula we see that number_leaves, segment_subtree and segment_index_on_axis have a Pr(>|t|) > 0.05 and a low t value
# formula_all = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree + branching_order + segment_index_on_axis + axis_length + number_leaves + segment_subtree + n_segments_axis)
formula_all = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree + branching_order + segment_index_on_axis + axis_length + number_leaves + segment_subtree + n_segments_axis)

# @df dropmissing(df_inf50, [:cross_section,:length]) scatter(
#     :cross_section,
#     # :axis_length,
#     :pathlength_subtree,
#     group = :unique_branch
# )

# Model trained on all segments (not what we are searching for, but keep it as a reference):
ols_all = lm(formula_all, df)
coefnames(ols_all)
df[!,:pred_cross_section] = predict(ols_all, df)

# Same but only for segments < 50mm
formula_50 = @formula(cross_section ~ 0 + cross_section_pipe_50 + pathlength_subtree + branching_order + n_segments_axis)
# formula_50 = @formula(cross_section ~ 0 + pathlength_subtree + branching_order  + segment_index_on_axis + number_leaves + segment_subtree + n_segments_axis + nleaf_proportion_siblings)
ols_50 = lm(formula_50, df_inf50)
df[!,:pred_cross_section_50] = predict(ols_50, df)
# Setting back the bigger segments to the "measurement" as we don't want to re-simulate them:
df[df.cross_section .>= min_cross_section,:pred_cross_section_50] =
    df[df.cross_section .>= min_cross_section,:cross_section]

# Plotting the measured VS simulated cros-section with different methods:
df_plot = dropmissing(df, [:cross_section_pipe_50, :pred_cross_section, :pred_cross_section_50])

nRMSEs =
combine(
     filter(x -> x.cross_section < min_cross_section, df_plot),
    [:cross_section, :cross_section_pipe] => nRMSE => :pipe_model,
    [:cross_section, :pred_cross_section] => nRMSE => :stat_model,
    [:cross_section, :cross_section_pipe_50] => nRMSE => :pipe_model_50,
    [:cross_section, :pred_cross_section_50] => nRMSE => :stat_model_50
)

EFs =
combine(
     filter(x -> x.cross_section < min_cross_section, df_plot),
    [:cross_section, :cross_section_pipe] => EF => :pipe_model,
    [:cross_section, :pred_cross_section] => EF => :stat_model,
    [:cross_section, :cross_section_pipe_50] => EF => :pipe_model_50,
    [:cross_section, :pred_cross_section_50] => EF => :stat_model_50
)

df_plot = dropmissing(select(df, :pred_cross_section, :cross_section, :cross_section_pipe), [:pred_cross_section])

scatter(
    df_plot[!,:cross_section],
    Array(select(df_plot, :cross_section_pipe, :pred_cross_section)),
    label = hcat(
        "Pipe model, nRMSE: $(nRMSEs.pipe_model[1]), EF: $(EFs.pipe_model[1])",
        "Stat. mod., nRMSE: $(nRMSEs.stat_model[1]), EF: $(EFs.stat_model[1])"
        ),
    yguide = "Predicted cross-section (mm²)",
    xguide = "Measured cross section (mm²)",
    legend = :topleft,
    dpi = 300
)
Plots.abline!(1,0, line = :dash, label = "identity")

scatter!(
    df_plot[!,:cross_section],
    Array(select(df_plot, :cross_section_pipe, :pred_cross_section)),
    label = "",
    link = :both,
    xlims = (-Inf, min_cross_section),
    ylims = (-Inf, min_cross_section),
    inset = (1, bbox(0.0, 0.1, 0.4, 0.4, :bottom, :right)),
    subplot = 2
)
Plots.abline!(1,0, line = :dash, lw = 2, label = "", subplot = 2)


# scatter(
#     df_plot[!,:cross_section],
#     Array(select(df_plot, :cross_section_pipe, :cross_section_pipe_50, :pred_cross_section, :pred_cross_section_50)),
#     label = hcat(
#         "Pipe model, nRMSE: $(nRMSEs.pipe_model[1]), EF: $(EFs.pipe_model[1])",
#         "Pipe model ⌀<50mm, nRMSE: $(nRMSEs.pipe_model_50[1]), EF: $(EFs.pipe_model_50[1])",
#         "Stat. mod. all segments, nRMSE: $(nRMSEs.stat_model[1]), EF: $(EFs.stat_model[1])",
#         "Stat. mod. ⌀<50mm, nRMSE: $(nRMSEs.stat_model_50[1]), EF: $(EFs.stat_model_50[1])"
#         ),
#     yguide = "Predicted cross-section (mm²)",
#     xguide = "Measured cross section (mm²)",
#     legend = :topleft,
#     dpi = 300
# )
# Plots.abline!(1,0, line = :dash, label = "identity")

# scatter!(
#     df_plot[!,:cross_section],
#     Array(select(df_plot, :cross_section_pipe, :cross_section_pipe_50, :pred_cross_section, :pred_cross_section_50)),
#     label = "",
#     link = :both,
#     xlims = (-Inf, min_cross_section),
#     ylims = (-Inf, min_cross_section),
#     inset = (1, bbox(0.0, 0.1, 0.4, 0.4, :bottom, :right)),
#     subplot = 2
# )
# Plots.abline!(1,0, line = :dash, lw = 2, label = "", subplot = 2)


savefig("2-results/2-plots/step_3_fit_statistic_model.png")


# Other plots:
scatter(df[!,:diameter], df[!,:pathlength_subtree], label = "")
ylabel!("Total path length of the subtree (mm)")
xlabel!("Measured cross section (mm²)")

scatter(df_inf50[!,:cross_section], df_inf50[!,:pathlength_subtree], label = "")
ylabel!("Total path length of the subtree (mm)")
xlabel!("Measured cross section (mm²)")

filter(x -> x.pathlength_subtree > 4e4, dropmissing(df_inf50, :pathlength_subtree))

scatter(df_inf50[!,:cross_section], df_inf50[!,:branching_order], label = "Measured")
ylabel!("Topological order (#)")
xlabel!("Measured cross section (mm²)")

scatter(df_inf50[!,:cross_section], df_inf50[!,:axis_length], label = "Measured")
ylabel!("Axis length (mm)")
xlabel!("Measured cross section (mm²)")

scatter(df_inf50[!,:cross_section], df_inf50[!,:number_leaves], label = "Measured")
ylabel!("Number of terminal nodes (#)")
xlabel!("Measured cross section (mm²)")

scatter(df_inf50[!,:cross_section], df_inf50[!,:cross_section_all], label = "pipe model")

scatter(df_inf50[!,:cross_section], df_inf50[!,:cross_section_pipe], label = "pipe model")
scatter!(df_inf50[!,:cross_section], df_inf50[!,:cross_section_pipe_50], label = "pipe model < 50mm")
ylabel!("Pipe model's cross section (mm²)")
xlabel!("Measured cross section (mm²)")
Plots.abline!(1,0, line = :dash, label = "identity")

scatter(df_inf50[!,:cross_section], df_inf50[!,:cross_section_pipe], label = "pipe model")
scatter!(df_inf50[!,:cross_section], df_inf50[!,:cross_section_pipe_50], label = "pipe model < 50mm")
ylabel!("Pipe model's cross section (mm²)")
xlabel!("Measured cross section (mm²)")
Plots.abline!(1,0, line = :dash, label = "identity")

histogram(df_inf50[!,:cross_section])


BrowseTables.open_html_table(filter(x -> x.cross_section_pipe_50 < 0, df))


scatter(df_inf50[!,:cross_section], df_inf50[!,:cross_section_children], ylab = "CS children", xlab = "CS")


scatter(df[!,:cross_section], df[!,:cross_section_children], ylab = "CS children", xlab = "CS")
filter(x -> x.pathlength_subtree > 4e4, dropmissing(df, :CS))



filter(x -> x.cross_section_children < 750 && x.cross_section > 1000, dropmissing(df, [:cross_section_children]))  |> open_html_table

scatter(df[!,:cross_section], df[!,:cross_section_pipe], ylab = "CS pipe", xlab = "CS")

filter(x -> x.cross_section_children < 750 && x.cross_section > 1000, dropmissing(df, [:cross_section_children]))  |> open_html_table


scatter(df[!,:cross_section], df[!,:cross_section_pipe], ylab = "CS pipe", xlab = "CS",
    hover = df[!,:unique_branch], hovermode = "closest")

df[!,:id2] = join.([df[!,:unique_branch], df[!,:length]])

transform!(df, [:unique_branch, :length, :diameter] => ByRow((x, y, z) -> join([x,y,z], ", ")) => :id2)
transform!(df_inf50, [:unique_branch, :length, :diameter] => ByRow((x, y, z) -> join([x,y,z], ", ")) => :id2)

scatter(df[!,:cross_section], df[!,:axis_length], ylab = "axis_length", xlab = "CS",
    hover = df[!,:id2], hovermode = "closest")

scatter(df_inf50[!,:cross_section], df_inf50[!,:cross_section_pipe_50], ylab = "cross_section_pipe_50", xlab = "CS",
    hover = df_inf50[!,:id2], hovermode = "closest")

scatter(df_inf50[!,:cross_section], df_inf50[!,:pathlength_subtree], ylab = "pathlength_subtree", xlab = "CS",
    hover = df_inf50[!,:id2], hovermode = "closest")


filter(x -> x.cross_section_pipe < 100 && x.cross_section > 900, dropmissing(df, [:cross_section_pipe]))  |> open_html_table


filter(x -> x.unique_branch == "tree11l" && x.length == 80 && x.diameter == 4.39, dropmissing(df, [:cross_section_pipe]))  |> open_html_table

df = bind_csv_files(csv_files)

filter(x -> x.length === missing && x.symbol == "S", df)  |> open_html_table




bar(df[!,:diameter], df[!,:tree])


scatter(df[!,:diameter], df[!,:length], label = "")

@df dropmissing(df, [:cross_section,:pathlength_subtree]) scatter(
    :cross_section,
    :pathlength_subtree,
    group = :unique_branch
)
