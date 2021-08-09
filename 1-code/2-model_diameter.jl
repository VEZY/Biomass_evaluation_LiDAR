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
using TerminalPager
using Revise

includet("1-code/functions.jl")
using .BiomassFromLiDAR

# plotlyjs()

# Listing the csv files:
csv_files =
    filter(
        x -> endswith(x, ".csv"),
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

# dataframe with only the segments with a cross-section < 314 mm² ≈ 20 mm diameter
min_diam = 20
min_cross_section = π * ((min_diam / 2.0)^2)
df_inf20 = filter(x -> x.cross_section < min_cross_section, df)

# The formula used for the general model:
# formula = @formula(cross_section ~ 0 + cross_section_pipe_20 + pathlength_subtree + branching_order + segment_index_on_axis + number_leaves + axis_length + segment_subtree)
# NB: using this formula we see that number_leaves, segment_subtree and segment_index_on_axis have a Pr(>|t|) > 0.05 and a low t value
formula_all = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree + branching_order + segment_index_on_axis + axis_length + number_leaves + segment_subtree + n_segments_axis)

# @df dropmissing(df_inf20, [:cross_section,:length]) scatter(
#     :cross_section,
#     # :axis_length,
#     :pathlength_subtree,
#     group = :unique_branch
# )

# Model trained on all segments (not what we are searching for, but keep it as a reference):
ols_all = lm(formula_all, df)
df[!,:pred_cross_section] = predict(ols_all, df)

# Same but only for segments < 20mm
formula_20 = @formula(cross_section ~ 0 + cross_section_pipe_20 + pathlength_subtree + branching_order  + segment_index_on_axis + number_leaves + segment_subtree + n_segments_axis + nleaf_proportion_siblings)
ols_20 = lm(formula_20, df_inf20)
df[!,:pred_cross_section_20] = predict(ols_20, df)
# Setting back the bigger segments to the "measurement" as we don't want to re-simulate them:
df[df.cross_section .>= min_cross_section,:pred_cross_section_20] =
    df[df.cross_section .>= min_cross_section,:cross_section]

# Plotting the measured VS simulated cros-section with different methods:
df_plot = dropmissing(df, [:cross_section_pipe_20, :pred_cross_section, :pred_cross_section_20])
RMSEs =
combine(
     filter(x -> x.cross_section < min_cross_section, df_plot),
    [:cross_section, :cross_section_pipe] => RMSE => :pipe_model,
    [:cross_section, :pred_cross_section] => RMSE => :stat_model,
    [:cross_section, :cross_section_pipe_20] => RMSE => :pipe_model_20,
    [:cross_section, :pred_cross_section_20] => RMSE => :stat_model_20
)

EFs =
combine(
     filter(x -> x.cross_section < min_cross_section, df_plot),
    [:cross_section, :cross_section_pipe] => EF => :pipe_model,
    [:cross_section, :pred_cross_section] => EF => :stat_model,
    [:cross_section, :cross_section_pipe_20] => EF => :pipe_model_20,
    [:cross_section, :pred_cross_section_20] => EF => :stat_model_20
)

scatter(
    df_plot[!,:cross_section],
    Array(select(df_plot, :cross_section_pipe, :cross_section_pipe_20, :pred_cross_section, :pred_cross_section_20)),
    label = hcat(
        "Pipe model, RMSE: $(RMSEs.pipe_model[1]), EF: $(EFs.pipe_model[1])",
        "Pipe model ⌀<20mm, RMSE: $(RMSEs.pipe_model_20[1]), EF: $(EFs.pipe_model_20[1])",
        "Stat. mod. all segments, RMSE: $(RMSEs.stat_model[1]), EF: $(EFs.stat_model[1])",
        "Stat. mod. ⌀<20mm, RMSE: $(RMSEs.stat_model_20[1]), EF: $(EFs.stat_model_20[1])"
        ),
    yguide = "Predicted cross-section (mm²)",
    xguide = "Measured cross section (mm²)",
    legend = :topleft
)
Plots.abline!(1,0, line = :dash, label = "identity")

scatter!(
    df_plot[!,:cross_section],
    Array(select(df_plot, :cross_section_pipe, :cross_section_pipe_20, :pred_cross_section, :pred_cross_section_20)),
    label = "",
    link = :both,
    xlims = (-Inf, min_cross_section),
    ylims = (-Inf, min_cross_section),
    inset = (1, bbox(0.0, 0.1, 0.4, 0.4, :bottom, :right)),
    subplot = 2
)
Plots.abline!(1,0, line = :dash, lw = 2, label = "", subplot = 2)


# The model:
# Coefficients:
# ─────────────────────────────────────────────────────────────────────────────────────────────
#                                  Coef.  Std. Error       t  Pr(>|t|)    Lower 95%   Upper 95%
# ─────────────────────────────────────────────────────────────────────────────────────────────
# cross_section_pipe_20        0.217432   0.0131183    16.57    <1e-59    0.191715    0.243149
# pathlength_subtree           0.0226391  0.00146669   15.44    <1e-51    0.0197638   0.0255144
# branching_order             19.2056     0.895974     21.44    <1e-97   17.4491     20.962
# segment_index_on_axis        6.99042    0.469006     14.90    <1e-48    6.07098     7.90986
# number_leaves              -10.0844     0.957852    -10.53    <1e-24  -11.9622     -8.20666
# segment_subtree              3.61329    0.439797      8.22    <1e-15    2.75111     4.47547
# n_segments_axis              0.9353     0.0597443    15.66    <1e-53    0.818177    1.05242
# nleaf_proportion_siblings   -6.03946    1.07981      -5.59    <1e-07   -8.15632    -3.9226
# ─────────────────────────────────────────────────────────────────────────────────────────────

function cross_section_stat_mod(x)
    0.217432 * x[:cross_section_pipe_20] + 0.0226391 * x[:pathlength_subtree] + 19.2056 * x[:branching_order] +
    6.99042 * x[:segment_index_on_axis] - 10.0844 * x[:number_leaves] + 3.61329 * x[:segment_subtree] +
    0.9353 * x[:n_segments_axis] - 6.03946 * x[:nleaf_proportion_siblings]
end

# Other plots:


scatter(df[!,:diameter], df[!,:pathlength_subtree], label = "")
ylabel!("Total path length of the subtree (mm)")
xlabel!("Measured cross section (mm²)")


scatter(df_inf20[!,:cross_section], df_inf20[!,:pathlength_subtree], label = "")
ylabel!("Total path length of the subtree (mm)")
xlabel!("Measured cross section (mm²)")

filter(x -> x.pathlength_subtree > 4e4, dropmissing(df_inf20, :pathlength_subtree))

scatter(df_inf20[!,:cross_section], df_inf20[!,:branching_order], label = "Measured")
ylabel!("Topological order (#)")
xlabel!("Measured cross section (mm²)")

scatter(df_inf20[!,:cross_section], df_inf20[!,:axis_length], label = "Measured")
ylabel!("Axis length (mm)")
xlabel!("Measured cross section (mm²)")

scatter(df_inf20[!,:cross_section], df_inf20[!,:number_leaves], label = "Measured")
ylabel!("Number of terminal nodes (#)")
xlabel!("Measured cross section (mm²)")

scatter(df_inf20[!,:cross_section], df_inf20[!,:cross_section_all], label = "pipe model")

scatter(df_inf20[!,:cross_section], df_inf20[!,:cross_section_pipe], label = "pipe model")
scatter!(df_inf20[!,:cross_section], df_inf20[!,:cross_section_pipe_20], label = "pipe model < 20mm")
ylabel!("Pipe model's cross section (mm²)")
xlabel!("Measured cross section (mm²)")
Plots.abline!(1,0, line = :dash, label = "identity")

scatter(df_inf20[!,:cross_section], df_inf20[!,:cross_section_pipe], label = "pipe model")
scatter!(df_inf20[!,:cross_section], df_inf20[!,:cross_section_pipe_20], label = "pipe model < 20mm")
ylabel!("Pipe model's cross section (mm²)")
xlabel!("Measured cross section (mm²)")
Plots.abline!(1,0, line = :dash, label = "identity")

histogram(df_inf20[!,:cross_section])


BrowseTables.open_html_table(filter(x -> x.cross_section_pipe_20 < 0, df))


scatter(df_inf20[!,:cross_section], df_inf20[!,:cross_section_children], ylab = "CS children", xlab = "CS")


scatter(df[!,:cross_section], df[!,:cross_section_children], ylab = "CS children", xlab = "CS")
filter(x -> x.pathlength_subtree > 4e4, dropmissing(df, :CS))



filter(x -> x.cross_section_children < 750 && x.cross_section > 1000, dropmissing(df, [:cross_section_children]))  |> open_html_table

scatter(df[!,:cross_section], df[!,:cross_section_pipe], ylab = "CS pipe", xlab = "CS")

filter(x -> x.cross_section_children < 750 && x.cross_section > 1000, dropmissing(df, [:cross_section_children]))  |> open_html_table


scatter(df[!,:cross_section], df[!,:cross_section_pipe], ylab = "CS pipe", xlab = "CS",
    hover = df[!,:unique_branch], hovermode = "closest")

df[!,:id2] = join.([df[!,:unique_branch], df[!,:length]])

transform!(df, [:unique_branch, :length, :diameter] => ByRow((x, y, z) -> join([x,y,z], ", ")) => :id2)
transform!(df_inf20, [:unique_branch, :length, :diameter] => ByRow((x, y, z) -> join([x,y,z], ", ")) => :id2)

scatter(df[!,:cross_section], df[!,:axis_length], ylab = "axis_length", xlab = "CS",
    hover = df[!,:id2], hovermode = "closest")

scatter(df_inf20[!,:cross_section], df_inf20[!,:cross_section_pipe_20], ylab = "cross_section_pipe_20", xlab = "CS",
    hover = df_inf20[!,:id2], hovermode = "closest")

scatter(df_inf20[!,:cross_section], df_inf20[!,:pathlength_subtree], ylab = "pathlength_subtree", xlab = "CS",
    hover = df_inf20[!,:id2], hovermode = "closest")


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
