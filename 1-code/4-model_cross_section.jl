### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 8d606a5d-4d1f-4754-98f2-80097817c479
begin
    using CSV
    using MultiScaleTreeGraph
    using DataFrames
    using GLM
    using Statistics
    using StatsBase
    using Random
    using AlgebraOfGraphics
    using CairoMakie
    using ColorSchemes
    using MLBase
    using CategoricalArrays
end

# ╔═╡ 393b8020-3743-11ec-2da9-d1600147f3d1
md"""
# Modelling cross-section surface

The purpose of this notebook is to make a model that predicts the cross-section of any segment using only features we can derive from LiDAR data.

Then we use the model to predict the cross-section of all segments from the measured branches and the whole trees.
"""

# ╔═╡ 3506b454-fb9c-4632-8dfb-15804b66add2
md"""
## Pre-requisites for the Notebook
"""

# ╔═╡ 3d8871e5-1204-46a4-a03f-a2081054d5e7
md"""
Re-defining the model names:
"""

# ╔═╡ 12c90bd3-fb41-4e5d-9752-9e482225f634
models = Dict("Topo. mod." => "SM", "Pipe mod." => "PMT")

# ╔═╡ 8b711c1e-7d4e-404b-b2c8-87f536728fee
md"""
Defining the colors for the plot:
"""

# ╔═╡ 6bee7b4a-c3a1-4562-a17f-71335b8d39ae
begin
    cols = Dict("Topo. mod." => ColorSchemes.Set2_5.colors[1], "Pipe mod." => ColorSchemes.Set2_5.colors[2])
    colors = [i.second => cols[i.first] for i in models]
end

# ╔═╡ 6b8d93de-2fb4-411d-befe-29cb29132b40
md"""
Listing the input MTG files:
"""

# ╔═╡ 796a13d2-a65c-46f6-ad42-5fd42811c8a8
csv_files =
    filter(
        x -> endswith(x, ".csv"), # all MTGs
        # x -> endswith(x, r"tree[1,3].\.csv"), # train only on 2020 MTGs
        readdir(joinpath("../0-data", "1.2-mtg_manual_measurement_corrected_enriched"), join=true)
    )

# ╔═╡ 220dfbff-15fc-4e75-a6a2-39e60c08e8dc
md"""
Importing the data into a common DataFrame:
"""

# ╔═╡ 068bccf7-7d01-40f5-b06b-97f6f51abcdd
md"""
!!! note
	The local functions definitions can be found at the end of the notebook
"""

# ╔═╡ 0b8d39b2-9255-4bd7-a02f-2cc055bf61fd
md"""
## Model training
"""

# ╔═╡ fa9cf6f4-eb79-4c70-ba1f-4d80b3c3e62a
md"""
First, we define which variables will be used in our model. In our case we will use all data we can derive from the LiDAR.
"""

# ╔═╡ 3d0a6b24-f11b-4f4f-b59b-5c40ea9be838
formula_all = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree + branching_order + segment_index_on_axis + number_leaves + segment_subtree)

# ╔═╡ b8800a04-dedb-44b3-82fe-385e3db1d0d5
md"""
### Model k-fold cross-validation

The first step is to evaluate our model using a k-fold validation to get a better estimate of its statistics. It helps us evaluate the structure of the model (*e.g.* which variables to use), while getting a good idea of its prediction capability. The model is trained and evaluated k-times on sub-samples of the data. The output statistic is then an out-of-sample statistic, which is much more conservative than the in-sample one.

Here's the nRMSE resulting from the cross-validation:
"""

# ╔═╡ bde004a8-d54c-4049-98f6-87c579785641
md"""
### Model training and evaluation on the whole data

Then, we train our model on the whole data to compute the right parameters for use in our next reconstructions.
"""

# ╔═╡ a7bf20e9-211c-4161-a5d2-124866afa76e
md"""
*Table 1. Linear model summary.*
"""

# ╔═╡ f2eb6a9d-e788-46d0-9957-1bc22a98ad5d
md"""
## Model evaluation

The model is then calibrated and evaluated on the whole dataset. Fig. 1 shows the accuracy of both modelling approaches compared to observations. Both approaches are robust and show a high modelling efficiency and relatively low error, however the pipe model presents a higher positive bias (Table 2).
"""

# ╔═╡ 3944b38d-f45a-4ff9-8348-98a8e04d4ad1
md"""
*Table 2. Prediction accuracy of the two modelling approach, computed with normalized root mean squared error (nRMSE, %) and modelling efficiency (EF).*
"""

# ╔═╡ 0e160fb4-f2ba-4321-b661-4ce6eb422c6b
md"""
*Figure 3. Figure 1 + FIgure 2.*
"""

# ╔═╡ 9c04906b-10cd-4c53-a879-39d168e5bd1f
md"""
## Compute the volume for LiDAR-based MTGs
"""

# ╔═╡ e5c0c40a-eb0a-4726-b58e-59c64cb39eae
md"""
### Importing the data
"""

# ╔═╡ d66aebf5-3681-420c-a342-166ea05dda2e
md"""
Importing the wood density data
"""

# ╔═╡ 7de574d4-a8b8-4945-a2f1-5b2928b9d231
df_density = let
    df_dens1 = CSV.read("../0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-2-branches-juin.csv", DataFrame, normalizenames=true)
    df_dens2 = CSV.read("../0-data/0-raw/2-manual_measurements/2-wood_density_measurements/sample-data-4-branches-avril.csv", DataFrame, normalizenames=true)

    x = vcat(df_dens1, df_dens2)

    select!(
        x,
        :branches,
        [:dry_weight_g, :volume_without_parafilm_cm3] => ((x, y) -> x ./ y) => :dry_density,
        :conventional_method_density => :fresh_density
    )

    x = groupby(x, :branches)
    combine(x, :dry_density => mean, :fresh_density => mean, renamecols=false)
end

# ╔═╡ f26a28b2-d70e-4543-b58e-2d640c2a0c0d
md"""
Importing the MTG files
"""

# ╔═╡ 9290e9bf-4c43-47c7-96ec-8b44ad3c6b23
begin
    dir_path_lidar = joinpath("..", "0-data", "3-mtg_lidar_plantscan3d", "5-corrected_segmentized_id")
    dir_path_lidar_raw = joinpath("..", "0-data", "3-mtg_lidar_plantscan3d", "3-raw_output_segmentized")
    dir_path_manual = joinpath("..", "0-data", "1.2-mtg_manual_measurement_corrected_enriched")
    dir_path_lidar_new = joinpath("..", "0-data", "3-mtg_lidar_plantscan3d", "6-corrected_segmentized_id_enriched")
    mtg_files =
        filter(
            x -> splitext(basename(x))[2] in [".xlsx"],
            readdir(dir_path_lidar)
        )
end

# ╔═╡ 466aa3b3-4c78-4bb7-944d-5d55128f8cf6
md"""
### Computing the volumes and biomass of the branches using the model
"""

# ╔═╡ b710e330-e783-48f7-bebe-4257e7fec385
md"""
Computing the cross-section, volumes and biomass of the whole trees
"""

# ╔═╡ db44b868-f9b8-466a-b13e-274ad9b9d74a
trees =
    filter(
        x -> endswith(x, ".mtg"),
        readdir("../0-data/3-mtg_lidar_plantscan3d/8-tree_scale_segmentized", join=true)
    )

# ╔═╡ f50a2242-64ee-4c91-8c9d-3d2d3f11ac5d
md"""
Write the data to disk:
"""

# ╔═╡ 1103b80b-7dc6-41ed-b5bd-a6ef739d0624
md"""
Write the plot to disk:
"""

# ╔═╡ 30f8608f-564e-4ffc-91b2-1f104fb46c1e
md"""
## References
"""

# ╔═╡ 8bb72bea-be21-47c1-bdd3-484690c3cfd4
"""
	RMSE(obs,sim; digits = 2)

Returns the Root Mean Squared Error between observations `obs` and simulations `sim`.
The closer to 0 the better.
"""
function RMSE(obs, sim; digits=2)
    return round(sqrt(sum((obs .- sim) .^ 2) / length(obs)), digits=digits)
end

# ╔═╡ ffe53b41-96bd-4f44-b313-94aabdc8b1a6
"""
	    nRMSE(obs,sim)

	Returns the normalized Root Mean Squared Error between observations `obs` and simulations `sim`.
	The closer to 0 the better.
	"""
function nRMSE(obs, sim; digits=4)
    return round(sqrt(sum((obs .- sim) .^ 2) / length(obs)) / (findmax(obs)[1] - findmin(obs)[1]), digits=digits)
end

# ╔═╡ 12d7aca9-4fa8-4461-8077-c79a99864391
"""
	    EF(obs,sim)

	Returns the Efficiency Factor between observations `obs` and simulations `sim` using [NSE (Nash-Sutcliffe efficiency) model](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient).
	The closer to 1 the better.
	"""
function EF(obs, sim, digits=4)
    SSres = sum((obs - sim) .^ 2)
    SStot = sum((obs .- mean(obs)) .^ 2)
    return round(1 - SSres / SStot, digits=digits)
end

# ╔═╡ b6fa2e19-1375-45eb-8f28-32e1d00b5243
"""
	    RME(obs,sim)

	Relative mean error between observations `obs` and simulations `sim`.
	The closer to 0 the better.
	"""
function RME(obs, sim, digits=4)
    return round(mean((sim .- obs) ./ obs), digits=digits)
end

# ╔═╡ 21fd863d-61ed-497e-ba3c-5f327e354cee
"""
	    Bias(obs,sim)

	Returns the bias between observations `obs` and simulations `sim`.
	The closer to 0 the better.
	"""
function Bias(obs, sim, digits=4)
    return round(mean(sim .- obs), digits=digits)
end

# ╔═╡ 45b20f0f-2018-406f-89b6-b8e058909e66
"""
	nBias(obs,sim; digits = 2)

Returns the normalised bias (%) between observations `obs` and simulations `sim`.
The closer to 0 the better.
"""
function nBias(obs, sim; digits=2)
    return round(mean((sim .- obs)) / (findmax(obs)[1] - findmin(obs)[1]), digits=digits)
end

# ╔═╡ 0195ac30-b64f-409a-91ad-e68cf37d7c3b
function bind_csv_files(csv_files)
    dfs = []
    for i in csv_files
        df_i = CSV.read(i, DataFrame)
        df_i[:, :branch] .= splitext(basename(i))[1]

        transform!(
            df_i,
            :branch => ByRow(x -> x[5:end-1]) => :tree
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


# ╔═╡ 492fc741-7a3b-4992-a453-fcac2bbf35ad
df = let
    x = dropmissing(bind_csv_files(csv_files), :cross_section)
    filter!(x -> ismissing(x.comment) || !(x.comment in ["casse", "CASSE", "AVORTE", "Portait aussi un axe peut-être cassé lors de la manip"]), x)
    filter!(x -> x.symbol == "S", x)
    # filter!(x -> !in(x.tree, ["2","4"]), x) # Remove pruned trees
    filter!(x -> in(x.tree, ["11", "12", "13"]), x) # Remove 2020 trees
    x
end

# ╔═╡ 68fdcbf2-980a-4d44-b1f9-46b8fcd5bea1
begin
    # Define the function to compute the RMSE for each cross-validation fold:
    function compute_rmse(mod, df_ref)
        x = deepcopy(df_ref)
        x[!, "Topo. mod."] = predict(mod, x)
        x = dropmissing(x, ["Topo. mod.", "cross_section"])
        nRMSE(x[:, "Topo. mod."], x[:, :cross_section])
    end

    # Sample size
    const n = size(df, 1)
    Random.seed!(1234)
    # cross validation
    scores = cross_validate(
        inds -> lm(formula_all, df[inds, :]),        # training function
        (mod, inds) -> compute_rmse(mod, df[inds, :]),  # evaluation function
        n,              # total number of samples
        Kfold(n, 10)    # 10-fold cross validation plan
    )
    # get the mean and std of the scores
    (m, s) = round.(mean_and_std(scores), digits=4)

    "nRMSE: $m ± sd $s"
end

# ╔═╡ aaa829ee-ec36-4116-8424-4b40c581c2fc
model = lm(formula_all, df)

# ╔═╡ b49c4235-a09e-4b8c-a392-d423d7ed7d4c
df_all = let x = deepcopy(df)
    x[:, "Topo. mod."] = predict(model, x)
    rename!(x, Dict(:cross_section_pipe => "Pipe mod."))

    x = DataFrames.stack(
        dropmissing(x, ["Pipe mod.", "Topo. mod.", "cross_section"]),
        ["Pipe mod.", "Topo. mod."],
        [:tree, :branch, :id, :symbol, :scale, :index, :parent_id, :link, :cross_section],
        variable_name=:origin,
        value_name=:cross_section_pred
    )
    transform!(x, :origin => ByRow(x -> models[x]) => :Model)
end

# ╔═╡ d587f110-86d5-41c0-abc7-2671d711fbdf
begin
    plt_cs_all =
        data(df_all) *
        (
            mapping(
                :cross_section => "Measured CSA (mm²)",
                :cross_section => "Predicted CSA (mm²)"
            ) *
            visual(Lines) +
            mapping(
                :cross_section => "Measured CSA (mm²)",
                :cross_section_pred => "Predicted CSA (mm²)",
                color=:Model => "Model:",
                marker=(:tree, :branch) => ((x, y) -> join([x, y], "-")) => "Tree-Branch:"
            ) *
            visual(Scatter, markersize=8, alpha=0.8)
        )
    p = draw(plt_cs_all, palettes=(; color=colors))
end

# ╔═╡ 806c5fba-6166-41d9-a109-9fac15eb107a
save("../2-results/2-plots/step_3_statistical_model_evaluation.png", p, px_per_unit=3)

# ╔═╡ e2f20d4c-77d9-4b95-b30f-63febb7888c3
md"""
*Figure 1. Measured (x-axis) and predicted (y-axis) cross-section at axis scale (n = $(nrow(df_all)÷length(unique(df_all.Model))) for each model). The prediction is done either using the topological model (SM), or the pipe model (PMT).*
"""

# ╔═╡ dc2bd8f0-c321-407f-9592-7bcdf45f9634
begin
    stats_all =
        combine(
            groupby(df_all, [:Model]),
            [:cross_section_pred, :cross_section] => RMSE => :RMSE,
            [:cross_section_pred, :cross_section] => nRMSE => :nRMSE,
            [:cross_section_pred, :cross_section] => EF => :EF,
            [:cross_section_pred, :cross_section] => Bias => :Bias,
            [:cross_section_pred, :cross_section] => ((x, y) -> nBias(x, y, digits=4)) => :nBias
        )
    sort(stats_all, :nRMSE)
end

# ╔═╡ 19bc88a8-8299-49fe-bf19-60e0fdfbd1ba
begin
    p2, plt_cs_all2, xticks2 = let
        #classes = 0.0:0.001:0.10
        classes = 0.0:1.0:80

        df_plot = DataFrames.transform(
            df_all,
            :cross_section_pred => (x -> sqrt.(x ./ π) .* 1e-3 .* 2 .* 1000) => :diameter_pred,
            :cross_section => (x -> sqrt.(x ./ π) .* 1e-3 .* 2 .* 1000) => :diameter,
        )

        filter!(row -> row.diameter < maximum(df_plot.diameter), df_plot)

        transform!(
            df_plot,
            [:diameter_pred, :diameter] => ((pred, obs) -> (obs .- pred) ./ obs .* 100) => :diameter_error_rel,
            [:diameter_pred, :diameter] => ((pred, obs) -> (obs .- pred)) => :diameter_error,
        )

        transform!(
            df_plot,
            :diameter => (x -> cut(x, classes, labels=classes[2:end])) => :diameter_class,
        )

        df_plot = combine(
            groupby(df_plot, [:Model, :diameter_class]),
            :diameter_error => mean => :diameter_error,
            :diameter => mean => :diameter,
            :diameter_pred => mean => :diameter_pred,
        )

        sort!(df_plot, order(:diameter))
        classes_points = unique(df_plot.diameter_class)
        nclasses = length(classes_points)
        plt_ =
            data(df_plot) *
            mapping(
                :diameter_class,
                :diameter_error,
                color=:Model,
                dodge=:Model,
            ) *
            visual(BarPlot)#, dodge_gap=0.1, gap=0.1)
        xticks = (0:10:nclasses, string.(Int.(unwrap.(df_plot.diameter_class[1:10:nclasses]))))
        p_ = draw(
            plt_;
            figure=(fontsize=25,),
            palettes=(; color=colors),
            axis=
            (;
                xticks=xticks,
                #width=800, height=800, 
                ylabel="Prediction error (mm)", xlabel="Diameter (mm)",
            )
        )
        p_, plt_, xticks
    end
    p2
end

# ╔═╡ c0ea7917-f0bf-445a-a612-40abd67fe9ad
p3 = let
    f = Figure(; size=(800, 300))
    ax1 = Axis(f[1, 1], aspect=1, ylabel="Predicted CSA (mm²)", xlabel="Measured CSA (mm²)", title="1. Prediction~Measurement", titlealign=:left)
    ax2 = Axis(f[1, 2], aspect=1, ylabel="Prediction error (mm)", xlabel="Diameter (mm)", title="2. Error by diameter class", titlealign=:left, xticks=xticks2)

    d1 = draw!(ax1, plt_cs_all, palettes=(; color=colors),)
    draw!(ax2, plt_cs_all2, palettes=(; color=colors),)
    legend!(f[1, 3], d1)
    f
end

# ╔═╡ bfe70951-a89d-4783-98a8-ac39fabed66b
save("../2-results/2-plots/Figure_5-statistical_model_evaluation.png", p3, px_per_unit=3)

# ╔═╡ a802bffa-a5c3-4563-adde-7a393b47881c
save("../2-results/2-plots/step_3_statistical_model_evaluation2.png", p2, px_per_unit=3)

# ╔═╡ 61f8f651-1dba-4b91-a46e-bbe61cffb725
md"""
*Figure 2. Measured diameter (x-axis, m) and relative prediction error (y-axis, m) of segment diameter from the pipe model theory (PMT) and the structural model (SM). n = $(nrow(df_all)÷length(unique(df_all.Model))) for each model.*
"""

# ╔═╡ d7a3c496-0ef0-454b-9e32-e5835928f4d5
function compute_cross_section_all(x, var=:cross_section)
    if x.MTG.symbol == "A"
        desc_cross_section = descendants(x, var, symbol="S", recursivity_level=1)
        if length(desc_cross_section) > 0
            return desc_cross_section[1]
        else
            @warn "$(x.name) has no descendants with a value for $var."
        end
    else
        x[var]
    end
end

# ╔═╡ eb39ed1b-6dee-4738-a762-13b759f74411
"""
	compute_A1_axis_from_start(x, vol_col = :volume; id_cor_start)

Compute the sum of a variable over the axis starting from node that has `id_cor_start` value.
"""
function compute_A1_axis_from_start(x, vol_col=:volume; id_cor_start)
    length_gf_A1 = descendants(x, vol_col, symbol="S", link=("/", "<"), all=false)
    id_cor_A1 = descendants(x, :id_cor, symbol="S", link=("/", "<"), all=false)
    first_id_cor = findfirst(x -> x == id_cor_start, id_cor_A1)
    if first_id_cor === nothing
        id_cor_A1 = descendants(x, :id_cor, symbol="S", ignore_nothing=true)
        first_id_cor = findfirst(x -> x == id_cor_start, id_cor_A1)
        if first_id_cor === nothing
            @warn "No id_cor_start found in $(x.name)."
            return 0.0
        end
    end
    sum(length_gf_A1[first_id_cor:end])
end

# ╔═╡ ee46e359-36bd-49c4-853c-d3ff29888473
function compute_var_axis_A2(x, vol_col=:volume)
    sum(descendants!(x, vol_col, symbol="S"))
end

# ╔═╡ b2e75112-be43-4df9-86df-2eeeb58f47c3
filter_A1_A2(x) = x.MTG.symbol == "A" && (x.MTG.index == 1 || x.MTG.index == 2)

# ╔═╡ b01851d1-d9d9-4016-b02e-6d3bfc449b8a
filter_A1_A2_S(x) = x.MTG.symbol == "S" || filter_A1_A2(x)

# ╔═╡ 14fde936-fa95-471a-aafb-5d69871e5a87
function compute_axis_length(x)
    length_descendants = filter(x -> x !== nothing, descendants!(x, :length, symbol="S", link=("/", "<"), all=false))
    length(length_descendants) > 0 ? sum(length_descendants) : nothing
end

# ╔═╡ e3ba9fec-c8b3-46e6-8b1d-29ab19198c9c
function get_axis_length(x)
    axis_length = ancestors(x, :axis_length, symbol="A", recursivity_level=1)
    if length(axis_length) > 0
        axis_length[1]
    else
        nothing
    end
end

# ╔═╡ 9e967170-9388-43e4-8b18-baccb18f4b4e
function compute_volume(x)
    if x[:diameter] !== nothing && x[:length] !== nothing
        π * ((x[:diameter] / 2.0)^2) * x[:length]
    end
end

# ╔═╡ 979ca113-6a22-4313-a011-0aca3cefdbf7
function compute_cross_section(x)
    if x[:diameter] !== nothing
        π * ((x[:diameter] / 2.0)^2)
    end
end

# ╔═╡ 43967391-6580-4aac-9ac1-c9effbf3c948
function compute_cross_section_children(x)
    cross_section_child = filter(x -> x !== nothing, descendants!(x, :cross_section, symbol="S", recursivity_level=1))

    return length(cross_section_child) > 0 ? sum(cross_section_child) : nothing
end

# ╔═╡ de63abdd-3879-4b7c-86f7-844f6288f987
function compute_cross_section_leaves(x)
    cross_section_leaves = filter(x -> x !== nothing, descendants!(x, :cross_section; filter_fun=isleaf))

    return length(cross_section_leaves) > 0 ? sum(cross_section_leaves) : nothing
end

# ╔═╡ d17d7e96-bd15-4a79-9ccb-6182e7d7c023
function compute_volume_subtree(x)
    volume_descendants = filter(x -> x !== nothing, descendants!(x, :volume, symbol="S", self=true))
    length(volume_descendants) > 0 ? sum(volume_descendants) : nothing
end

# ╔═╡ 27a0dcef-260c-4a0c-bef3-04a7d1b79805
function cross_section_stat_mod(node, model)

    # Get the node attributes as a DataFrame for the model:
    attr_names = coefnames(model)
    attr_values = []

    for i in attr_names
        if i == "(Intercept)"
            continue
        end
        node_val = node[i]
        if node_val === nothing
            # No missing values allowed for predicting
            return missing
        end

        push!(attr_values, i => node_val)
    end

    predict(model, DataFrame(attr_values...))[1]
end

# ╔═╡ 666e9daf-e28f-4e14-b52a-bcc6b5aadb67
cross_section_stat_mod_all = cross_section_stat_mod

# ╔═╡ 665cb43b-ab86-4001-88a3-c67ed16b28e8
function compute_data_mtg_tree!(mtg_tree, fresh_density, dry_density, model)

    @mutate_mtg!(mtg_tree, diameter = node[:radius] * 2 * 1000, symbol = "S") # diameter in mm

    @mutate_mtg!(
        mtg_tree,
        pathlength_subtree = sum(filter(x -> x !== nothing, descendants!(node, :length, symbol="S", self=true))),
        symbol = "S",
        filter_fun = x -> x[:length] !== nothing
    )

    @mutate_mtg!(
        mtg_tree,
        segment_subtree = length(descendants!(node, :length, symbol="S", self=true)),
        number_leaves = nleaves!(node),
        symbol = "S"
    )

    branching_order!(mtg_tree, ascend=false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Compute the index of each segment on the axis in a basipetal way (from tip to base)
    @mutate_mtg!(
        mtg_tree,
        n_segments = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)),
        symbol = "A"
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    @mutate_mtg!(
        mtg_tree,
        n_segments_axis = ancestors(node, :n_segments, symbol="A")[1],
        segment_index_on_axis = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)) + 1,
        symbol = "S"
    )

    # Compute the total length of the axis in mm:
    @mutate_mtg!(
        mtg_tree,
        axis_length = compute_axis_length(node),
        symbol = "A"
    )

    # Associate the axis length to each segment:
    @mutate_mtg!(mtg_tree, axis_length = get_axis_length(node), symbol = "S")

    # How many leaves the sibling of the node has:
    @mutate_mtg!(mtg_tree, nleaves_siblings = sum(nleaves_siblings!(node)))

    # How many leaves the node has in proportion to its siblings + itself:
    @mutate_mtg!(mtg_tree, nleaf_proportion_siblings = node[:number_leaves] / (node[:nleaves_siblings] + node[:number_leaves]), symbol = "S")

    # Use the first cross-section for the first value to apply the pipe-model:
    first_cross_section = π * ((descendants(mtg_tree, :diameter, ignore_nothing=true, recursivity_level=5)[1] / 2.0)^2)
    @mutate_mtg!(mtg_tree, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(
        mtg_tree,
        (
            cross_section=first_cross_section,
            cross_section_pipe=first_cross_section,
            cross_section_stat_mod=first_cross_section
        )
    )
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    @mutate_mtg!(mtg_tree, cross_section_pipe = compute_cross_section_all(node, :cross_section_pipe))
    @mutate_mtg!(mtg_tree, cross_section_stat_mod = cross_section_stat_mod_all(node, model), symbol = "S")

    # Add the values for the axis:
    @mutate_mtg!(mtg_tree, cross_section_stat_mod = compute_cross_section_all(node, :cross_section_stat_mod))

    # Recompute the volume:
    compute_volume_stats(x, var) = x[var] * x[:length]

    @mutate_mtg!(mtg_tree, volume_stat_mod = compute_volume_stats(node, :cross_section_stat_mod), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg_tree, volume_pipe_mod = compute_volume_stats(node, :cross_section_pipe), symbol = "S") # volume in mm3

    # And the biomass:
    @mutate_mtg!(mtg_tree, fresh_mass = node[:volume_stat_mod] * fresh_density * 1e-3, symbol = "S") # in g
    @mutate_mtg!(mtg_tree, dry_mass = node[:volume_stat_mod] * dry_density * 1e-3, symbol = "S") # in g

    @mutate_mtg!(mtg_tree, fresh_mass_pipe_mod = node[:volume_pipe_mod] * fresh_density * 1e-3, symbol = "S") # in g
    @mutate_mtg!(mtg_tree, dry_mass_pipe_mod = node[:volume_pipe_mod] * dry_density * 1e-3, symbol = "S") # in g

    # Clean-up the cached variables:
    clean_cache!(mtg_tree)
end

# ╔═╡ a6ea9696-d778-4033-a7df-76da4ea1f5fe
for i in trees
    tree = match(r"[0-9]+", basename(i)).match
    println("Computing tree $tree")

    # Get the density for a tree:
    df_density_tree = df_density[replace.(df_density.branches, r"[a-z]" => "").==tree, :]

    # Compute the average per tree:
    dry_density = mean(df_density_tree.dry_density)
    fresh_density = mean(df_density_tree.fresh_density)

    # Import its MTG (computed from plantscan3d):
    mtg_tree = read_mtg(i)

    compute_data_mtg_tree!(mtg_tree, fresh_density, dry_density, model)

    # Write the computed LiDAR MTG to disk:
    write_mtg(joinpath("../0-data/3-mtg_lidar_plantscan3d/9-tree_scale_segmentized_enriched", basename(i)), mtg_tree)
end

# ╔═╡ 77486fa7-318d-4397-a792-70fd8d2148e3
function compute_data_mtg_lidar!(mtg, fresh_density, dry_density, model)

    @mutate_mtg!(mtg, diameter = node[:radius] * 2 * 1000, symbol = "S") # diameter in mm


    @mutate_mtg!(
        mtg,
        pathlength_subtree = sum(filter(x -> x !== nothing, descendants!(node, :length, symbol="S", self=true))),
        symbol = "S",
        filter_fun = x -> x[:length] !== nothing
    )

    @mutate_mtg!(
        mtg,
        segment_subtree = length(descendants!(node, :length, symbol="S", self=true)),
        number_leaves = nleaves!(node),
        symbol = "S"
    )

    branching_order!(mtg, ascend=false)
    # We use basipetal topological order (from tip to base) to allow comparisons between branches of
    # different ages (the last emitted segment will always be of order 1).

    # Compute the index of each segment on the axis in a basipetal way (from tip to base)
    @mutate_mtg!(
        mtg,
        n_segments = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)),
        symbol = "A"
    )

    # now use n_segments to compute the index of the segment on the axis (tip = 1, base = n_segments)
    @mutate_mtg!(
        mtg,
        n_segments_axis = ancestors(node, :n_segments, symbol="A")[1],
        segment_index_on_axis = length(descendants!(node, :length, symbol="S", link=("/", "<"), all=false)) + 1,
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

    # Use the first cross-section for the first value to apply the pipe-model:
    first_cross_section = filter(x -> x !== nothing, descendants(mtg, :cross_section, recursivity_level=5))[1]
    @mutate_mtg!(mtg, cross_section_pipe = pipe_model!(node, first_cross_section))

    # Adding the cross_section to the root:
    append!(
        mtg,
        (
            cross_section=first_cross_section,
            cross_section_pipe=first_cross_section,
            cross_section_stat_mod=first_cross_section
        )
    )
    # Compute the cross-section for the axes nodes using the one measured on the S just below:
    @mutate_mtg!(mtg, cross_section_ps3d = compute_cross_section_all(node))
    @mutate_mtg!(mtg, cross_section_pipe = compute_cross_section_all(node, :cross_section_pipe))

    # Use the pipe model, but only on nodes with a cross_section <= 1963.5 (≈50mm diameter)
    @mutate_mtg!(mtg, cross_section_pipe_50 = pipe_model!(node, :cross_section_ps3d, 1963.5, allow_missing=true))
    @mutate_mtg!(mtg, cross_section_pipe_50 = compute_cross_section_all(node, :cross_section_pipe_50))

    # @mutate_mtg!(mtg, cross_section_stat_mod_50 = cross_section_stat_mod(node,model), symbol = "S")

    transform!(mtg, (x -> cross_section_stat_mod(x, model)) => :cross_section_stat_mod_50, symbol="S")

    @mutate_mtg!(mtg, cross_section_stat_mod = cross_section_stat_mod_all(node, model), symbol = "S")

    # Add the values for the axis:
    @mutate_mtg!(mtg, cross_section_stat_mod = compute_cross_section_all(node, :cross_section_stat_mod))
    @mutate_mtg!(mtg, cross_section_stat_mod_50 = compute_cross_section_all(node, :cross_section_stat_mod_50))

    # Compute the A2 lengths to match measurements =total length of all segments they bear:
    @mutate_mtg!(mtg, length_sim = compute_var_axis_A2(node, :length), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    # A1 length in mm (just itself, excluding A2 length):
    mtg[1][:length_sim] = compute_A1_axis_from_start(mtg[1], :length, id_cor_start=0)

    # Recompute the volume:
    compute_volume_stats(x, var) = x[var] * x[:length]

    @mutate_mtg!(mtg, volume_ps3d = compute_volume_stats(node, :cross_section), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg, volume_stat_mod = compute_volume_stats(node, :cross_section_stat_mod), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg, volume_stat_mod_50 = compute_volume_stats(node, :cross_section_stat_mod_50), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod = compute_volume_stats(node, :cross_section_pipe), symbol = "S") # volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod_50 = compute_volume_stats(node, :cross_section_pipe_50), symbol = "S") # volume in mm3

    # Compute the A2 volume, which is the volume of all segments they hold
    @mutate_mtg!(mtg, volume_ps3d = compute_var_axis_A2(node, :volume_ps3d), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    @mutate_mtg!(mtg, volume_stat_mod = compute_var_axis_A2(node, :volume_stat_mod), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    @mutate_mtg!(mtg, volume_stat_mod_50 = compute_var_axis_A2(node, :volume_stat_mod_50), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod = compute_var_axis_A2(node, :volume_pipe_mod), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    @mutate_mtg!(mtg, volume_pipe_mod_50 = compute_var_axis_A2(node, :volume_pipe_mod_50), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3

    # A1 volume in mm3 (just itself, excluding A2 volumes:
    mtg[1][:volume_ps3d] = compute_A1_axis_from_start(mtg[1], :volume_ps3d, id_cor_start=0)
    mtg[1][:volume_stat_mod] = compute_A1_axis_from_start(mtg[1], :volume_stat_mod, id_cor_start=0)
    mtg[1][:volume_stat_mod_50] = compute_A1_axis_from_start(mtg[1], :volume_stat_mod_50, id_cor_start=0)
    mtg[1][:volume_pipe_mod] = compute_A1_axis_from_start(mtg[1], :volume_pipe_mod, id_cor_start=0)
    mtg[1][:volume_pipe_mod_50] = compute_A1_axis_from_start(mtg[1], :volume_pipe_mod_50, id_cor_start=0)

    # Branch-scale volume, the sum of A1 and all the A2:
    mtg[:volume_ps3d] = sum(descendants!(mtg, :volume_ps3d, symbol="A", filter_fun=filter_A1_A2))
    mtg[:volume_stat_mod] = sum(descendants!(mtg, :volume_stat_mod, symbol="A", filter_fun=filter_A1_A2))
    mtg[:volume_stat_mod_50] = sum(descendants!(mtg, :volume_stat_mod_50, symbol="A", filter_fun=filter_A1_A2))
    mtg[:volume_pipe_mod] = sum(descendants!(mtg, :volume_pipe_mod, symbol="A", filter_fun=filter_A1_A2))
    mtg[:volume_pipe_mod_50] = sum(descendants!(mtg, :volume_pipe_mod_50, symbol="A", filter_fun=filter_A1_A2))

    # And the biomass:
    @mutate_mtg!(mtg, fresh_mass_ps3d = node[:volume_ps3d] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass_ps3d = node[:volume_ps3d] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    @mutate_mtg!(mtg, fresh_mass = node[:volume_stat_mod] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass = node[:volume_stat_mod] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    @mutate_mtg!(mtg, fresh_mass_50 = node[:volume_stat_mod_50] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass_50 = node[:volume_stat_mod_50] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    @mutate_mtg!(mtg, fresh_mass_pipe_mod = node[:volume_pipe_mod] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass_pipe_mod = node[:volume_pipe_mod] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    @mutate_mtg!(mtg, fresh_mass_pipe_mod_50 = node[:volume_pipe_mod_50] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg, dry_mass_pipe_mod_50 = node[:volume_pipe_mod_50] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    # Clean-up the cached variables:
    clean_cache!(mtg)

    return nothing
end

# ╔═╡ 97871566-4904-4b40-a631-98f7e837a2f4
function compute_volume_model(branch, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, df_density, model)

    # Compute the average density:
    dry_density = filter(x -> x.branches == branch, df_density).dry_density[1]
    fresh_density = filter(x -> x.branches == branch, df_density).fresh_density[1]

    # Importing the mtg from the manual measurement data:
    mtg_manual = read_mtg(joinpath(dir_path_manual, branch * ".mtg"))

    # Gap-filling the measured values of the cross-section using the pipe-model (some segments were not measured):
    @mutate_mtg!(mtg_manual, cross_section_gap_filled = pipe_model!(node, :cross_section, -1, allow_missing=true))

    # Add the cross-section to the axis:
    @mutate_mtg!(mtg_manual, cross_section = compute_cross_section_all(node, :cross_section))

    # Gap-filling the Length by putting 0 (if not measured, probably broken):
    gap_fill_length(x) = x[:length] === nothing ? 0 : x[:length]
    @mutate_mtg!(mtg_manual, length_gap_filled = gap_fill_length(node))

    # Compute the A2 length, which is the total length of all segments they bear:
    @mutate_mtg!(mtg_manual, length_gap_filled = compute_var_axis_A2(node, :length_gap_filled), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3
    # A1 length in mm (just itself, excluding A2 length and segments not present in the LiDAR measurement):
    mtg_manual[1][:length_gap_filled] = compute_A1_axis_from_start(mtg_manual[1], :length_gap_filled, id_cor_start=0)

    # Recompute the volume:
    compute_volume_gapfilled(x) = x[:cross_section_gap_filled] * x[:length_gap_filled]
    @mutate_mtg!(mtg_manual, volume_gf = compute_volume_gapfilled(node), symbol = "S") # volume of the segment in mm3

    # Compute the A2 volume, which is the volume of all segments they hold
    @mutate_mtg!(mtg_manual, volume_gf = compute_var_axis_A2(node, :volume_gf), symbol = "A", filter_fun = x -> x.MTG.index == 2) # Axis volume in mm3

    # A1 volume in mm3 (just itself, excluding A2 volumes, but also excluding the first segment because we don't know:
    mtg_manual[1][:volume_gf] = compute_A1_axis_from_start(mtg_manual[1], :volume_gf, id_cor_start=0)

    # NB: the first matching segment is identified with a value of 0 in the `id_cor` column.

    # Branch-scale volume, the sum of A1 and all the A2:
    mtg_manual[:volume_gf] =
        sum(
            descendants!(
                mtg_manual,
                :volume_gf,
                symbol="A",
                filter_fun=filter_A1_A2
            )
        )

    # fresh_density = mtg_manual.attributes[:mass_g] / (mtg_manual.attributes[:volume_gf] * 1e-3)
    # println("Density = $fresh_density")

    @mutate_mtg!(mtg_manual, fresh_mass = node[:volume_gf] * fresh_density * 1e-3, filter_fun = filter_A1_A2_S) # in g
    @mutate_mtg!(mtg_manual, dry_mass = node[:volume_gf] * dry_density * 1e-3, filter_fun = filter_A1_A2_S) # in g

    # Compute the mass of A1 using A1 = tot_mass - ∑A2:
    mass_A2 = descendants!(mtg_manual, :mass_g, symbol="A", filter_fun=x -> x.MTG.index == 2)
    id_cor_A10 = findfirst(x -> x == 0, descendants!(mtg_manual[1], :id_cor, symbol="S", link=("/", "<"), all=false))
    mass_A2 = mass_A2[id_cor_A10:end]
    # NB: the A2 axis that are not found in the LiDAR data are removed from the computation (before id_cor = 0)

    # But compute it only for branches where all A2 where measured:
    if !any(mass_A2 .=== nothing)
        println("All A2 measured (ᵔᴥᵔ)")
        mtg_manual[1][:mass_g] = mtg_manual[:mass_g] - sum(mass_A2)
    end

    # Importing the mtg from the LiDAR data (plantscan3d, not corrected):
    mtg_lidar_ps3d_raw = read_mtg(joinpath(dir_path_lidar_raw, branch * ".mtg"))

    # Add the id for the first segment that we can match with the manual measurement:
    id_cor0_raw = Dict("tree11h" => "node_21", "tree11l" => "node_7", "tree12h" => "node_49", "tree12l" => "node_7", "tree13h" => "node_4", "tree13l" => "node_7")
    get_node(mtg_lidar_ps3d_raw, id_cor0_raw[branch])[:id_cor] = 0

    compute_data_mtg_lidar!(mtg_lidar_ps3d_raw, fresh_density, dry_density, model)

    # Importing the mtg from the LiDAR, and compute the volume using different methods:
    mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".xlsx"))

    compute_data_mtg_lidar!(mtg_lidar_model, fresh_density, dry_density, model)

    return (mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model)
end

# ╔═╡ 073e32dd-c880-479c-8933-d53c9655a04d
function volume_stats(mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model, df_density)
    df_lidar_raw = DataFrame(mtg_lidar_ps3d_raw, [:volume_ps3d, :volume_stat_mod, :volume_pipe_mod, :volume_pipe_mod_50, :length, :cross_section_stat_mod])
    df_lidar_model = DataFrame(mtg_lidar_model, [:volume_ps3d, :volume_stat_mod, :volume_pipe_mod, :volume_pipe_mod_50, :length, :cross_section_stat_mod])
    df_manual = DataFrame(mtg_manual, [:volume_gf, :length_gap_filled, :cross_section_gap_filled])

    # Getting the densities:
    dry_density = filter(x -> x.branches == mtg_lidar_model.MTG.symbol, df_density).dry_density[1]
    fresh_density = filter(x -> x.branches == mtg_lidar_model.MTG.symbol, df_density).fresh_density[1]

    tot_lenght_lidar = sum(filter(x -> x.symbol == "S", df_lidar_model).length) / 1000 # length in m
    tot_lenght_lidar_raw = sum(filter(x -> x.symbol == "S", df_lidar_raw).length) / 1000 # length in m
    tot_lenght_manual = sum(filter(x -> x.symbol == "S", df_manual).length_gap_filled) / 1000

    tot_vol_lidar = filter(x -> x.scale == 1, df_lidar_model).volume_ps3d[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_raw = filter(x -> x.scale == 1, df_lidar_raw).volume_ps3d[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_stat_mod = filter(x -> x.scale == 1, df_lidar_model).volume_stat_mod[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_pipe_mod = filter(x -> x.scale == 1, df_lidar_model).volume_pipe_mod[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_stat_mod_raw = filter(x -> x.scale == 1, df_lidar_raw).volume_stat_mod[1] * 1e-9 # Total volume in m3
    tot_vol_lidar_pipe_mod_raw = filter(x -> x.scale == 1, df_lidar_raw).volume_pipe_mod[1] * 1e-9 # Total volume in m3
    tot_vol_manual = filter(x -> x.scale == 1, df_manual).volume_gf[1] * 1e-9 # Total volume in m3

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
    fresh_biomass_lidar_stat_mod_raw = tot_vol_lidar_stat_mod_raw * fresh_density * 1000 # fresh biomass in kg

    fresh_biomass_lidar_pipe_mod = tot_vol_lidar_pipe_mod * fresh_density * 1000 # fresh biomass in kg
    fresh_biomass_lidar_pipe_mod_raw = tot_vol_lidar_pipe_mod_raw * fresh_density * 1000 # fresh biomass in kg

    dry_biomass_manual = tot_vol_manual * dry_density * 1000 # mass in kg
    fresh_biomass_manual = tot_vol_manual * fresh_density * 1000 # fresh biomass in kg

    true_fresh_biomass = mtg_manual.attributes[:mass_g] / 1000

    DataFrame(
        variable=["length", "length", "volume", "volume", "volume", "volume", "volume", "volume", "biomass", "biomass", "biomass", "biomass", "biomass", "biomass"],
        model=["plantscan3d cor.", "plantscan3d raw", "plantscan3d cor.", "plantscan3d raw", "Topo. model cor.", "Pipe model cor.", "Topo. model raw", "Pipe model raw", "plantscan3d cor.", "plantscan3d raw", "Topo. model cor.", "Pipe model cor.", "Topo. model raw", "Pipe model raw"],
        measurement=[tot_lenght_manual, tot_lenght_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass],
        prediction=[tot_lenght_lidar, tot_lenght_lidar_raw, tot_vol_lidar, tot_vol_lidar_raw, tot_vol_lidar_stat_mod, tot_vol_lidar_pipe_mod, tot_vol_lidar_stat_mod_raw, tot_vol_lidar_pipe_mod_raw, fresh_biomass_lidar, fresh_biomass_lidar_raw, fresh_biomass_lidar_stat_mod, fresh_biomass_lidar_pipe_mod, fresh_biomass_lidar_stat_mod_raw, fresh_biomass_lidar_pipe_mod_raw]
    )
end

# ╔═╡ 0a19ac96-a706-479d-91b5-4ea3e091c3e8
function summarize_data(mtg_files, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, dir_path_lidar_new, model)
    branches = first.(splitext.(mtg_files))

    df_stats_branch = DataFrame(
        :branch => String[],
        :variable => String[],
        :measurement => Float64[],
        :prediction => Float64[],
        :model => String[]
    )

    df_all = DataFrame(
        :origin => String[],
        :branch => String[],
        :id => Int[],
        :symbol => String[],
        :scale => Int[],
        :index => Int[],
        :parent_id => Int[],
        :link => Float64[],
        :mass_g => Float64[],
        :fresh_mass => Float64[],
        :volume => Float64[],
        :length => Float64[],
        :cross_section => Float64[],
        :id_cor => Int[]
    )

    for i in branches
        println("Computing branch $i")
        (mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model) =
            compute_volume_model(i, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, df_density, model)

        # Write the computed LiDAR MTG to disk:
        write_mtg(joinpath(dir_path_lidar_new, i * ".mtg"), mtg_lidar_model)

        df = volume_stats(mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model, df_density)
        df[!, :branch] .= i
        df_stats_branch = vcat(df_stats_branch, df)

        # Manual measurement:
        df_manual = DataFrame(mtg_manual, [:mass_g, :cross_section, :length_gap_filled, :fresh_mass, :volume_gf, :id_cor])
        df_manual[!, :branch] .= i
        rename!(df_manual, Dict(:volume_gf => "volume", :length_gap_filled => "length"))
        df_manual[!, :origin] .= "measurement"

        # plantscan3d, as-is:
        df_ps3d_raw = DataFrame(mtg_lidar_ps3d_raw, [:fresh_mass, :volume_ps3d])
        rename!(df_ps3d_raw, Dict(:volume_ps3d => "volume"))
        df_ps3d_raw[!, :branch] .= i
        # df_ps3d_raw[!,:id_cor] .= missing
        df_ps3d_raw[!, :origin] .= "plantscan3d, raw"

        # topological model:
        df_stat_mod = DataFrame(
            mtg_lidar_model,
            [
                :length_sim,
                :cross_section_ps3d,
                :cross_section_pipe,
                :cross_section_pipe_50,
                :cross_section_stat_mod_50,
                :cross_section_stat_mod,
                :fresh_mass,
                :fresh_mass_50,
                :fresh_mass_ps3d,
                :fresh_mass_pipe_mod,
                :fresh_mass_pipe_mod_50,
                :volume_stat_mod,
                :volume_stat_mod_50,
                :volume_ps3d,
                :volume_pipe_mod,
                :volume_pipe_mod_50,
                :id_cor
            ]
        )

        df_stat_mod_biomass = select(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :id_cor,
            :fresh_mass => "Topo. mod.",
            :fresh_mass_50 => "Topo. mod. ⌀<50",
            :fresh_mass_ps3d => "plantscan3d",
            :fresh_mass_pipe_mod => "Pipe model",
            :fresh_mass_pipe_mod_50 => "Pipe mod. ⌀<50"
        )

        df_stat_mod_biomass = DataFrames.stack(
            df_stat_mod_biomass,
            ["Topo. mod.", "Topo. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor],
            variable_name=:origin,
            value_name=:fresh_mass
        )


        df_stat_mod_cs = select(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :id_cor,
            :cross_section_stat_mod => "Topo. mod.",
            :cross_section_stat_mod_50 => "Topo. mod. ⌀<50",
            :cross_section_ps3d => "plantscan3d",
            :cross_section_pipe => "Pipe model",
            :cross_section_pipe_50 => "Pipe mod. ⌀<50"
        )

        df_stat_mod_cs = DataFrames.stack(
            df_stat_mod_cs,
            ["Topo. mod.", "Topo. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor],
            variable_name=:origin,
            value_name=:cross_section
        )

        select!(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :length_sim => :length, :id_cor,
            :volume_stat_mod => "Topo. mod.",
            :volume_stat_mod_50 => "Topo. mod. ⌀<50",
            :volume_ps3d => "plantscan3d",
            :volume_pipe_mod => "Pipe model",
            :volume_pipe_mod_50 => "Pipe mod. ⌀<50"
        )

        df_stat_mod = DataFrames.stack(
            df_stat_mod,
            ["Topo. mod.", "Topo. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor, :length],
            variable_name=:origin,
            value_name=:volume
        )

        df_stat_mod = leftjoin(df_stat_mod, df_stat_mod_biomass[:, [:origin, :id, :fresh_mass]], on=[:origin, :id])
        df_stat_mod = leftjoin(df_stat_mod, df_stat_mod_cs[:, [:origin, :id, :cross_section]], on=[:origin, :id])
        df_stat_mod[!, :branch] .= i

        df_all = vcat(
            df_all,
            df_manual[:, Not(:tree)],
            df_ps3d_raw[:, Not(:tree)],
            df_stat_mod,
            cols=:union
        )
    end

    return df_all, df_stats_branch
end

# ╔═╡ 87140df4-3fb5-443c-a667-be1f19b016f6
df_all_branches, df_stats_branch = summarize_data(mtg_files, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, dir_path_lidar_new, model);

# ╔═╡ 73515bd3-0124-42a4-9997-3730e7dcbf4c
begin
    CSV.write("../2-results/1-data/df_stats_branch.csv", df_stats_branch)
    CSV.write("../2-results/1-data/df_all.csv", df_all_branches, delim=";")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
MultiScaleTreeGraph = "dd4a991b-8a45-4075-bede-262ee62d5583"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.6.11"
CSV = "~0.10.4"
CairoMakie = "~0.11.9"
CategoricalArrays = "~0.10.8"
ColorSchemes = "~3.24.0"
DataFrames = "~1.3.4"
GLM = "~1.8.0"
MLBase = "~0.9.0"
MultiScaleTreeGraph = "~0.12.2"
StatsBase = "~0.34.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "6777684974abdcf8a2c8fd67100f1011cadb8d2d"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "222ee9e50b98f51b5d78feb93dd928880df35f06"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.3.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AlgebraOfGraphics]]
deps = ["Colors", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "PrecompileTools", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "3fbdee81b0cdc2b106b681dd2b9d4bdc60ca35a2"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.6.18"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c5aeb516a84459e0318a02507d2261edad97eb75"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "679e69c611fff422038e9e21e270c4197d49d918"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.12"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "6dc1bbdd6a133adf4aa751d12dbc2c6ae59f873d"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.11.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "aef70bb349b20aa81a82a19704c3ef339d4ee494"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.22.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "9b1ca1aa6ce3f71b3d1840c538a8210a043625eb"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1fb174f0d48fe7d142e1109a10636bc1d14f5ac2"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.17"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "d4e9dc4c6106b8d44e40cd4faf8261a678552c7c"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.12"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "1f3b7b0d321641c1f2e519f7aed77f8e1f6cb133"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.29"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "380053d61bb9064d6aa4a9777413b40429c79901"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.2.0"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "c5c28c245101bd59154f649e19b038d15901b5dc"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.2"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "73d1214fec245096717847c62d389a5d2ac86504"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.22.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "f3cf88025f6d03c194d73f5d13fee9004a108329"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.6"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "055626e1a35f6771fe99060e835b72ca61a52621"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "97829cfda0df99ddaeaafb5b370d6cab87b7013e"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.3"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d4f85701f569584f2cff7ba67a137d03f0cfb7d0"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.3"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "5694b56ccf9d15addedc35e9a4ba9c317721b788"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.10"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "899050ace26649433ef1af25bc17a815b3db52b7"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.9.0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "af13a277efd8a6e716d79ef635d5342ccb75be61"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "RoundingEmulator"]
git-tree-sha1 = "2d6d22fe481eff6e337808cc0880c567d7324f9a"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.8"
weakdeps = ["DiffRules", "ForwardDiff", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "5ea6acdd53a51d897672edb694e3cc2912f3f8a7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.46"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5edc369a598dfde567269dc6add5812cfa10cd5"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "3a994404d3f6709610701c7dabfc03fed87a81f8"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Primes", "SimplePolynomials"]
git-tree-sha1 = "d76cec8007ec123c2b681269d40f94b053473fcf"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.2.7"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "a113a8be4c6d0c64e217b472fb6e61c760eb4022"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.6.3"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase"]
git-tree-sha1 = "ac79beff4257e6e80004d5aee25ffeee79d91263"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.9.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "27af6be179c711fb916a597b6644fbb5b80becc0"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.8"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "248b7a4be0f92b497f7a331aed02c1e9a878f46b"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.3"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "679c1aec6934d322783bd15db4d18f898653be4f"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.27.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "a385fe5aa1384647e55c0c8773457b71e9b08518"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mods]]
git-tree-sha1 = "924f962b524a71eef7a21dae1e6853817f9b658f"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "2.2.4"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MultiScaleTreeGraph]]
deps = ["AbstractTrees", "DataFrames", "Dates", "DelimitedFiles", "Graphs", "MetaGraphsNext", "MutableNamedTuples", "OrderedCollections", "Printf", "SHA", "XLSX"]
git-tree-sha1 = "47286f2af18bdd7458115f514d6bb2c7db8275e4"
uuid = "dd4a991b-8a45-4075-bede-262ee62d5583"
version = "0.12.2"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "302fd161eb1c439e4115b51ae456da4e9984f130"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.1"

[[deps.MutableNamedTuples]]
git-tree-sha1 = "0faaabea6ebbfde9a5a01455f851009fb2603aac"
uuid = "af6c499f-54b4-48cc-bbd2-094bba7533c7"
version = "0.1.3"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "6a731f2b5c03157418a20c12195eb4b74c8f8621"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.13.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "MathOptInterface", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d024bfb56144d947d4fafcd9cb5cafbe3410b133"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4745216e94f71cb768d58330b059c9b76f32cb66"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.14+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "eb3f9df2457819bf0a9019bd93cc451697a0751e"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.20"

[[deps.PikaParser]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "d6ff87de27ff3082131f31a714d25ab6d0a88abf"
uuid = "3bbf5609-3e7b-44cd-8549-7c69f321e792"
version = "0.6.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "a9c7a523d5ed375be3983db190f6a5874ae9286d"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.6"
weakdeps = ["ChainRulesCore", "FFTW", "MakieCore", "MutableArithmetics"]

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9e8fed0505b0c15b4c1295fd59ea47b411c019cf"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.2"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "f65caa24a622f985cc341de81d3f9744435d0d0f"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.6"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "e182b9e5afb194142d4668536345a365ea19363a"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.2"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "7063828369cafa93f3187b3d0159f05582011405"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.17"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableHashTraits]]
deps = ["Compat", "PikaParser", "SHA", "Tables", "TupleTools"]
git-tree-sha1 = "10dc702932fe05a0e09b8e5955f00794ea1e8b12"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "1.1.8"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "5cf6c4583533ee38639f73b880f35fc85f2941e0"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.3"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TranscodingStreams]]
git-tree-sha1 = "54194d92959d8ebaa8e26227dbe3cdefcdcd594f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.3"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TupleTools]]
git-tree-sha1 = "41d61b1c545b06279871ef1a4b5fcb2cac2191cd"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XLSX]]
deps = ["Artifacts", "Dates", "EzXML", "Printf", "Tables", "ZipFile"]
git-tree-sha1 = "319b05e790046f18f12b8eae542546518ef1a88f"
uuid = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"
version = "0.10.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "07e470dabc5a6a4254ffebc29a1b3fc01464e105"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1ea2ebe8ffa31f9c324e8c1d6e86b4165b84a024"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─393b8020-3743-11ec-2da9-d1600147f3d1
# ╠═8d606a5d-4d1f-4754-98f2-80097817c479
# ╟─3506b454-fb9c-4632-8dfb-15804b66add2
# ╟─3d8871e5-1204-46a4-a03f-a2081054d5e7
# ╠═12c90bd3-fb41-4e5d-9752-9e482225f634
# ╟─8b711c1e-7d4e-404b-b2c8-87f536728fee
# ╟─6bee7b4a-c3a1-4562-a17f-71335b8d39ae
# ╟─6b8d93de-2fb4-411d-befe-29cb29132b40
# ╟─796a13d2-a65c-46f6-ad42-5fd42811c8a8
# ╟─220dfbff-15fc-4e75-a6a2-39e60c08e8dc
# ╠═492fc741-7a3b-4992-a453-fcac2bbf35ad
# ╟─068bccf7-7d01-40f5-b06b-97f6f51abcdd
# ╟─0b8d39b2-9255-4bd7-a02f-2cc055bf61fd
# ╟─fa9cf6f4-eb79-4c70-ba1f-4d80b3c3e62a
# ╠═3d0a6b24-f11b-4f4f-b59b-5c40ea9be838
# ╟─b8800a04-dedb-44b3-82fe-385e3db1d0d5
# ╟─68fdcbf2-980a-4d44-b1f9-46b8fcd5bea1
# ╟─bde004a8-d54c-4049-98f6-87c579785641
# ╟─a7bf20e9-211c-4161-a5d2-124866afa76e
# ╠═aaa829ee-ec36-4116-8424-4b40c581c2fc
# ╟─f2eb6a9d-e788-46d0-9957-1bc22a98ad5d
# ╟─b49c4235-a09e-4b8c-a392-d423d7ed7d4c
# ╟─d587f110-86d5-41c0-abc7-2671d711fbdf
# ╟─e2f20d4c-77d9-4b95-b30f-63febb7888c3
# ╟─3944b38d-f45a-4ff9-8348-98a8e04d4ad1
# ╟─dc2bd8f0-c321-407f-9592-7bcdf45f9634
# ╟─19bc88a8-8299-49fe-bf19-60e0fdfbd1ba
# ╟─61f8f651-1dba-4b91-a46e-bbe61cffb725
# ╟─c0ea7917-f0bf-445a-a612-40abd67fe9ad
# ╟─0e160fb4-f2ba-4321-b661-4ce6eb422c6b
# ╟─9c04906b-10cd-4c53-a879-39d168e5bd1f
# ╟─e5c0c40a-eb0a-4726-b58e-59c64cb39eae
# ╟─d66aebf5-3681-420c-a342-166ea05dda2e
# ╟─7de574d4-a8b8-4945-a2f1-5b2928b9d231
# ╟─f26a28b2-d70e-4543-b58e-2d640c2a0c0d
# ╟─9290e9bf-4c43-47c7-96ec-8b44ad3c6b23
# ╟─466aa3b3-4c78-4bb7-944d-5d55128f8cf6
# ╠═87140df4-3fb5-443c-a667-be1f19b016f6
# ╟─b710e330-e783-48f7-bebe-4257e7fec385
# ╟─db44b868-f9b8-466a-b13e-274ad9b9d74a
# ╟─a6ea9696-d778-4033-a7df-76da4ea1f5fe
# ╟─f50a2242-64ee-4c91-8c9d-3d2d3f11ac5d
# ╠═73515bd3-0124-42a4-9997-3730e7dcbf4c
# ╟─1103b80b-7dc6-41ed-b5bd-a6ef739d0624
# ╠═806c5fba-6166-41d9-a109-9fac15eb107a
# ╠═a802bffa-a5c3-4563-adde-7a393b47881c
# ╠═bfe70951-a89d-4783-98a8-ac39fabed66b
# ╟─30f8608f-564e-4ffc-91b2-1f104fb46c1e
# ╟─0a19ac96-a706-479d-91b5-4ea3e091c3e8
# ╟─665cb43b-ab86-4001-88a3-c67ed16b28e8
# ╟─8bb72bea-be21-47c1-bdd3-484690c3cfd4
# ╟─ffe53b41-96bd-4f44-b313-94aabdc8b1a6
# ╟─12d7aca9-4fa8-4461-8077-c79a99864391
# ╟─b6fa2e19-1375-45eb-8f28-32e1d00b5243
# ╟─21fd863d-61ed-497e-ba3c-5f327e354cee
# ╟─45b20f0f-2018-406f-89b6-b8e058909e66
# ╟─0195ac30-b64f-409a-91ad-e68cf37d7c3b
# ╟─77486fa7-318d-4397-a792-70fd8d2148e3
# ╟─97871566-4904-4b40-a631-98f7e837a2f4
# ╟─d7a3c496-0ef0-454b-9e32-e5835928f4d5
# ╟─eb39ed1b-6dee-4738-a762-13b759f74411
# ╟─ee46e359-36bd-49c4-853c-d3ff29888473
# ╟─b2e75112-be43-4df9-86df-2eeeb58f47c3
# ╟─b01851d1-d9d9-4016-b02e-6d3bfc449b8a
# ╟─14fde936-fa95-471a-aafb-5d69871e5a87
# ╟─e3ba9fec-c8b3-46e6-8b1d-29ab19198c9c
# ╟─9e967170-9388-43e4-8b18-baccb18f4b4e
# ╟─979ca113-6a22-4313-a011-0aca3cefdbf7
# ╟─43967391-6580-4aac-9ac1-c9effbf3c948
# ╟─de63abdd-3879-4b7c-86f7-844f6288f987
# ╟─d17d7e96-bd15-4a79-9ccb-6182e7d7c023
# ╟─27a0dcef-260c-4a0c-bef3-04a7d1b79805
# ╟─666e9daf-e28f-4e14-b52a-bcc6b5aadb67
# ╟─073e32dd-c880-479c-8933-d53c9655a04d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
