### A Pluto.jl notebook ###
# v0.18.4

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

# ╔═╡ 8b711c1e-7d4e-404b-b2c8-87f536728fee
md"""
Defining the colors for the plot:
"""

# ╔═╡ 6bee7b4a-c3a1-4562-a17f-71335b8d39ae
colors = ["Stat. mod." => ColorSchemes.Set2_5.colors[1], "Pipe mod." => ColorSchemes.Set2_5.colors[2]]

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
formula_all = @formula(cross_section ~ 0 + cross_section_pipe + pathlength_subtree + branching_order + segment_index_on_axis + axis_length + number_leaves + segment_subtree + n_segments_axis)

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

# ╔═╡ e2f20d4c-77d9-4b95-b30f-63febb7888c3
md"""
*Figure 1. Measured (x-axis) and predicted (y-axis) cross-section at axis scale. The prediction is done either using the statistical model (Stat. mod.), or the pipe model (Pipe mod.).*
"""

# ╔═╡ 3944b38d-f45a-4ff9-8348-98a8e04d4ad1
md"""
*Table 2. Prediction accuracy of the two modelling approach, computed with normalized root mean squared error (nRMSE, %) and modelling efficiency (EF).*
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

# ╔═╡ 665cb43b-ab86-4001-88a3-c67ed16b28e8
function compute_data_mtg_tree!(mtg_tree, fresh_density, dry_density)

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
first_cross_section = π * ((descendants(mtg_tree, :diameter, ignore_nothing = true, recursivity_level=5)[1] / 2.0)^2)
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
	df_density_tree = df_density[replace.(df_density.branches, r"[a-z]" => "") .== tree,:]
	
	# Compute the average per tree:
	dry_density = mean(df_density_tree.dry_density)
	fresh_density = mean(df_density_tree.fresh_density)

	# Import its MTG (computed from plantscan3d):
	mtg_tree = read_mtg(i)

    compute_data_mtg_tree!(mtg_tree, fresh_density, dry_density)

	# Write the computed LiDAR MTG to disk:
	write_mtg(joinpath("../0-data/3-mtg_lidar_plantscan3d/9-tree_scale_segmentized_enriched",basename(i)), mtg_tree)
end

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
    filter!(x -> ismissing(x.comment) || !(x.comment in ["casse", "CASSE", "AVORTE","Portait aussi un axe peut-être cassé lors de la manip"]), x)
    filter!(x -> x.symbol == "S", x)
    # filter!(x -> !in(x.tree, ["2","4"]), x) # Remove pruned trees
    filter!(x -> !in(x.tree, ["1", "2", "3", "4"]), x) # Remove pruned trees
    x
end

# ╔═╡ 68fdcbf2-980a-4d44-b1f9-46b8fcd5bea1
begin
    # Define the function to compute the RMSE for each cross-validation fold:
    function compute_rmse(mod, df_ref)
        x = deepcopy(df_ref)
        x[!, "Stat. mod."] = predict(mod, x)
        x = dropmissing(x, ["Stat. mod.", "cross_section"])
        nRMSE(x[:, "Stat. mod."], x[:, :cross_section])
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
    x[:, "Stat. mod."] = predict(model, x)
    rename!(x, Dict(:cross_section_pipe => "Pipe mod."))
    stack(
        dropmissing(x, ["Pipe mod.", "Stat. mod.", "cross_section"]),
        ["Pipe mod.", "Stat. mod."],
        [:unique_branch, :id, :symbol, :scale, :index, :parent_id, :link, :cross_section],
        variable_name=:origin,
        value_name=:cross_section_pred
    )
end;

# ╔═╡ d587f110-86d5-41c0-abc7-2671d711fbdf
begin
    plt_cs_all =
        data(df_all) *
        (
		   mapping(
                :cross_section => "Measured cross-section (mm²)",
                :cross_section => "Predicted cross-section (mm²)") * visual(Lines) +
            mapping(
                :cross_section => "Measured cross-section (mm²)",
                :cross_section_pred => "Predicted cross-section (mm²)",
                color=:origin => "Model",
                marker=:unique_branch => "Branch") *
            visual(Scatter, markersize = 20, alpha = 0.8)
        )
    p = draw(plt_cs_all, palettes=(; color=colors))
end

# ╔═╡ 806c5fba-6166-41d9-a109-9fac15eb107a
save("../2-results/2-plots/step_3_statistical_model_evaluation.png", p, px_per_unit=3)

# ╔═╡ dc2bd8f0-c321-407f-9592-7bcdf45f9634
begin
    stats_all =
        combine(
            groupby(df_all, [:origin]),
            [:cross_section_pred, :cross_section] => nRMSE => :nRMSE,
            [:cross_section_pred, :cross_section] => EF => :EF,
            [:cross_section_pred, :cross_section] => Bias => :Bias,
            [:cross_section_pred, :cross_section] => RME => :RME
        )
    sort(stats_all, :nRMSE)
end

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
    length_gf_A1 = descendants!(x, vol_col, symbol="S", link=("/", "<"), all=false)
    id_cor_A1 = descendants!(x, :id_cor, symbol="S", link=("/", "<"), all=false)
    sum(length_gf_A1[findfirst(x -> x == id_cor_start, id_cor_A1):end])
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

# ╔═╡ 77486fa7-318d-4397-a792-70fd8d2148e3
function compute_data_mtg_lidar!(mtg, fresh_density, dry_density)

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
function compute_volume_model(branch, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, df_density)

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

    compute_data_mtg_lidar!(mtg_lidar_ps3d_raw, fresh_density, dry_density)

    # Importing the mtg from the LiDAR, and compute the volume using different methods:
    mtg_lidar_model = read_mtg(joinpath(dir_path_lidar, branch * ".xlsx"))

    compute_data_mtg_lidar!(mtg_lidar_model, fresh_density, dry_density)

    return (mtg_manual, mtg_lidar_ps3d_raw, mtg_lidar_model)
end

# ╔═╡ 666e9daf-e28f-4e14-b52a-bcc6b5aadb67
cross_section_stat_mod_all = cross_section_stat_mod

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
        model=["plantscan3d cor.", "plantscan3d raw", "plantscan3d cor.", "plantscan3d raw", "stat. model cor.", "Pipe model cor.", "stat. model raw", "Pipe model raw", "plantscan3d cor.", "plantscan3d raw", "stat. model cor.", "Pipe model cor.", "stat. model raw", "Pipe model raw"],
        measurement=[tot_lenght_manual, tot_lenght_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, tot_vol_manual, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass, true_fresh_biomass],
        prediction=[tot_lenght_lidar, tot_lenght_lidar_raw, tot_vol_lidar, tot_vol_lidar_raw, tot_vol_lidar_stat_mod, tot_vol_lidar_pipe_mod, tot_vol_lidar_stat_mod_raw, tot_vol_lidar_pipe_mod_raw, fresh_biomass_lidar, fresh_biomass_lidar_raw, fresh_biomass_lidar_stat_mod, fresh_biomass_lidar_pipe_mod, fresh_biomass_lidar_stat_mod_raw, fresh_biomass_lidar_pipe_mod_raw]
    )
end

# ╔═╡ 0a19ac96-a706-479d-91b5-4ea3e091c3e8
function summarize_data(mtg_files, dir_path_lidar, dir_path_lidar_raw, dir_path_manual,dir_path_lidar_new)
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
            compute_volume_model(i, dir_path_lidar, dir_path_lidar_raw, dir_path_manual, df_density)

		# Write the computed LiDAR MTG to disk:
		write_mtg(joinpath(dir_path_lidar_new,i*".mtg"), mtg_lidar_model)
		
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

        # statistical model:
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
            :fresh_mass => "stat. mod.",
            :fresh_mass_50 => "stat. mod. ⌀<50",
            :fresh_mass_ps3d => "plantscan3d",
            :fresh_mass_pipe_mod => "Pipe model",
            :fresh_mass_pipe_mod_50 => "Pipe mod. ⌀<50"
        )

        df_stat_mod_biomass = stack(
            df_stat_mod_biomass,
            ["stat. mod.", "stat. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor],
            variable_name=:origin,
            value_name=:fresh_mass
        )


        df_stat_mod_cs = select(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :id_cor,
            :cross_section_stat_mod => "stat. mod.",
            :cross_section_stat_mod_50 => "stat. mod. ⌀<50",
            :cross_section_ps3d => "plantscan3d",
            :cross_section_pipe => "Pipe model",
            :cross_section_pipe_50 => "Pipe mod. ⌀<50"
        )

        df_stat_mod_cs = stack(
            df_stat_mod_cs,
            ["stat. mod.", "stat. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
            [:id, :symbol, :scale, :index, :parent_id, :link, :id_cor],
            variable_name=:origin,
            value_name=:cross_section
        )

        select!(
            df_stat_mod,
            :id, :symbol, :scale, :index, :parent_id, :link, :length_sim => :length, :id_cor,
            :volume_stat_mod => "stat. mod.",
            :volume_stat_mod_50 => "stat. mod. ⌀<50",
            :volume_ps3d => "plantscan3d",
            :volume_pipe_mod => "Pipe model",
            :volume_pipe_mod_50 => "Pipe mod. ⌀<50"
        )

        df_stat_mod = stack(
            df_stat_mod,
            ["stat. mod.", "stat. mod. ⌀<50", "plantscan3d", "Pipe model", "Pipe mod. ⌀<50"],
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
df_all_branches, df_stats_branch = summarize_data(mtg_files, dir_path_lidar, dir_path_lidar_raw, dir_path_manual,dir_path_lidar_new);

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
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
MultiScaleTreeGraph = "dd4a991b-8a45-4075-bede-262ee62d5583"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AlgebraOfGraphics = "~0.6.5"
CSV = "~0.10.3"
CairoMakie = "~0.7.5"
ColorSchemes = "~3.17.1"
DataFrames = "~1.3.2"
GLM = "~1.6.1"
MLBase = "~0.9.0"
MultiScaleTreeGraph = "~0.5.0"
StatsBase = "~0.33.16"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[AlgebraOfGraphics]]
deps = ["Colors", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "032144cbb772cf0aef2954dfe5cc2c0bebeaaadd"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.6.5"

[[Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "d49f55ff9c7ee06930b0f65b1df2bfa811418475"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "4.0.4"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "9310d9495c1eb2e4fa1955dd478660e2ecab1fbb"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.3"

[[Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "StaticArrays"]
git-tree-sha1 = "4a0de4f5aa2d5d27a1efa293aeabb1a081e46b2b"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.7.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Dictionaries]]
deps = ["Indexing", "Random"]
git-tree-sha1 = "7e73a524c6c282e341de2b046e481abedbabd073"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.19"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c43e992f186abaf9965cc45e372f4693b7754b22"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.52"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "90b158083179a6ccbce2c7eb1446d5bf9d7ae571"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.7"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "d7ab55febfd0907b285fbf8dc0c73c0825d9d6aa"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.3.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "0fa3b52a04a4e210aeb1626def9c90df3ae65268"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.1.0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "StaticArrays"]
git-tree-sha1 = "770050893e7bc8a34915b4b9298604a3236de834"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "fb764dacfa30f948d52a6a4269ae293a479bbc62"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.6.1"

[[GeoInterface]]
deps = ["RecipesBase"]
git-tree-sha1 = "6b1a29c757f56e0ae01a35918a2c39260e2c4b98"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "0.5.7"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "169c3dc5acae08835a573a8a3e25c62f689f8b5c"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.6.5"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageIO]]
deps = ["FileIO", "JpegTurbo", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "464bdef044df52e6436f8c018bea2d48c40bb27b"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.1"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "81b9477b49402b47fbe7f7ae0b252077f53e4a08"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.22"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase"]
git-tree-sha1 = "3bd9fd4baf19dfc1edf344bc578da7f565da2e18"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.9.0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "63de3b8a5c1f764e4e3a036c7752a632b4f0b8d1"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.16.6"

[[MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "c5fb1bfac781db766f9e4aef96adc19a729bc9b2"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.2.1"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "70e733037bbf02d691e78f95171a1fa08cdc6332"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.2.1"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[MetaGraphsNext]]
deps = ["Graphs", "JLD2"]
git-tree-sha1 = "f8e0351036278130f6bf966cb903e5fddb93778c"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.2.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultiScaleTreeGraph]]
deps = ["AbstractTrees", "DataFrames", "DelimitedFiles", "Graphs", "MetaGraphsNext", "MutableNamedTuples", "OrderedCollections", "Printf", "SHA", "XLSX"]
git-tree-sha1 = "d0dc30c70217c03211d690a6e16f77552aa58fb0"
uuid = "dd4a991b-8a45-4075-bede-262ee62d5583"
version = "0.5.0"

[[MutableNamedTuples]]
git-tree-sha1 = "f84525e443ce35292f4c3bc7fa9642f90c6991ba"
uuid = "af6c499f-54b4-48cc-bbd2-094bba7533c7"
version = "0.1.0"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "eb4dbb8139f6125471aa3da98fb70f02dc58e49c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.14"

[[Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a121dfbba67c94a5bec9dde613c3d0cbcf3a12b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.3+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMD]]
git-tree-sha1 = "7dbc15af7ed5f751a82bf3ed37757adf76c32402"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.1"

[[ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "9cc2955f2a254b18be655a4ee70bc4031b2b189e"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "65068e4b4d10f3c31aaae2e6cb92b6c6cedca610"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.5.6"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6976fab022fea2ffea3d945159317556e5dad87c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.2"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "03c99c7ef267c8526953cafe3c4239656693b8ab"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.29"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "aaa19086bc282630d82f818456bc40b4d314307d"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.4"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XLSX]]
deps = ["Dates", "EzXML", "Printf", "Tables", "ZipFile"]
git-tree-sha1 = "2af4b3e329b51f1a41acb346e64156f904860a74"
uuid = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"
version = "0.7.9"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─393b8020-3743-11ec-2da9-d1600147f3d1
# ╠═8d606a5d-4d1f-4754-98f2-80097817c479
# ╟─3506b454-fb9c-4632-8dfb-15804b66add2
# ╟─8b711c1e-7d4e-404b-b2c8-87f536728fee
# ╟─6bee7b4a-c3a1-4562-a17f-71335b8d39ae
# ╟─6b8d93de-2fb4-411d-befe-29cb29132b40
# ╟─796a13d2-a65c-46f6-ad42-5fd42811c8a8
# ╟─220dfbff-15fc-4e75-a6a2-39e60c08e8dc
# ╟─492fc741-7a3b-4992-a453-fcac2bbf35ad
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
# ╠═d587f110-86d5-41c0-abc7-2671d711fbdf
# ╟─e2f20d4c-77d9-4b95-b30f-63febb7888c3
# ╟─3944b38d-f45a-4ff9-8348-98a8e04d4ad1
# ╟─dc2bd8f0-c321-407f-9592-7bcdf45f9634
# ╟─9c04906b-10cd-4c53-a879-39d168e5bd1f
# ╟─e5c0c40a-eb0a-4726-b58e-59c64cb39eae
# ╟─d66aebf5-3681-420c-a342-166ea05dda2e
# ╟─7de574d4-a8b8-4945-a2f1-5b2928b9d231
# ╟─f26a28b2-d70e-4543-b58e-2d640c2a0c0d
# ╟─9290e9bf-4c43-47c7-96ec-8b44ad3c6b23
# ╟─466aa3b3-4c78-4bb7-944d-5d55128f8cf6
# ╠═87140df4-3fb5-443c-a667-be1f19b016f6
# ╟─b710e330-e783-48f7-bebe-4257e7fec385
# ╠═db44b868-f9b8-466a-b13e-274ad9b9d74a
# ╠═a6ea9696-d778-4033-a7df-76da4ea1f5fe
# ╟─665cb43b-ab86-4001-88a3-c67ed16b28e8
# ╟─0a19ac96-a706-479d-91b5-4ea3e091c3e8
# ╟─f50a2242-64ee-4c91-8c9d-3d2d3f11ac5d
# ╠═73515bd3-0124-42a4-9997-3730e7dcbf4c
# ╟─1103b80b-7dc6-41ed-b5bd-a6ef739d0624
# ╠═806c5fba-6166-41d9-a109-9fac15eb107a
# ╟─30f8608f-564e-4ffc-91b2-1f104fb46c1e
# ╟─ffe53b41-96bd-4f44-b313-94aabdc8b1a6
# ╟─12d7aca9-4fa8-4461-8077-c79a99864391
# ╟─b6fa2e19-1375-45eb-8f28-32e1d00b5243
# ╟─21fd863d-61ed-497e-ba3c-5f327e354cee
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
