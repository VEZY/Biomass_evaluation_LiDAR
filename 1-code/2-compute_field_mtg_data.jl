using MTG
using Statistics
using MTG:parent

branch = "E:/Agrobranche_Alexis_Bonnet/Biomass_evaluation_LiDAR/0-data/2-mtg/tree3h.mtg"
mtg = read_mtg(branch)

compute_data_mtg(mtg)

# write the resulting mtg to disk:
write_mtg("E:/Agrobranche_Alexis_Bonnet/Biomass_evaluation_LiDAR/0-data/5-enriched_manual_mtg/tree3h.mtg", mtg)

# And the resulting DataFrame to a csv file:
df =
    DataFrame(
        mtg,
        [
            :density, :length, :diameter, :axis_length, :topological_order,
            :segment_index_on_axis, :mass_g, :volume, :volume_subtree, :cross_section,
            :cross_section_children, :cross_section_leaves, :volume_subtree,
            :number_leaves, :pathlength_subtree, :segment_subtree
        ])

CSV.write("E:/Agrobranche_Alexis_Bonnet/Biomass_evaluation_LiDAR/0-data/5-enriched_manual_mtg/tree3h.csv",df)
