using MTG

branch = "0-data/4-mtg-lidar-segments/mtg-segments-sans-corec/A1BL-s-sans-corec.mtg"
mtg = read_mtg(branch)

@mutate_mtg!(mtg,volume = [:radius] * 3.1415 * [:length], scale = 2) # length is in meters

mtg.MTG.symbol = replace(basename(branch), ".mtg" => "")

write_mtg("0-data/4-mtg-lidar-segments/mtg-segments-sans-corec/A1BL-s-sans-corec-essaie.mtg", mtg)
