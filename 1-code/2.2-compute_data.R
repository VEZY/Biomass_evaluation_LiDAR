# Aim: Compute new variables in the MTG and export the results in a data.frame.
# Author: A. Bonnet & M. Millan and R. Vezy
# Date of creation: 12/07/2021

# Imports -----------------------------------------------------------------

# devtools::install_github("VEZY/XploRer")
library(XploRer)
library(ggplot2)
library(data.table)
source("1-code/0-function.R")

# mtg = read_mtg(file = "0-data/1-mtg/tree1h.mtg")

mtg_files = list.files("0-data/2-mtg",pattern = "mtg$",
                       full.names = TRUE)

# Initialize an empty list of same length than mtg_files:
mtgs = vector(mode = "list", length = length(mtg_files))

# Read each mtg and put the result in the list by index:
for (i in seq_along(mtg_files)) {
  mtgs[[i]] = read_mtg(file = mtg_files[i])
}

# The for loop does:
# mtgs[[1]] = read_mtg(file = mtg_files[1])
# mtgs[[2]] = read_mtg(file = mtg_files[2])
# mtgs[[3]] = read_mtg(file = mtg_files[3])
# mtgs[[...]] = read_mtg(file = mtg_files[...])
# mtgs[[12]] = read_mtg(file = mtg_files[12])

# Name each element in the list as the name of the mtg file:
names(mtgs) = basename(mtg_files)%>%gsub(".mtg","",.)

# Plot
# autoplot(mtgs$tree1l)
# plotly_mtg(mtgs$tree1h)


# Adding new variables ----------------------------------------------------

# We will use as factors: diameter, length, section index on axis (from 1 at the tip
# until max at the base), the axis length, the segment volume, and the total volume
# of the sub-tree supported by the segment.

df_mtg = lapply(mtgs, compute_data_mtg)

df_mtg = dplyr::bind_rows(df_mtg, .id = "branch")


# Write the results -------------------------------------------------------

fwrite(df_mtg,"0-data/5-enriched_manual_mtg/enriched_field_mtg.csv")


