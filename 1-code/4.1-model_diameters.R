# model 2 -----------------------------------------------------------------

# using a simple model that we correct with a proportionality factor that
# is computed using the measurements that we know are well made (e.g. diameters > 2cm)  

library(caret)
library(data.table)
library(tidyverse)
source("1-code/0-function.R")


df_mtg = fread("0-data/5-enriched_manual_mtg/enriched_field_mtg.csv", data.table = FALSE)

df_mtg$tree = stringr::str_sub(gsub("tree","",df_mtg$branch), start = 1, end = 1)
df_mtg$unique_branch = df_mtg$branch
df_mtg$branch = stringr::str_sub(gsub("tree","",df_mtg$branch), start = 2)

# Adding one to the number of leaves for the terminal leaves (they bear themselves)
df_mtg$number_leaves[df_mtg$number_leaves==0] = 1
df_mtg = 
  df_mtg%>%
  filter(!is.na(cross_section) & cross_section > 0.0)

# The formula used for the general model:
# formula = cross_section ~ number_leaves + pathlength_subtree +
#   segment_index_on_axis + axis_length + segment_subtree
# formula = cross_section ~ number_leaves + pathlength_subtree + segment_index_on_axis
formula = cross_section ~ pathlength_subtree + segment_index_on_axis
min_diam = 20

# # Plotting the relationship between the variables used and the cross_section:
# reshape2::melt(df_mtg%>%select(branch,tree,cross_section_log,tidyselect::all_of(all.vars(formula)[-1])),
#                id.vars = c("tree","branch","cross_section_log"))%>%
#   ggplot(aes(x = cross_section_log, y = value, color = paste(tree,branch)))+
#   # facet_wrap(variable + branch ~ ., scales = "free_y")+
#   facet_grid(rows = vars(variable), cols = vars(tree,branch), scales = "free_y")+
#   geom_point()

# df_mtg$cross_section_log = log(df_mtg$cross_section)

# Fitting the general model, and applying a correction factor based on each branch to it: 
model = fit_model(data = df_mtg, formula = formula, min_diam = min_diam)
caret::varImp(model$model)
summary(model$model)
model$plots
model$statistics

# Getting the resulting plot of applying the model and correction to the full data: 
res = apply_model_cor(data = df_mtg, model = model$model, min_diam = 20)

res$plots$corrected

res$statistics$corrected%>%
  ungroup()%>%
  summarise(across(where(is.numeric),mean, na.rm = TRUE))

res$statistics$corrected_min_diam%>%
  ungroup()%>%
  summarise(across(where(is.numeric),mean, na.rm = TRUE))

df_cross = cross_validate(data = df_mtg, formula = formula, min_diam = min_diam)

df_cross$plots

df_cross$statistics$general_model%>%
  ungroup()%>%
  summarise(across(where(is.numeric),mean, na.rm = TRUE))

df_cross$statistics$corrected%>%
  ungroup()%>%
  summarise(across(where(is.numeric),mean, na.rm = TRUE))

df_cross$statistics$general_model_min_diam%>%
  ungroup()%>%
  summarise(across(where(is.numeric),mean, na.rm = TRUE))

df_cross$statistics$corrected_min_diam%>%
  ungroup()%>%
  summarise(across(where(is.numeric),mean, na.rm = TRUE))


# TODO: Calculer les volumes predis, et comparer aux volumes issus de mesures 

