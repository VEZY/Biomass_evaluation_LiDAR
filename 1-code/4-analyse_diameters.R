# 
library(ggplot2)
library(magrittr)
library(data.table)
library(tidyverse)
library(tidymodels)
library(patchwork)

source("1-code/0-function.R")

df_mtg = fread("0-data/5-enriched_manual_mtg/enriched_field_mtg.csv", data.table = FALSE)
df_mtg$tree = stringr::str_sub(gsub("tree","",df_mtg$branch), start = 1, end = 1)
df_mtg$branch = stringr::str_sub(gsub("tree","",df_mtg$branch), start = 2)

# df_mtg$cross_section[df_mtg$cross_section > 80 & !is.na(df_mtg$cross_section)] = 
#   df_mtg$cross_section[df_mtg$cross_section > 80 & !is.na(df_mtg$cross_section)] / 100


df_mtg%>%
  ggplot(aes(x = cross_section, y = cross_sec_children))+
  geom_point(aes(color = paste0(tree,", ",branch)))+
  geom_abline(slope = 1, intercept = 0, lty = 2, color = "grey60")+
  geom_smooth(method='lm', formula= y~x)+
  # stat_cor(label.y = 80, digits = 3) +
  # stat_regline_equation(label.y = 75) +
  labs(colour = "Tree, branch", y = "Cross section children (cm2)",
       x = "Cross section segment (cm2)")

CroPlotR::Bias(sim = df_mtg$cross_sec_children, obs = df_mtg$cross_section)

df_mtg%>%
  ggplot(aes(x = cross_section, y = cross_sec_leaves))+
  geom_point(aes(color = paste0(tree,", ",branch)))+
  geom_abline(slope = 1, intercept = 0, lty = 2, color = "grey60")+
  geom_smooth(method='lm', formula= y~x, aes(color = paste0(tree,", ",branch)))+
  stat_cor(label.y = 80, digits = 3) +
  stat_regline_equation(label.y = 75) +
  labs(colour = "Tree, branch", y = "Cross section leaves (cm2)",
       x = "Cross section segment (cm2)")
CroPlotR::Bias(sim = df_mtg$cross_sec_children, obs = df_mtg$cross_section)

df_mtg%>%
  ggplot(aes(x = cross_section, y = cross_sec_leaves))+
  geom_point(aes(color = paste0(tree,", ",branch)))+
  geom_abline(slope = 1, intercept = 0, lty = 2, color = "grey60")+
  geom_smooth(method='lm', formula= y~x, aes(color = paste0(tree,", ",branch)))+
  facet_wrap(tree ~ ., scales = "free")+
  labs(colour = "Tree, branch", y = "Cross section leaves (cm2)",
       x = "Cross section segment (cm2)")

df_mtg%>%
  ggplot(aes(x = cross_section, y = number_leaves))+
  geom_point(aes(color = paste0(tree,", ",branch)))+
  geom_abline(slope = 1, intercept = 0, lty = 2, color = "grey60")+
  geom_smooth(method='lm', formula= y~x, aes(color = paste0(tree,", ",branch)))+
  facet_wrap(tree ~ ., scales = "free")+
  labs(colour = "Tree, branch", y = "Number of leaves",
       x = "Cross section segment (cm2)")


# NUMBER of LEAVES
# toutes les branches
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= cross_section, y= number_leaves))+
  geom_point(aes(color = branch), size=3)+xlim(0,100)


df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= cross_section, y= number_leaves/6.3))+
  geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = branch), size=3)+xlim(0,100)


df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= cross_section, y= (number_leaves+1)/6.3))+
  geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = branch), size=3)+xlim(0,100)

# arbre par arbre
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree1")))%>%
  ggplot(aes(x= cross_section, y= number_leaves))+
  geom_point(aes(color = branch), size=3)
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree2")))%>%
  ggplot(aes(x= cross_section, y= number_leaves))+
  geom_point(aes(color = branch), size=3)+xlim(0,25)
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree3")))%>%
  ggplot(aes(x= cross_section, y= number_leaves))+
  geom_point(aes(color = branch), size=3)
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree4")))%>%
  ggplot(aes(x= cross_section, y= number_leaves))+
  geom_point(aes(color = branch), size=3)+xlim(0,15)

df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree4m")))%>%
  ggplot(aes(x= cross_section, y= number_leaves))+
  geom_point(aes(color = branch), size=3) + xlim(0,13) + ylim(0,13)
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree4m")))%>%
  ggplot(aes(x= cross_section, y= number_leaves+1))+
  geom_point(aes(color = branch), size=3) + xlim(0,13) + ylim(0,13)

# DIAMETRE vs PATHLENGTH
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= cross_section, y= pathlength_subtree))+
  geom_point(aes(color = branch), size=3)+xlim(0,30)
# arbre par arbre
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree1")))%>%
  ggplot(aes(x= cross_section, y= pathlength_subtree))+
  geom_point(aes(color = branch), size=3)+xlim(0,25)
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree2")))%>%
  ggplot(aes(x= cross_section, y= pathlength_subtree))+
  geom_point(aes(color = branch), size=3)+xlim(0,30)
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree3")))%>%
  ggplot(aes(x= cross_section, y= pathlength_subtree))+
  geom_point(aes(color = branch), size=3)+xlim(0,30)
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & (str_detect(branch,"tree4")))%>%
  ggplot(aes(x= cross_section, y= pathlength_subtree))+
  geom_point(aes(color = branch), size=3)+xlim(0,15)


diam_terminal_segments = 
  df_mtg%>%
  filter(topological_order == 1)


ggplot(diam_terminal_segments, aes(x= branch, y = diameter))+
  geom_violin(aes(fill = branch))+
  geom_point()

ggplot(diam_terminal_segments, aes(x= branch, y = diameter))+
  geom_boxplot(aes(fill = branch))+
  geom_point()


# Average segment diameter at extremity -----------------------------------

df_mtg%>%
  filter(.data$segment_index_on_axis == 1)%>%
  ggplot(aes(y = diameter, x = branch))+
  geom_boxplot()+
  labs(y = "Segment diameter (mm)", x = "Branch")
  # geom_point(alpha = 0.1)+
  # geom_boxplot(alpha = 0.8,  outlier.size = 0.0, outlier.alpha = 0.0)

df_mtg%>%
  filter(.data$segment_index_on_axis == 1)%>%
  group_by(branch)%>%
  summarise(diam = mean(diameter, na.rm = TRUE))

df_mtg%>%
  filter(.data$segment_index_on_axis == 1)%>%
  summarise(diam = mean(diameter, na.rm = TRUE))
# OK, we'll use 5.7 cm for the average diameter at extremity.
