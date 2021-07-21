# Aim: Analyze the variability of wood density along branches from mtg data.
# Author: M. Millan, J. Dauzat and R. Vezy
# Date of creation: 12/08/2020

library(ggplot2)
library(magrittr)
library(data.table)
library(tidyverse)

# Read the data.frame -----------------------------------------------------

df_mtg = fread("0-data/5-enriched_manual_mtg/enriched_field_mtg.csv", data.table = FALSE)

df_mtg

df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= topological_order, y= density))+
  geom_boxplot(aes(group = topological_order))

df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= segment_index_on_axis, y= density))+
  geom_point(aes(color = branch))

df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= segment_index_on_axis, y= density))+
  facet_wrap(.~branch)+
  geom_point(aes(color = branch))

df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515)%>%
  ggplot(aes(x= diameter, y= density))+
  facet_wrap(.~branch, scales = "free")+
  geom_point(aes(color = segment_index_on_axis))


df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515)%>%
  ggplot(aes(x= volume, y= density))+
  facet_wrap(.~branch, scales = "free")+
  geom_point(aes(color = segment_index_on_axis))

df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515)%>%
  ggplot(aes(x= volume_subtree, y= density))+
  facet_wrap(.~branch, scales = "free")+
  geom_point(aes(color = segment_index_on_axis))


# VOLUME
# volume du ? ?corce
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296)%>%
  ggplot(aes(x= volume_phse, y= volume_ph))+
  # facet_wrap(.~branch)+
  geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = branch))
df_mtg%>%
  dplyr::filter(ID != 1500 & ID != 1515 & ID != 4296 & volume_phse/volume_ph<1.1)%>%
  ggplot(aes(x= diameter, y= (volume_ph-volume_phse)/volume_ph))+
  # facet_wrap(.~branch)+
  # geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = branch))


# Variation de volume avec hydratation
# Different delta de volume suivant branches 
# Peut-etre traitements mais prob avec tree1h: volumes peuvent diminuer apr?s rehydratation
df_mtg%>%
  # dplyr::filter(ID != 1500 & ID != 1515)%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= density, y= volume_delta))+
  geom_point(aes(color = diameter))+
  scale_color_viridis_c()

df_mtg%>%
  # dplyr::filter(ID != 1500 & ID != 1515)%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  # dplyr::filter(branch == "tree1h")%>%
  ggplot(aes(x= density, y= volume_delta))+
  # facet_wrap(.~branch, scales = "free")+
  # geom_point(aes(color = segment_index_on_axis))
  geom_point(aes(color = branch))


df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  # ggplot(aes(x= dry_weight, y= volume_delta))+
  ggplot(aes(x= diameter, y= volume_delta))+
  geom_point(aes(color = factor(branch)))




# density wood only vs density with bark
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  # dplyr::filter(branch != "tree1h")%>%
  ggplot(aes(x= density, y= density_wood_only))+
  geom_point(aes(color = factor(branch), size=3))
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  # dplyr::filter(branch != "tree1h")%>%
  ggplot(aes(x= diameter, y= density_wood_on_tot))+
  geom_point(aes(color = factor(branch), size=3))


# Bois versus ?corce et diam?tre de l'?chantillon
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= dry_weight, y= dry_weight_bark))+
  geom_point(aes(color = factor(branch)))

df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= ratio_bark_wood))+
  geom_point(aes(color = factor(branch)))

df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= ratio_bark_wood, y= volume_delta))+
  geom_point(aes(color = factor(branch)))
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= ratio_bark_wood, y= volume_delta))+
  geom_point(aes(color = diameter))+
  scale_color_viridis_c()

# dry weight wood and bark
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= dry_weight_wood, y= dry_weight_bark))+
  geom_point(aes(color = diameter))+
  scale_color_viridis_c()
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= dry_weight_bark/(dry_weight_wood+dry_weight_bark)))+
  geom_point(aes(color = factor(branch)))
# == ratio poids sec ?corce / poids sec bois
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= dry_weight_bark/dry_weight_wood))+
  geom_point(aes(color = factor(branch)))
# == ratio volume ?corce / volume bois
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= (volume_ph-volume_phse)/volume_phse))+
  geom_point(aes(color = factor(branch)))

# VOLUME_PH de bois et ?corce
# == ratio volumePH bark/total
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= (volume_ph-volume_phse)/volume_ph))+
  geom_point(aes(color = factor(branch)), size=3)+ylim(0,0.6)
# == ratio poidsPH bark/poids total
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= dry_weight_bark/(dry_weight_bark+dry_weight_wood)))+
  geom_point(aes(color = factor(branch)), size=3)+ylim(0,0.6)
# == Comparaison des ratios des contribution de bark sur total pour poids et volume
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= dry_weight_bark/(dry_weight_bark+dry_weight_wood), y= (volume_ph-volume_phse)/volume_ph))+
  # geom_point(aes(shape=factor(branch), color = factor(branch)), size=3)+ylim(0,0.6)
  geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = factor(branch)), size=3)+ylim(0,0.6)+ xlim(0,0.6)
# == Rapport des ratios pour poids et volume
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x=diameter, y=((volume_ph-volume_phse)/volume_ph) / (dry_weight_bark/(dry_weight_bark+dry_weight_wood))))+
  geom_point(aes(color = factor(branch)), size=3) + ylim(0,2)


# == density wood post hydratation
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= dry_weight_wood/volume_phse))+
  geom_point(aes(color = factor(branch)), size=3)+
  ylim(0,0.6)
# == density bark post hydratation
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= dry_weight_bark/(volume_ph-volume_phse)))+
  geom_point(aes(color = factor(branch)), size=3)+
                   ylim(0,0.6)
# rapport des densit? ?corce / bois
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= (dry_weight_bark/(volume_ph-volume_phse)) / (dry_weight_wood/volume_phse)))+
  geom_point(aes(color = factor(branch)), size=3)+
  ylim(0,2)
# rapport volumes massique
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= diameter, y= ((volume_ph-volume_phse)/dry_weight_bark) / (volume_phse/dry_weight_wood)))+
  geom_point(aes(color = factor(branch)), size=3)+
  ylim(0,2)


# == ratio poids vs ratio volume
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= dry_weight_bark/dry_weight_wood, y= (volume_ph-volume_phse)/volume_phse))+
  geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = factor(branch)), size=3) + ylim(0,2)


df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= dry_weight_bark/dry_weight_wood, y= (volume_ph-volume_phse)/volume_phse))+
  geom_point(aes(color = diameter), size=3)+
  ylim(0,2)
  scale_color_viridis_c()




# == test age (year)
df_mtg%>%
  dplyr::filter(year<2018)%>%
  ggplot(aes(x= year, y= diameter))+
  geom_point(aes(color = factor(branch)), size=3) + ylim(0,50)
# == diameter ou cross section vs borne
df_mtg%>%
  dplyr::filter(diameter<100, branch!="tree2h")%>%
  ggplot(aes(x= diameter, y= volume_subtree))+
  geom_point(aes(color = factor(branch)), size=3)
# == cross_section vs volume subtree
df_mtg%>%
  dplyr::filter(diameter<100, branch!="tree2h")%>%
  ggplot(aes(y= cross_section, x= volume_subtree))+
  geom_point(aes(color = factor(branch)), size=3) + ylim(0,30)
# == cross_section vs total pathlength
df_mtg%>%
  dplyr::filter(diameter<100, branch!="tree2h")%>%
  ggplot(aes(y= cross_section, x= pathlength_subtree))+
  geom_point(aes(color = factor(branch)), size=3) + ylim(0,30)

# ============== TEST section vs nbleaves
df_mtg%>%
  dplyr::filter(diameter<100, branch!="tree2h")%>%
  ggplot(aes(y= cross_section, x= number_leaves))+
  geom_point(aes(color = factor(branch)), size=3)


df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= dry_weight_bark, y= volume_ph-volume_phse))+
  geom_point(aes(color = factor(branch)))
df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= dry_weight_wood, y= volume_phse))+
  geom_point(aes(color = factor(branch)))

df_mtg%>%
  dplyr::filter(density < 1.0 & density > 0.2)%>%
  ggplot(aes(x= density_ph_wood, y= density_ph_bark))+
  geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = factor(branch)))


# density_ph wood vs density_ph wood+bark
# density_ph_wood
# df_mtg%>%
#   dplyr::filter(density < 1.0 & density > 0.2)%>%
#   ggplot(aes(x= diameter, y= dry_weight_wood/volume_phse))+
#   geom_point(aes(color = factor(branch)))
# df_mtg%>%
#   dplyr::filter(density < 1.0 & density > 0.2)%>%
#   ggplot(aes(x= density_ph, y= dry_weight_wood/volume_phse))+
#   geom_abline(slope = 1,intercept = 0)+
#   geom_point(aes(color = factor(branch)))



# df_mtg%>%
#   dplyr::filter(density < 1.0 & branch != "tree1h")%>%
#   ggplot(aes(x= density, y= density_ph))+
#   geom_abline(slope = 1,intercept = 0)+
#   # facet_wrap(.~branch, scales = "free")+
#   # geom_point(aes(color = volume))+
#   geom_point(aes(color = volume_delta))+
#   scale_color_viridis_c()
df_mtg%>%
  dplyr::filter(density < 1.0 & branch != "tree1h" & diameter<100)%>%
  ggplot(aes(x= density, y= density_ph))+
  geom_abline(slope = 1,intercept = 0)+
  # facet_wrap(.~branch, scales = "free")+
  # geom_point(aes(color = volume))+
  geom_point(aes(color = factor(branch)))



df_mtg%>%
  # dplyr::filter(density < 1.0)%>%
  # dplyr::filter(density < 1.0 & branch != "tree1h")%>%
  dplyr::filter(density < 1.0 & density_ph < 0.6 & branch != "tree1h")%>%
  # ggplot(aes(x= diameter, y= density_ph/density))+
  ggplot(aes(x= diameter, y= density))+
  # geom_abline(slope = 1,intercept = 0)+
  # facet_wrap(.~branch, scales = "free")+
  geom_point(aes(color = volume))+
  scale_color_viridis_c()


df_mtg%>%
  dplyr::filter(diameter<100 & cross_sec_children<65)%>%
  ggplot(aes(x= cross_section, y= cross_sec_children))+
  geom_abline(slope = 1,intercept = 0)+
  geom_point(aes(color = factor(topological_order)))




# faire diametre fct du nombre de "leaf"




# autoplot(tree1l)
# plotly_mtg(tree1l)
