


#' xlsx mtg into mtg file
#'
#' Convert xlsx mtg into mtg file
#'
#' @param xlsx_file Path to the xlsx file
#' @param mtg_file  Path to the mtg file
#'
#' @return
#' @export
#'
#' @examples
xlsx_to_mtg = function(xlsx_file, mtg_file){
  mtg = readxl::read_excel(xlsx_file,sheet = 1,col_names = FALSE)
  data.table::fwrite(mtg, file = mtg_file, sep = "\t",col.names = FALSE)
}


# Apply xlsx_to_mtg to all files in directories
all_xlsx_to_mtg= function(xlsx_dir, mtg_dir){
  xlsx_paths = list.files(xlsx_dir,full.names = TRUE,pattern = "xls.")
  
  mtg_names = basename(xlsx_paths)
  mtg_names = gsub(pattern = "xls.", replacement = "mtg",mtg_names)
  mtg_paths = file.path(mtg_dir,mtg_names)
  
  for(i in seq_along(xlsx_paths)){
    xlsx_to_mtg(xlsx_file = xlsx_paths[i], mtg_file = mtg_paths[i])
  }
}





#' Add variables to mtg
#' 
#' Compute meaningful variables to add to the mtg 
#'
#' @param mtg An MTG
#'
#' @return A data.frame with the data
#' @export
#'
compute_data_mtg = function(mtg){
  # Compute the wood density:
  
  if("length" %in% attr(mtg,"features")$NAME){
    # Files from M. Millan (2020) have length in cm, we need it in mm 
    mutate_mtg(mtg, length = node$length / 10, .symbol = "S")
  }else if("length_mm" %in% attr(mtg,"features")$NAME){
    # Files from A. Bonnet (2021) have length in mm in the length_mm column
    mutate_mtg(mtg, length = node$length_mm,.symbol = "S")
  }
  
    
  if("dry_weight_p1" %in% attr(mtg,"features")$NAME){
    mutate_mtg(mtg, dry_weight = node$dry_weight_p1,.symbol = "S")
  }else if("dry_weight_p2" %in% attr(mtg,"features")$NAME){
    mutate_mtg(mtg, dry_weight = node$dry_weight_p2,.symbol = "S")
  }
  
  if("fresh_density" %in% attr(mtg,"features")$NAME){
    mutate_mtg(mtg, density = node$fresh_density, .symbol = "S")
  }
  
  mutate_mtg(mtg, 
             density = node$dry_weight/node$volume_bh,
             density_ph = node$dry_weight/node$volume_ph,
             density_ph_wood = node$dry_weight_wood/node$volume_phse,
             density_ph_bark = node$dry_weight_bark/(node$volume_ph-node$volume_phse),
             .symbol = "S")
  
  # Total pathlength of sub-tree, in mm:
  mutate_mtg(mtg, pathlength_subtree = sum(descendants(attribute = "length", symbol = "S",
                                                       self = TRUE)), .symbol = "S")
  
  # Number of segments each segment bears:
  mutate_mtg(mtg, segment_subtree = length(descendants(attribute = "length", symbol = "S",
                                                       self = TRUE)), .symbol = "S")
  
  # Number of leaves (terminal nodes) a node bear (i.e. all for the first, 0 for a terminal node)
  mutate_mtg(mtg, number_leaves = length(leaves(attribute = "topological_order", symbol = "S")), .symbol = "S")
  
  # density of wood without bark taking account for wood dry weight without bark
  # (Note: without considering the volume of bark)
  # mutate_mtg(mtg, density_wood_only = node$dry_weight_wood/node$volume_bh,
  #            .symbol = "S")
  # 
  # mutate_mtg(mtg, density_wood_on_tot = node$density_wood_only/node$density,
  #            .symbol = "S")
  
  
  
  # add relative change of volume after rehydrating
  mutate_mtg(mtg, volume_delta = (node$volume_ph-node$volume_bh)/node$volume_bh,
             .symbol = "S")
  
  
  # Topological order:
  topological_order(mtg,ascend = FALSE)
  # We use basipetal topological order (from tip to base) to allow comparisons between branches of 
  # different ages (the last emitted segment will always be of order 1).
  
  # Compute the index of each segment on the axis in a basipetal way (from tip to base)
  mutate_mtg(mtg, 
             segment_index_on_axis = length(descendants(attribute = ".symbol", symbol = "S",
                                                        link = c("/", "<"), 
                                                        continue = FALSE))+1,
             .symbol = "S")
  
  # Compute the total length of the axis in mm:
  mutate_mtg(mtg, 
             axis_length = sum(decompose(attribute = "length", symbol = "S")),
             .symbol = "A")
  # Associate the axis length to each segment:
  mutate_mtg(mtg, axis_length = parent(attribute = "axis_length", symbol = "A"), .symbol = "S")
  
  # New branches (>10, e.g. tree12l) diameters are measured twice on the same point at mid-segment, or even 
  # at two points (30% and 70% of the segment length) when the segment length is > 30 cm
  mutate_mtg(mtg, diameter = ifelse(is.na(node$diameter),
                                    mean(c(node$diameter_50_1,
                                           node$diameter_50_2,
                                           node$diameter_70_1,
                                           node$diameter_70_2), na.rm = TRUE),
                                    node$diameter/10), .symbol = "S") # diameter of the segment in mm
  
  mutate_mtg(mtg, volume = pi*((node$diameter/2)^2)*node$length, .symbol = "S") # volume of the segment in mm3
  
  # added JD
  mutate_mtg(mtg, cross_section = pi*((node$diameter/2)^2), .symbol = "S") # area of segment cross section in mm2
  mutate_mtg(mtg, cross_sec_children = sum(children(attribute = "cross_section", symbol = "S")), .symbol = "S")
  
  # Cross section of the terminal nodes for each node
  mutate_mtg(mtg, cross_sec_leaves = sum(leaves(attribute = "cross_section", symbol = "S"),
                                         na.rm = TRUE),
             .symbol = "S")
  # TODO 
  # somme des sections des UC terminales; a plotter vs section du porteur
  # Puis tester avec une valeur unique pour toutes les UC
  
  
  # Volume of wood the section bears (all the sub-tree):
  mutate_mtg(mtg, volume_subtree = sum(descendants(attribute = "volume", symbol = "S",
                                                   self = TRUE)), .symbol = "S")
  
  # segment diameter / axis length:
  mutate_mtg(mtg, d_seg_len_ax_ratio = node$diameter / node$axis_length, .symbol = "S")
  
  
  # ratio weight bark/wood
  mutate_mtg(mtg, ratio_bark_wood = node$dry_weight_bark / node$dry_weight, .symbol = "S")
  
  
  # data.tree::ToDataFrameTree(mtg$MTG,"ID","density","density_ph","diameter","length","axis_length",
  #                            "topological_order","segment_index_on_axis","dry_weight",
  #                            "volume","volume_subtree")
  data.tree::ToDataFrameTree(mtg,"ID","year","density","density_ph","volume_ph","volume_phse","volume_delta",
                             "volume_bh","diameter","length","axis_length","topological_order",
                             "segment_index_on_axis","fresh_weight","dry_weight","dry_weight_bark",
                             "dry_weight_wood","ratio_bark_wood", "volume","volume_subtree",
                             "cross_section","cross_sec_children","cross_sec_leaves","number_leaves",
                             "pathlength_subtree","density_ph_wood","segment_subtree","mass_g")
}


compute_diam = function(diameter_50_1,diameter_50_2,diameter_70_1,diameter_70_2){
  
  if(!is.na(diameter_50_2)){
    # If there's a value for diameter_50_2, we measured with the caliper
    if(!is.na(diameter_70_1)){
      return(mean(c(diameter_50_1,diameter_50_2,diameter_70_1,diameter_70_2)))
    }else{
      return(mean(c(diameter_50_1,diameter_50_2)))
    }
  }else{
    if(!is.na(diameter_70_1)){
      return(mean(c(diameter_50_1,diameter_70_1)))
    }else{
      return(diameter_50_1)
    }
  }
}

#' Fit a model
#' 
#' Fit a model and return the plots, stats and data (predictions, observations) 
#'
#' @param data A data.frame
#' @param formula The model
#' @param min_diam Minimal diameter at which the LiDAR can measure the diameters right 
#' @return A list with the plots, stats and data.
#' @export
#'
#' @examples
#' df_mtg = data.table::fread("2-results/data.csv", data.table = FALSE)
#' formula = cross_section ~ number_leaves + pathlength_subtree + segment_index_on_axis + axis_length
#' fit_model(data = df_mtg, formula = formula, min_diam = 20)
fit_model = function(data,formula,min_diam = 20){
  
  # Complete model (use all possible variables available from LiDAR):
  model = lm(formula, data = data)
  vars_in_model = all.vars(formula)
  
  data_no_na = 
    data%>%
    select(tidyselect::all_of(c("branch","tree","diameter",vars_in_model)))%>%
    filter_all(all_vars(!is.na(.)))%>%
    mutate(prediction = predict(model, newdata = .))
  
  # Our simple model, without any correction:
  simple_model_p = 
    data_no_na%>%
    ggplot(aes(x= !!sym(vars_in_model[1]), y = prediction, color = paste0(tree,", ",branch)))+
    geom_point()+
    geom_abline(slope = 1, intercept = 0)
  
  # Statistics:
  stats_simple_model = 
    data_no_na%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = prediction, obs = !!sym(vars_in_model[1])),
              EF = CroPlotR::EF(sim = prediction, obs = !!sym(vars_in_model[1])),
              Bias = CroPlotR::Bias(sim = prediction, obs = !!sym(vars_in_model[1])),
              .groups = "keep")
  
  stats_simple_model_min_diam = 
    data_no_na%>%
    filter(diameter < min_diam)%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = prediction, obs = !!sym(vars_in_model[1])),
              EF = CroPlotR::EF(sim = prediction, obs = !!sym(vars_in_model[1])),
              Bias = CroPlotR::Bias(sim = prediction, obs = !!sym(vars_in_model[1])),
              .groups = "keep")
  
  
  list(
    model = model,
    plots = simple_model_p,
    statistics = list(
      simple_model = stats_simple_model,
      simple_model_min_diam = stats_simple_model_min_diam),
    data = data_no_na
    # NB: data is full data, with a variable called split to get the data used for training and the data 
    # used for testing. pred_cross_section is the cross section predicted by the general model,
    # and pred_cross_section_cor is the cross section predicted after correction.
  )
}



#' Apply the model with correction
#'
#' @param data A data.frame
#' @param model The model fitted on data
#' @param min_diam Minimal diameter at which the LiDAR can measure the diameters right 
#' @return A list with the plots, stats and data.
#' @export
#'
apply_model_cor = function(data,model,min_diam = 20){
  
  vars_in_model = all.vars(model$terms)
  
  data_no_na = 
    data%>%
    select(tidyselect::all_of(c("branch","tree","diameter","length","cross_section",vars_in_model)))%>%
    filter_all(all_vars(!is.na(.)))%>%
    mutate(pred_cross_section = predict(model, newdata = .))
  
  # Fitting a correction factor (alpha) for each branch on "measured" points
  # i.e. the ones that have a diameter above min_diam:
  formula = as.formula(paste(vars_in_model[1], "~ 0 + pred_cross_section"))
  
  cor_model = ~lm(formula, data = .x)
  
  data_no_na_cor = 
    data_no_na%>%
    mutate(split = ifelse(.data$diameter >= min_diam, "train","test"))%>%
    nest(data = c(-branch,-tree))%>%
    mutate(
      train_data = map(data, function(x) x[x$split=="train",]),
      fit = map(train_data, possibly(~lm(formula, data = .x),otherwise = NA)),
      # test_data = map(data, function(x) x[x$split=="test",]),
      # cross_section_train = map(train_data, function(x) x$cross_section),
      # cross_section_test = map(test_data, function(x) x$cross_section),
      # pred_cross_section_train = map(fit, possibly(~ predict(.x), otherwise = NA)),
      # pred_cross_section_test = map2(fit,test_data, possibly(~ predict.lm(object = .x, newdata = .y), otherwise = NA)),
      diameter = map(data, function(x) x$diameter),
      length = map(data, function(x) x$length),
      cross_section = map(data, function(x) x$cross_section),
      cross_section_log = map(data, function(x) log(x$cross_section)),
      pred_cross_section = map(data, function(x) x$pred_cross_section),
      # pred_cross_section_cor = map(fit, possibly(~ predict(.x), otherwise = NA)),
      pred_cross_section_cor = map2(fit,data, possibly(~ predict.lm(object = .x, newdata = .y), otherwise = NA)),
      intercept = map(fit, possibly(~coef(.x)[1],otherwise = NA)),
      slope = map(fit, possibly(~coef(.x)[2],otherwise = NA)),
      # tidied = map(fit, tidy)
    )%>% 
    unnest(c(intercept,slope,diameter,length,cross_section,cross_section_log,
             pred_cross_section,pred_cross_section_cor))%>% 
    select(-data,-fit) #%>%
  # mutate(diameter_pred = sqrt(pred_cross_section_cor/pi)*10*2)
  
  stats_corrected = 
    data_no_na_cor%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = pred_cross_section_cor, obs = !!sym(vars_in_model[1])),
              EF = CroPlotR::EF(sim = pred_cross_section_cor, obs = !!sym(vars_in_model[1])),
              Bias = CroPlotR::Bias(sim = pred_cross_section_cor, obs = !!sym(vars_in_model[1])),
              .groups = "keep")
  
  stats_corrected_min_diam = 
    data_no_na_cor%>%
    filter(diameter < min_diam)%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = pred_cross_section_cor, obs = !!sym(vars_in_model[1])),
              EF = CroPlotR::EF(sim = pred_cross_section_cor, obs = !!sym(vars_in_model[1])),
              Bias = CroPlotR::Bias(sim = pred_cross_section_cor, obs = !!sym(vars_in_model[1])),
              .groups = "keep")
  
  
  # Plot to see what is the effect of the correction on each branch,
  # and what are the points used to train the correction:
  
  correction_comparison = 
    data_no_na_cor%>%
    mutate(Point = ifelse(diameter >= min_diam, paste("General model, d>",min_diam,"mm"),
                          paste("General model, d<",min_diam,"mm")),
           branch_name = paste0("Tree ",tree,", branch ", branch))%>%
    # filter(branch == "tree2h")%>%
    ggplot(aes(x= !!sym(vars_in_model[1]), color = Point))+
    facet_wrap(branch_name ~ ., scales = "free")+
    geom_point(aes(y = pred_cross_section))+
    geom_point(aes(y = pred_cross_section_cor, color = "Prediction with correction"))+
    geom_abline(slope = 1, intercept = 0)
  
  corrected = 
    data_no_na_cor%>%
    # mutate(Point = ifelse(diameter >= min_diam, "Cor training", "Cor left out"))%>%
    # filter(branch == "tree2h")%>%
    ggplot(aes(x= !!sym(vars_in_model[1])))+
    geom_point(aes(y = pred_cross_section_cor, color = "Corrected model"))+
    geom_abline(slope = 1, intercept = 0)
  
  list(
    plots = list(
      corrected_comparison = correction_comparison,
      corrected = corrected),
    statistics = list(
      corrected = stats_corrected,
      corrected_min_diam = stats_corrected_min_diam),
    data = data_no_na_cor
    # NB: data is full data, with a variable called split to get the data used for training and the data 
    # used for testing. pred_cross_section is the cross section predicted by the general model,
    # and pred_cross_section_cor is the cross section predicted after correction.
  )
}


#' Model cross validation
#'
#' @param data A data.frame
#' @param formula The equation for the model
#' @param min_diam Minimal diameter at which the LiDAR can measure the diameters right 
#' @return A list with the plots, stats and data.
#' @export
#'
cross_validate = function(data,formula,min_diam){
  
  branches = unique(data$unique_branch)
  
  df_cross_val_full = vector(mode = "list", length = length(branches))
  
  for (i in seq_along(branches)) {
    # Fitting the model on all branches except branch i:
    model = fit_model(data = data%>%filter(.data$unique_branch != branches[i]),
                      formula = formula, min_diam = min_diam)
    
    # Applying the model on branch i:
    res = apply_model_cor(data = data%>%filter(.data$unique_branch == branches[i]),
                          model = model$model, min_diam = min_diam)
    
    
    # Continuer ici: on doit garder en memoire les cross_section mesurees, les cross_section 
    # predites (soit pour toute la branche soit pour juste les petites structures), puis on 
    # calcule les stats a la fion sur le tout (apres la fin de la cross-validation).
    
    df_cross_val_full[[i]] =  
      res$data%>%
      select(branch,tree,diameter,cross_section,pred_cross_section,pred_cross_section_cor)
  }
  
  names(df_cross_val_full) = branches
  
  df_cross_val_full = dplyr::bind_rows(df_cross_val_full)
  
  
  stats = 
    df_cross_val_full%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = pred_cross_section, obs = cross_section),
              EF = CroPlotR::EF(sim = pred_cross_section, obs = cross_section),
              Bias = CroPlotR::Bias(sim = pred_cross_section, obs = cross_section),
              .groups = "keep")
  
  stats_min_diam = 
    df_cross_val_full%>%
    filter(diameter < min_diam)%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = pred_cross_section, obs = cross_section),
              EF = CroPlotR::EF(sim = pred_cross_section, obs = cross_section),
              Bias = CroPlotR::Bias(sim = pred_cross_section, obs = cross_section),
              .groups = "keep")
  
  stats_corrected = 
    df_cross_val_full%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = pred_cross_section_cor, obs = cross_section),
              EF = CroPlotR::EF(sim = pred_cross_section_cor, obs = cross_section),
              Bias = CroPlotR::Bias(sim = pred_cross_section_cor, obs = cross_section),
              .groups = "keep")
  
  stats_corrected_min_diam = 
    df_cross_val_full%>%
    filter(diameter < min_diam)%>%
    group_by(tree,branch)%>%
    summarise(nrmse = CroPlotR::nRMSE(sim = pred_cross_section_cor, obs = cross_section),
              EF = CroPlotR::EF(sim = pred_cross_section_cor, obs = cross_section),
              Bias = CroPlotR::Bias(sim = pred_cross_section_cor, obs = cross_section),
              .groups = "keep")
  
  p = 
    df_cross_val_full%>%
    ggplot(aes(x= cross_section))+
    facet_wrap(tree + branch ~ ., scales = "free")+
    geom_point(aes(y = pred_cross_section, color = "global model"))+
    geom_point(aes(y = pred_cross_section_cor, color = "corrected"))+
    geom_abline(slope = 1, intercept = 0)
  
  list(
    plots = p,
    statistics = list(
      general_model = stats,
      general_model_min_diam = stats_min_diam,
      corrected = stats_corrected,
      corrected_min_diam = stats_corrected_min_diam),
    data = df_cross_val_full
    # NB: data is full data, with a variable called split to get the data used for training and the data 
    # used for testing. pred_cross_section is the cross section predicted by the general model,
    # and pred_cross_section_cor is the cross section predicted after correction.
  )
}
