#remotes::install_github("VEZY/XploRer")
library(XploRer)
#lire toutes les fonctions dans le fichier donné 
source("1-code/0-function.R")

#on prend et on renvoie
#pour un 
#xlsx_to_mtg("0-data/1-xlsx/a3bh.xlsx",mtg_file = "0-data/2-mtg/a3bh.mtg" )
all_xlsx_to_mtg(xlsx_dir ="0-data/1-xlsx", mtg_dir = "0-data/2-mtg")

  

  