#importation de ggplot2
library(ggplot2)

#chercher le fichier dans les docs, le separateur est 
#par default ";"

donnees=read.table("0-data/0-raw/csv/sample-data.csv", header = TRUE,sep = ";")

#on reagarde comment se nommes nos variables 
names(donnees)

#on creer un plot en indiquant nos variables x et y grace à la liste sortie juste avant, 
#on introduit une légende "color" 
ggplot(donnees,aes(x= parafilm.volume.deducted.cm3
                   ,y= volume.without.parafilm.cm3 ,color="regression line"))+
  #on affiche les points avec une couleur en chiffre
  geom_point(color=1)+
  #on met des titres et on suprime le nom de la légende 
  labs(x= "parafilm volume deducted (cm3)", y= "volume without parafilm in (cm3)", 
       title = "volume without parafilm as a fonction of volume deducted without parafilm (cm3) ",color="")+
  #creation droite regression linéaire
  geom_smooth(method = "lm", se = FALSE)+
  #on ajoute l'equation de la droite de regression
  annotate("text",x=70, y=30,label="y= -0.4458 + 0.9755x" )+
  #On ajoute une droite x=y
  geom_line(aes(y=parafilm.volume.deducted.cm3,colour="x=y"))
   
#les 2 variables sont nommées x et y 
#y=volume direct method
#x=volume parafilm méthod
y=donnees$volume.without.parafilm.cm3
x=donnees$parafilm.volume.deducted.cm3

#creation d'un objet droite lineaire ici model
model=lm(y~x)

#Maintenant on affiche model, ceci a pour but de nous renseigner
#l'ordonne a l'origine (intercept) et la pente (x)
model

#Pour plus d'information on utilise egalement summary 
#2 informations importantes, Multiple R-squared plus il est proche
#de 1 plus la premiere variable explique la premiere 
#p-value si < 0,05 la corelation de la droite n'est pas egale ? 0
summary(model)


#Coefficients:
  #(Intercept)            x  
#-0.4458       0.9755  

#Multiple R-squared:  0.9988,	Adjusted R-squared:  0.9987
#F-statistic: 3.06e+04 on 1 and 38 DF,  p-value: < 2.2e-16