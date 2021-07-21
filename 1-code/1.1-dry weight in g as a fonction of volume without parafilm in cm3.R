library(ggplot2)
donnees=read.table("0-data/0-raw/csv/sample-data.csv", header = TRUE,sep = ";")

names(donnees)

ggplot(donnees,aes(x= volume.without.parafilm.cm3
 ,y= dry.weight.g ,color="regression line"))+
  geom_point(color=1)+
  
  labs(x= "volume without parafilm (cm3)", y= "dry weight (g)", 
       title = "dry weight (g) as a fonction of volume without parafilm (cm3)",
       colour="")+
  geom_smooth(method = "lm", se = FALSE)+
  annotate("text",x=70, y=20,label="y=-0.5454 + 0.5410x " )+
  geom_line(aes(y=volume.without.parafilm.cm3,colour="x=y"))
  
  
y=donnees$dry.weight.g
x=donnees$volume.without.parafilm.cm3  

model=lm(y~x)

model

summary(model)

#Coefficients:
#(Intercept)            x  
#-0.5454       0.5410 

#Multiple R-squared:  0.999,	Adjusted R-squared:  0.999 
#F-statistic: 3.923e+04 on 1 and 38 DF,  p-value: < 2.2e-16

