library(ggplot2)
donnees=read.table("0-data/0-raw/csv/sample-data.csv", header = TRUE,sep = ";")

names(donnees)

#autre méthode
#y=donnees$volume.without.parafilm.cm3
#x=donnees$volume.with.parafilm.cm3
#plot(x,y,main = "Volume without parafilm as a fonction of the parafilm volume", 
#xlab= "V.avec.para.moins.le.V.de.para", ylab="Volume without parafilm in cm3")

ggplot(donnees,aes(x= volume.with.parafilm.cm3 ,y= volume.without.parafilm.cm3,color="regression line"))+
  geom_point(color=1)+
  
  labs(x= "volume with parafilm (cm3)", y= "volume without parafilm (cm3)", 
       title = "Volume without parafilm as a fonction of the parafilm volume (cm3)",color="")+
  geom_smooth(method = "lm", se = FALSE)+
  annotate("text",x=70, y=30,label="y= -0.8386 + 0.9593x" )+
  geom_line(aes(y=volume.with.parafilm.cm3,colour="x=y"))

y=donnees$volume.without.parafilm.cm3
x=donnees$volume.with.parafilm.cm3
  
model=lm(y~x)

model

summary(model)

#Coefficients:
#  (Intercept)            x  
#-0.8386       0.9593 

#Multiple R-squared:  0.9987,	Adjusted R-squared:  0.9987 
#F-statistic: 3.034e+04 on 1 and 38 DF,  p-value: < 2.2e-16
