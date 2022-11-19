# Sommaire 

1. Analyse descriptive du jeu de données
2. Analyse des variables du jeu de données 
3. Analyse de la liaison entre la variable que l’on veut expliquer et les autres variables du jeu de données
4. Construction de modèles de régressions simples
5. Construction d’un modèle de régression multiple
6. ACP suivie d’une CAH


#1. Analyse descriptive du jeu de données (taille du jeu de données, type des variables à disposition, observations manquantes/aberrantes..)

# Importation des donnees

Data<-read.csv(file="path",header=T,dec=".")
DATA2<-read.csv(file="path",header=T,dec=".")
DATA3<-read.csv(file="path",header=T,dec=".")

# chargement des packages nécessaires 

library("Hmisc")
library("psych")
library("pastecs")
library("car")
install.packages("sjPlot")
install.packages("BioStatR")
install.packages("ggpubr")
library(BioStatR)
library("sjPlot")
library("ggpubr")
library(broom)

# Description globale des donnees 

D1 <- stat.desc(DATA, basic = TRUE)
D1
D2 <- stat.desc(DATA2, basic = TRUE)
D2
D3 <- stat.desc(DATA3, basic = TRUE)
D3

#2. Analyse des variables du jeu de données : traitements univariés (répartition, adéquation à une loi théorique...)

boxplot(DATA$V, ylab="Vide d'échappement")
boxplot(DATA$AP, ylab ="Pression ambiante")
boxplot(DATA$RH, ylab= "Humidité relative")
boxplot(DATA$ï..AT,ylab="Température ambiante")
boxplot(DATA$PE, ylab="Production d'énergie horaire nette (PE)")


'Détection et suppression des valeurs aberrantes (x2)' 

boxplot.stats(Data$AP)$out
boxplot.stats(Data$RH)$out

DATASVA1<-Data[-which(Data$AP %in% boxplot.stats(Data$AP)$out), ]
DATASVA2<-DataSVA1[-which(DataSVA1$RH %in% boxplot.stats(DataSVA1$RH)$out), ]
DAta<-DATASVA2

boxplot.stats(DAta$AP)$out
boxplot.stats(DAta$RH)$out

DATASVA3<-DAta[-which(DAta$AP %in% boxplot.stats(DAta$AP)$out), ]
DATASVA4<-DATASVA3[-which(DATASVA3$RH %in% boxplot.stats(DATASVA3$RH)$out), ]
DATA<-DATASVA4

# Représentation graphique et analyse de la distribution des données

summary(DATA)

hist(DATA$PE,col="yellow",freq=F, main = "Histogramme de densité de la variable PE ")
densite <- density(DATA$PE) # estimer la densité que représente ces différentes valeurs
lines(densite, col = "red",lwd=3) # Superposer une ligne de densité à l'histogramme

hist(DATA$V,col="yellow",freq=F, main = "Histogramme de densité de la variable V ")
densite2 <- density(DATA$V) # estimer la densité que représente ces différentes valeurs
lines(densite2, col = "red",lwd=3) # Superposer une ligne de densité à l'histogramme

hist(DATA$AP,col="yellow",freq=F, main = "Histogramme de densité de la variable AP ")
densite <- density(DATA$AP) # estimer la densité que représente ces différentes valeurs
lines(densite, col = "red",lwd=3) # Superposer une ligne de densité à l'histogramme

hist(DATA$RH,col="yellow",freq=F, main = "Histogramme de densité de la variable RH ")
densite <- density(DATA$RH) # estimer la densité que représente ces différentes valeurs
lines(densite, col = "red",lwd=3) # Superposer une ligne de densité à l'histogramme

hist(DATA$ï..AT,col="yellow",freq=F, main = "Histogramme de densité de la variable AT ")
densite <- density(DATA$ï..AT) # estimer la densité que représente ces différentes valeurs
lines(densite, col = "red",lwd=3) # Superposer une ligne de densité à l'histogramme


Normal_V<-ggqqplot(DATA$V)
Normal_RH<-ggqqplot(DATA$RH)
Normal_AP<-ggqqplot(DATA$AP)
Normal_AT<-ggqqplot(DATA$ï..AT)
Normal_PE<-ggqqplot(DATA$PE)


'test de normalité pour des grands échantillons'

ks.test(DATA$V,"pnorm",mean=54.3, sd=12.7)
ks.test(DATA$ï..AT,"pnorm",mean=19.65, sd=7.45)
ks.test(DATA$AP,"pnorm",mean=101.33, sd=5.94)
ks.test(DATA$RH,"pnorm",mean=73.3, sd=14.6)
ks.test(DATA$PE,"pnorm",mean=454.36, sd=1.707)

'Normalisation pour pouvoir effectuer la régresssion linéaire'

DATA$V=scale(DATA$V)
DATA$AP=scale(DATA$AP)
DATA$RH=scale(DATA$RH)
DATA$PE=scale(DATA$PE)
DATA$ï..AT=scale(DATA$ï..AT)

#Analyse des correlations avec la production (test non paramètrique)

cor_AT<-cor.test(DATA$ï..AT,DATA$PE, method="kendall")
cor_V<-cor.test(DATA$V,DATA$PE,method="kendall")
cor_AP<-cor.test(DATA$AP,DATA$PE,method="kendall")
cor_RH<-cor.test(DATA$RH,DATA$PE,method="kendall")
cor_AT
cor_V
cor_AP
cor_RH

'Visualisation'

plot(jitter(DATA$V),jitter(DATA$PE),xlab="Vide d’échappement",ylab="Production d'énergie")
plot(jitter(DATA$AP),jitter(DATA$PE),xlab="Pression ambiante",ylab="Production d'énergie")
plot(jitter(DATA$RH),jitter(DATA$PE),xlab="Humidité relative",ylab="Production d'énergie")
plot(jitter(DATA$ï..AT),jitter(DATA$PE),xlab="Température Ambiante",ylab="Production d'énergie")
