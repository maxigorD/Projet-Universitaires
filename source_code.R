# Sommaire 

1. Analyse descriptive du jeu de données - R
2. Analyse des variables du jeu de données - R
3. Analyse de la liaison entre la variable que l’on veut expliquer et les autres variables du jeu de données - R
4. Construction de modèles de régressions simples - SAS
5. Construction d’un modèle de régression multiple - SAS
6. ACP suivie d’une CAH - R


# chargement des packages nécessaires 

library("Hmisc")
library("psych")
library("pastecs")
library("car")
library(BioStatR)
library("sjPlot")
library("ggpubr")
library(broom)
library("openxlsx")
library("ggpubr")
library("FactoMineR")
library("factoextra")
library("Hmisc")
library("corrplot")
library(cluster)


#1. Analyse descriptive du jeu de données (taille du jeu de données, type des variables à disposition, observations manquantes/aberrantes..)

# Importation des donnees

Data<-read.csv(file="path",header=T,dec=".")
DATA2<-read.csv(file="path",header=T,dec=".")
DATA3<-read.csv(file="path",header=T,dec=".")

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

# Chargement de la table obtenue de SAS

don = read.xlsx(xlsxFile="C:/Users/Orlys/Desktop/SAS - Copie/Nouvelle table/NTable 1.xlsx",sheet=1, colNames=TRUE)
don$V=scale(don$V)
don$AP=scale(don$AP)
don$RH=scale(don$RH)
don$PE=scale(don$PE)
don$AT=scale(don$AT)

# ACP
res.pca <- PCA(don, graph = FALSE)

print(res.pca)
eig.val <- get_eigenvalue(res.pca)
eig.val

fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50), ylab="Diagramme des valeurs propres", xlab="Composante Principale")

var <- get_pca_var(res.pca)
var

# CoordonnÃ©es
head(var$coord)
# Cos2: qualitÃ© de rÃ©presentation
head(var$cos2)
# Contributions aux composantes principales
head(var$contrib)

# CoordonnÃ©es des variables
head(var$coord, 4)

#Pour visualiser les variables
fviz_pca_var(res.pca, col.var = "black")

#QualitÃ© de reprÃ©sentation
head(var$cos2, 4)

#visualiser le cos2 des variables sur toutes les dimensions 
corrplot(var$cos2, is.corr=FALSE)

# Cos2 total des variables sur Dim.1 et Dim.2
fviz_cos2(res.pca, choice = "var", axes = 1:2)

# Colorer en fonction du cos2: qualitÃ© de reprÃ©sentation
fviz_pca_var(res.pca, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

# Changer la transparence en fonction du cos2
fviz_pca_var(res.pca, alpha.var = "cos2")

#Contribution des variables
head(var$contrib, 5)

#mettre en Ã©vidence les variables les plus contributives pour chaque dimension
corrplot(var$contrib, is.corr=FALSE)

# Contributions des variables Ã  PC1
fviz_contrib(res.pca, choice = "var", axes = 1, top = 10)

# Contributions des variables Ã  PC2
fviz_contrib(res.pca, choice = "var", axes = 2, top = 10)

#La contribution totale Ã  PC1 et PC2
fviz_contrib(res.pca, choice = "var", axes = 1:2, top = 10)

#Mise en Ã©vidence des variables les plus importante
fviz_pca_var(res.pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))
             
res.desc <- dimdesc(res.pca, axes = c(1,2), proba = 0.05)
# Description de la dimension 1
res.desc$Dim.1
# Description de la dimension 2
res.desc$Dim.2

#Classification ascendante hiÃ©rarchique
res.pca <- PCA(don, graph = FALSE, ncp=2)
res.hcpc = HCPC(res.pca)

