#!/usr/bin/env python
# coding: utf-8

# ## 1 - Suppression de toutes les variables dans l'environnement

# In[1]:


from IPython import get_ipython
get_ipython().magic('reset -sf')


# ## 2 - Importing Library

# In[2]:


import pandas as pd
import numpy as np
from datetime import datetime
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
import time

warnings.filterwarnings("ignore")


# ## 3 - Importing Data and build the main dataset

# On dispose en tout de 40 fichiers regroupés par groupe de 4 par année. Nous avons les fichiers 'Caractéristique, usagers, lieux, 
# vehicule'. Nous avons liées l'ensemble de ces fichiers afin d'obtenir un dataset global comportant une trace de chaque accident 
#ainsi que le niveau de gravité qui est la variable qui nous interesse principalement.

# In[3]:


t1 = time.time()
for i in range(2010, 2021) :
    
    caracteristiques1 = 'caracteristiques-' + str(i) + '.csv'
    lieux1 = 'lieux-' + str(i) + '.csv'
    usagers1 = 'usagers-' + str(i) + '.csv'
    vehicules1 = 'vehicules-' + str(i) + '.csv'
    
    if i in range(2010, 2019) : 
        caracteristiques = pd.read_csv(caracteristiques1 , sep=",", encoding = 'latin-1', on_bad_lines='skip')
        lieux = pd.read_csv(lieux1, sep=",")
        usagers = pd.read_csv(usagers1, sep=",")
        vehicules = pd.read_csv(vehicules1, sep=",")
        
    else :
        caracteristiques = pd.read_csv(caracteristiques1 , sep=";")
        lieux = pd.read_csv(lieux1, sep=";")
        usagers = pd.read_csv(usagers1, sep=";")
        vehicules = pd.read_csv(vehicules1, sep=";")
        

    df1 = caracteristiques.merge(lieux, how = 'outer')

    df2 = usagers.merge(vehicules, how = 'outer')

    df2 = df2.groupby('Num_Acc', group_keys=False).apply(lambda x: x.loc[x.grav.idxmax()])

    df2 = df2.reset_index(drop=True)

    df_final = df1.merge(df2, how = 'outer')
    
    if i == 2010 :
        df_inter = df_final.copy()
    else : 
        df = pd.concat([df_final , df_inter], ignore_index = True) 
        if i !=2020 :
            df_inter = df.copy()   
    
df.to_csv('df_final.csv')
t2 = time.time()
print(t2-t1)


# In[88]:


df_suite = df.copy()


# In[5]:


df_suite.sample(5, random_state=2)


# ## 4 - Analyse Descritive de la notre base de donnée

# En raison du très grand nombres de variables, l'analyse descriptive sera regroupé dans un tableau pour chaque variable dans le rapport. Comme on peut le voir notre base de données finale nommée df est composé de 660008 ligne et de 57 variable. Au vue du traitement effectué précédemment on peut donc conclure que de 2010 à 2020 nous avons eu exactement 660008 accident avec un niveau de gravité allant de 2 à 4. Ceci nous permet déjà de tirer la conclusion suivante : 'les constats ont lui dans la majorité des cas grand le niveau de gravité est élevé ce qui est logique'

# In[89]:


df_suite.describe()


# Valeur Manquantes
valeur_na = df_suite.isna().sum().to_frame().rename(columns={ 0 : 'Valeur'})
valeur_na['pourcent'] = (valeur_na['Valeur']/660008) * 100
valeur_na.to_csv('proportion de valeur manquantes.csv')

# Modification des données erronnés de BASE_GENERALE

df_suite.jour = df_suite.jour.astype(str)
df_suite.mois = df_suite.mois.astype(str)
df_suite.hrmn = df_suite.hrmn.astype(str)
df_suite.an = df_suite.an.astype(str)
df_suite.dep = df_suite.dep.astype(str)

#Définition des fonctions de modification
def hour_correction(x) :
    if len(x) == 4:
        heure = x[:2] + ':' + x[2:]
        return heure
    elif len(x) == 3:
        heure = x[:1] + ':' + x[1:]
        return heure
    elif len(x)==2:
        heure = '00:' + x
        return heure
    elif len(x)==1:
        heure = '00:0' + x
        return heure
    else :
        return x
def year_correction(x) :
    if len(x)==2:
        return "20" + x
    else:
        return x
def dep_correction(x): 
    if x.endswith('0') and len(x)>=3:
        return x[:-1]
    elif len(x) == 1 :
        return '0' + x
    else :
        return x
    
# Correction
df_suite.an = df_suite.an.map(year_correction)
df_suite.hrmn = df_suite.hrmn.map(hour_correction)
df_suite.dep = np.where((df_suite['an'] != '2019') | 
                        (df_suite['an'] != '2020'), df_suite.dep.map(dep_correction) 
                        ,df_suite.dep)


# Retrait des département d'outre mer

list_outre_mer = [ '971','972','973','974','975','976','977','978','986','987','988']
    
for i in list_outre_mer :
    df_suite.drop( df_suite[ df_suite['dep'] == i ].index, inplace=True)
    
df_suite.shape


# Association des noms de région et de département en partant de fichier de l'insee
df_region = pd.read_csv('departements-france.csv')
df_region .head()

df_region.rename(columns={'code_departement': 'dep'},inplace=True)
df_region .head()

df_suite= pd.merge(df_suite, df_region, on ='dep', how ='left')
df_suite.shape

# Créaction de la colonne Date Accident
df_suite['Date Accident'] = ['/'.join(i) for i in zip(df_suite['an'], 
                                                      df_suite['mois'], df_suite['jour'])]
df_suite['Date Accident'] = pd.to_datetime(df_suite['Date Accident'])
df_suite.head().style.set_properties(subset = ['Date Accident'], 
                                     **{'background-color': '#B1C40F'})


# Data Visualisation avec Excel 

# Téléchargement des bonnes données csv pour la visualisation

tabviz1 = pd.DataFrame(list(df_suite.grav.value_counts().to_dict().items()), 
                       columns = ['Niveau de Gravité', 'Nombre de Cas'])
tabviz1.to_csv('Nombre de cas par niveau de gravité.csv')

accident_par_dep= df_suite.groupby("nom_departement")["nom_departement"].count().to_frame().rename(columns={'nom_departement': "Nombre d'accident"}).reset_index()
accident_par_dep.to_csv("Nombre d'accident grave par département par niveau.csv")

accident_par_dep_niveau  = df_suite.groupby(['nom_departement', 'grav'])['grav'].count().reset_index(name="Nombre d'accident").rename(columns={'grav':"Niveau de gravité"})
accident_par_dep_niveau.to_csv('Accident par niveau de gravité par région par niveau.csv')

accident_par_région = df_suite.groupby("nom_region")["nom_region"].count().to_frame().rename(columns={'nom_region': "Nombre d'accident"}).reset_index()
accident_par_région.to_csv("Nombre d'accident grave par région.csv")

v = df_suite.groupby(['nom_region', 'grav'])['grav'].count().reset_index(name="Nombre d'accident").rename(columns={'grav':"Niveau de gravité"})
v.to_csv('Accident par niveau de gravité par région.csv')

x = df_suite.groupby("an")["an"].count().to_frame().rename(columns={'an': "Nombre d'accident"}).reset_index().rename(columns={'an': 'Année'})
x.to_csv("Nombre d'accident grave par année.csv")

y = df_suite.groupby(["an",'grav'])['grav'].count().reset_index(name="Nombre d'accident").rename(columns={'an': 'Année', 'grav': 'Niveau de gravité'})
y.to_csv("Nombre d'accident par niveau de gravité par année.csv")


#Mise en index de la Date

df_suite = df_suite.set_index('Date Accident')
df_suite.sample(5, random_state=2)


# Création d'une copie de notre dataset pour commencer l'études des series temporelles

df_inter = df_suite.copy()

# Création des series temporelles pertinentes dans notre études

# Regroupement par jour du nombre d'accident par niveau de gravité

df_inter = df_inter.groupby(['Date Accident','grav'])['grav'].count().reset_index(name="Nombre total d'accident grave")
df_inter.head()

df_inter = df_inter.assign(
 Nb_Cas_grav2 = np.where(df_inter['grav'] == 2,df_inter["Nombre total d'accident grave"],0),
 Nb_Cas_grav3 = np.where(df_inter['grav'] == 3,df_inter["Nombre total d'accident grave"],0),
 Nb_Cas_grav4 = np.where(df_inter['grav'] == 4,df_inter["Nombre total d'accident grave"],0),

  ).groupby('Date Accident').agg({'Nb_Cas_grav2':sum, 'Nb_Cas_grav3':sum, 'Nb_Cas_grav4':sum, "Nombre total d'accident grave":sum}).reset_index()
df_inter = df_inter.set_index('Date Accident')
# df_inter.head()
df_inter = df_inter.rename(columns={"Nb_Cas_grav2": "Tué", "Nb_Cas_grav3": "Blessé Grave", "Nb_Cas_grav4" : "Blessé Léger"})
df_inter.sample(5)


# In[33]:


fig, ax = plt.subplots()
t_serie["serie_final"].plot(linewidth = .5, color = 'darkmagenta', figsize=(10, 8))
plt.title("Evolution de la série : serie_final")
fig.savefig('Représentation serie_final.svg', format='svg', dpi=1200, bbox_inches='tight', pad_inches = 0)
plt.show()


# In[99]:


pd.DataFrame(t.mean()).rename(columns = {0 : "Valeur Moyenne"})


# In[158]:


t_2015 = t.loc['2015'].copy()
t_2017 = t.loc['2017'].copy()
t_2016 = t.loc['2016'].copy()
t_2014 = t.loc['2014'].copy()


# In[159]:


pd.DataFrame(t_2014.std()).rename(columns = {0 : "Ecart Type 2014"})


# In[160]:


pd.DataFrame(t_2015.std()).rename(columns = {0 : "Ecart Type 2015"})


# In[143]:


pd.DataFrame(t_2017.std()).rename(columns = {0 : "Ecart Type 2017"})


# In[144]:


pd.DataFrame(t_2016.std()).rename(columns = {0 : "Ecart Type 2016"})


# In[28]:


t['serie_final'] = t['Tué'] + t['Blessé Grave']


# In[37]:


t_copy1 = t.copy()
t_copy1.drop(columns=['Blessé Grave','Blessé Léger',"Nombre total d'accident grave","Tué"], inplace = True)
t_copy1.head()


# In[39]:


t_copy1.to_csv('serie_entière.csv')


# In[30]:


t_serie = pd.DataFrame(t_copy1['serie_final'].resample('W').sum())


# In[31]:


t_serie.reset_index().rename(columns={'Date Accident': 'Date Semaine'}).set_index('Date Semaine').sample(6).T


# In[32]:


serie_final = t_serie.copy()


# In[35]:


serie_final.to_csv('serie_final.csv')


# In[156]:


t_serie_2015 = t_serie.loc['2015'].copy()
t_serie_2017 = t_serie.loc['2017'].copy()
t_serie_2016 = t_serie.loc['2016'].copy()
t_serie_2014 = t_serie.loc['2014'].copy()


# In[153]:


t_serie_2015.std()


# In[154]:


t_serie_2016.std()


# In[155]:


t_serie_2017.std()


# In[157]:


t_serie_2014.std()


# In[113]:


t_copy = t.copy()
t_copy.drop(columns=['Blessé Grave','Blessé Léger',"Nombre total d'accident grave"], inplace = True)
t_copy.head()


# In[116]:


valeur = pd.DataFrame(t_copy.values)
lag_df = pd.concat([valeur.shift(2), valeur.shift(1), valeur], axis=1)
lag_df.columns = ['t-2', 't-1', 't']
lag_df.head(5)


# In[31]:


df_inter['Year'] = df_inter.index.year
df_inter['Month'] = df_inter.index.month_name()
df_inter['Day'] = df_inter.index.day
df_inter['Day_Name'] = df_inter.index.day_name()
df_inter['Is_Weekend'] = df_inter.Day_Name.isin(["Sunday","Saturday"])


# In[69]:


df_serie = df_inter.copy()


# #### On se retrouve à présent en présence de 4 serie temporelles. Nous allons nous intéresser au nombre d'accident avec un niveau de gravité 2

# In[70]:


df_serie.isnull().mean()
# On n'a pas de valeurs manquantes ni de valeurs abérentes


# In[71]:


df_serie_columns = ['Nb_Cas_grav2','Nb_Cas_grav3','Nb_Cas_grav4',"Nombre total d'accident grave"]
df_serie_mois = df_serie[df_serie_columns].resample('M').sum()
df_serie_mois.head()


# In[72]:


df_serie_semaine = df_serie[df_serie_columns].resample('W').sum()
df_serie_semaine.head()


# ## Visualisation de quelques series

# #### Evolution du nombre total d'accident par jour

# In[118]:


sns.set_style("whitegrid")
df_serie["Nb_Cas_grav2"].plot(linewidth = .5, figsize=(10, 8))
plt.title('Evolution Journalière de la série : Tué')
plt.savefig('serie_nbaccident_par_jour.png', bbox_inches='tight')
plt.show()


# #### Evolution du nombre total d'accident par semaine

# In[37]:


sns.set_style("whitegrid")
df_serie_semaine["Nb_Cas_grav2"].plot(linewidth = .5, figsize=(10, 8))
plt.show()


# #### Evolution du nombre total d'accident par mois

# In[38]:


sns.set_style("whitegrid")
df_serie_mois["Nb_Cas_grav2"].plot(linewidth = .5, figsize=(10, 8))
plt.show()


# #### A ce stade nous avons énormément de serie temporelle que nous pouvons étudier

# ##### Nous allons nous concentrer sur l'évolution du nombre d'accident grave par mois 

# In[39]:


sns.set_style("whitegrid")
df_serie_mois["Nb_Cas_grav2"].plot(linewidth = .5, figsize=(10, 8))
plt.show()


# ## Etude de la Serie Temporelle

# ### Exploring Time Series Data

# ### Création de la serie à étudier 

# In[40]:


# grav2 + grav 3 
df_serie['serie'] = df_serie['Nb_Cas_grav2'] + df_serie['Nb_Cas_grav3']
df_serie.head()


# In[41]:


df_serie.drop(columns=['Nb_Cas_grav2','Nb_Cas_grav3','Nb_Cas_grav4',"Nombre total d'accident grave"], inplace = True)


# In[42]:


df_serie.size
# On n'a donc 2300 point de données hebdomadaire depuis 2010


# In[43]:


df_serie.describe()


# #### Création des variables temporelles

# In[44]:


df_serie['Year'] = df_serie.index.year
df_serie['Month'] = df_serie.index.month_name()
df_serie['Day'] = df_serie.index.day
df_serie['Day_Name'] = df_serie.index.day_name()
df_serie['Is_Weekend'] = df_serie.Day_Name.isin(["Sunday","Saturday"])
df_serie.head()


# #### Création des données aggrégées surles semaines, les mois et l'année

# In[45]:


df_serie_columns = ['serie']
df_serie_mois = df_serie[df_serie_columns].resample('M').sum()
df_serie_mois.head()


# In[46]:


df_serie_semaine = df_serie[df_serie_columns].resample('W').sum()
df_serie_semaine.head()


# ## Series Data Visualisation

# ### Etape 1 : Voir visualisation des données aggrégé par semaine par semaine

# In[47]:


sns.set_style("whitegrid")
df_serie_semaine["serie"].plot(linewidth = .5, figsize=(10, 8))
plt.show()


# ### Etape 2 : Visualisation de la densité des données aggrégé par semaine

# In[48]:


df_serie_semaine["serie"].plot(style='k.')
plt.show()


# ### Etape 3 : Histograme et densité

# In[49]:


df_serie_semaine["serie"].hist()
plt.show()
# On observe la proportion qu"on a pour chaque occurence


# In[50]:


df_serie_semaine["serie"].plot(kind='kde')
plt.show()


# ### Graphiques en boîte et moustaches par intervalle

# In[51]:


sns.boxplot(data=df_serie, x="Month", y='serie')


# In[52]:


g = sns.boxplot(data=df_serie, x='Day_Name', y='serie')
g.set_xticklabels(rotation=30, labels=df_serie['Day_Name'].unique())
plt.show()


# ### Nombre d'accident grav2 en semaine vs en weekend

# In[53]:


g = sns.boxplot(data=df_serie, x='Is_Weekend', y='serie')
g.set_xticklabels(rotation=30, labels=df_serie['Is_Weekend'].unique())
plt.show()


# In[54]:


g = sns.boxplot(data=df_inter, x='Is_Weekend', y="Nb_Cas_grav2")
g.set_xticklabels(rotation=30, labels=df_inter['Is_Weekend'].unique())
plt.show()


# ### Visualisation 2018

# In[55]:


one_year = df_serie.loc['2018']
groups = one_year["serie"].groupby(pd.Grouper(freq='M'))
months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
months = pd.DataFrame(months)
months.columns = range(1,13)
months.boxplot()
plt.show()


# ### Modelisation

def v_aberrante(v):
    Q1 = np.quantile(v, 0.25)  
    Q3 = np.quantile(v, 0.75)
    EIQ = Q3 - Q1
    LI = Q1 - (EIQ*1.5)
    LS = Q3 + (EIQ*1.5)    
    i = list(v.index[(v < LI) | (v > LS)])
    val = list(v[i])
    return i, val

serie_entiere = pd.read_csv('serie_entière.csv', index_col = 
                            'Date Accident', parse_dates = True)
serie_entiere.tail()

serie_final = pd.read_csv('serie_final.csv', index_col = 
                          'Date Accident', parse_dates = True)
serie_final.tail()

#serie_entière = df_inter et serie_finale = la série étudié

serie_final.describe().T

# Evolution de la série : serie_final
fig, ax = plt.subplots()
serie_final["serie_final"].plot(linewidth = .5, color = 
                                'darkmagenta', figsize=(10, 8))
plt.title("Evolution de la série : serie_final")
fig.savefig('Représentation serie_final.svg', format='svg', 
            dpi=1200, bbox_inches='tight', pad_inches = 0)
plt.show()

# Box Plot
plt.boxplot(serie_final);
ax = sns.boxplot(data=serie_final.serie_final, orient="h", palette="Set2")

# Détection et changement des valeurs aberrantes
for label in serie_final: 
    print(label ,"=",len((v_aberrante(serie_final['serie_final'])[1])))

tuplue_aberrant = v_aberrante(serie_final['serie_final'])

df_v_aberrante = {'Date de Semaine':tuplue_aberrant[0],
                  'Valeur extrême':tuplue_aberrant[1]}
df_v_aberrante = pd.DataFrame(df_v_aberrante)
df_v_aberrante.set_index('Date de Semaine', inplace = True)
df_v_aberrante

for i in tuplue_aberrant[0] : 
    serie_final.loc[i] = np.nan

serie_final['serie_final'] = serie_final['serie_final'].ffill()

serie_final['serie_final'] = serie_final['serie_final'].bfill()

serie_final.isnull().sum()

ax = sns.boxplot(data=serie_final.serie_final, orient="h", palette="Set2")

serie_final.describe().T

# Visualisation des différentes séries
fig, ax = plt.subplots()
serie_final["serie_final"].plot(linewidth = .5, color = 'b', figsize=(10, 8))
plt.title("Evolution de la série : serie_final")
fig.savefig('Représentation serie_final_sans_vab.svg', format=
            'svg', dpi=1200, bbox_inches='tight', pad_inches = 0)
plt.show()

serie_final.plot(kind='kde')
plt.show()

plt.figure(figsize=(11,4), dpi= 80)
serie_final["serie_final"].plot(linewidth = 0.5)

serie_month = pd.DataFrame(serie_entiere['serie_final'].resample('M').sum())

plt.figure(figsize=(11,4), dpi= 80)
serie_month["serie_final"].plot(linewidth = 0.5)


# Etudes des composantes déterministes

result = seasonal_decompose(serie_final, model='additif')
result.plot(observed = False, resid = False)
plt.show()

plt.figure(figsize=(11,4), dpi= 80)
result.seasonal.plot(color = 'c');

plt.figure(figsize=(11,4), dpi= 100)
result.trend.plot(color = 'b')

df_2019 = serie_final.loc['2019':'2020']
result_1 = seasonal_decompose(df_2019, model='additif')

plt.figure(figsize=(11,4), dpi= 100)
result_1.seasonal.plot(color = 'darkmagenta');

plt.figure(figsize=(11,4), dpi= 80)
result.resid.plot(color = 'c');

pd.plotting.autocorrelation_plot(result.resid)
plt.show()


# ### Modelisation

# ### Modèle de Persistence

# In[33]:


valeur = pd.DataFrame(serie_final.values)
df_persistence = pd.concat([valeur.shift(1), valeur], axis=1)
df_persistence.columns = ['t', 't+1']
df_persistence.head(5).T


# In[34]:


X = df_persistence.values
train_size = int(len(X) * 0.80)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]


# In[35]:


X = serie_final.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # predict
    yhat = history[-1]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


# Visualisation des prédictions
plt.figure(figsize=(11,6), dpi= 80)
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()

plt.figure(figsize=(11,6), dpi= 80)
# plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()

residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)
print(residuals.head())

residuals.plot()
plt.show()

pd.plotting.autocorrelation_plot(residuals)
plt.show()

residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = np.array(residuals)
qqplot(residuals, line='r')
plt.show()


# Modelisation : ARIMA

# In[44]:
serie_final.sample(5)

plot_acf(serie_final, lags=50)
plt.show()

plot_pacf(serie_final, lags=50)
plt.show()

# Fonction d'évaluation d'un modèle ARIMA pour un ensemble de paramètre (p, d, q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.80)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

# Fonction d'évaluation de la meilleure combinaison des paramètres
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

t1 = time.time()
p_values = list(range(0, 15))
d_values = list(range(0, 4))
q_values = list(range(0, 4))
evaluate_models(serie_final.values, p_values, d_values, q_values)
t2 = time.time()
print(t2-t1)
