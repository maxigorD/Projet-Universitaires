import streamlit as st
import numpy as np
import pandas as pd
import xlrd
from datetime import datetime
import plotly.express as px
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

####################### Fonction pour changer la couleur de st.write ############## 
def write(url):
     st.write(f'<p style="color:#83656B;font-size:20px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
        
#************************* Début de l'application streamlit **************************

################ Etape de Chargement des fichiers par l'utilisateur ##################

st.header("Analyse des données des dépenses de la Caisse Pimaire d'Assurance Maladie CPAM")

st.markdown("<h1 style='text-align: center; color: green;font-size:50px;'>****************</h1>", unsafe_allow_html=True)
st.write('***************************************************')

st.markdown("<h1 style='text-align: center; color: black;font-size:35px;'> Instructions for use. </h1>", unsafe_allow_html=True)

original_text = '<p style="font-family:Courier; color:Blue; font-size: 20px;"> Hello user ! How are you ?</p>'

st.markdown(original_text, unsafe_allow_html=True)

original_text1 = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Welcome to the dynamic and interactive streamlit analysis application. </p>'
st.markdown(original_text1 , unsafe_allow_html=True)

st.write("The objective of this application is to propose an automatic analysis model of the expenses of the primary health insurance fund. The analyzed data come from an extraction of the National Inter-Scheme Health Insurance System (SNIIRAM) concerning all health insurance reimbursements, all schemes included. ")

st.write("Expenditures are detailed according to six analysis axes (period, benefit, reimbursement organization, care recipient, performing health professional, prescribing health professional) and seven indicators of amount (total expenditure, reimbursement base, reimbursed amount, overrun) and volume (count, quantity, coefficient). In total, each service line is described by 55 variables.")

st.write("The processing of information from each database requires the use of the excel file which is the descriptive file associated with the data. During the execution of the application you will realize that there is a certain anomaly due to the fact that not all the values of the database are present in all the databases. The application will provide you each time with the values concerned so that you can have them corrected.") 

st.write("The visualizations that we have done are still quite basic, because in order to do very relevant visualizations, you need to have a great knowledge of the field. Nevertheless, the statistics that you will download at the end of the session are correct.")

st.markdown("<h1 style='text-align: center; color: black;font-size:30px;'> So let's go !!! </h1>", unsafe_allow_html=True) 

st.write('***************************************************')



st.subheader("Chargement des fichiers")

st.write("Afin de commencer à réaliser votre analyse, veuillez vous rendre sur le site https://www.data.gouv.fr/fr/datasets/open-damir-base-complete-sur-les-depenses-dassurance-maladie-inter-regimes/#resources afin de télécharger et charger dans l'application le fichier à analyser.")

file_gz = st.file_uploader("Upload a gz file", type=([".gz"]))

st.write("Veuillez charger également le fichier Lexique qui contient les informations relative aux données de la Base")

file_xls = st.file_uploader("Upload a xls file", type=([".xls"]))

st.markdown("<h1 style='text-align: center; color: green;font-size:35px;'>**********</h1>", unsafe_allow_html=True)

################ Etape de traitement ##################
if file_gz and file_xls:
    
    class MyDict(dict):
        def __missing__(self, key):
            return key

    def get_lenght_for_spliting_dataset(file): # Cette fonction permet de renvoyer le la taille du fichier (.gz) et le nombre qui permettra de diviser la base de données.
        file_lenght = len(pd.read_csv(file, compression = 'gzip', delimiter = ',', usecols = [2]))
        lenght_for_spliting_dataset = int(file_lenght/6)
        return [file_lenght, lenght_for_spliting_dataset] 
    
    def remove_last_columns(_dataframe_): # Dans certains cas la base de données possèdent une dernière colonne nommée ''Unnamed: 55' qui ne sert à rien et qui est vide. Cette fonction permet de supprimer cette colonne la si elle existe.
        last_columns_name = list(_dataframe_.iloc[:, [-1]].columns)[0] 
        if last_columns_name == 'Unnamed: 55' :
            _dataframe_.drop(columns = last_columns_name, axis = 1, inplace = True)
        return _dataframe_
    
    def read_dataframe(file): # Cette fonction permet de créer et de définir dynamiquement les n dataframes
        df_chunk = pd.read_csv(file, compression = 'gzip', delimiter = ',', chunksize = 1000000)
        dataframe_number = 0
        len_df_chunk = 0
        for chunk in df_chunk :
            dataframe_number +=1
            chunk.reset_index(drop=True, inplace = True)
            chunk = remove_last_columns(chunk)
            name_of_dataframe = f"dataframe_{dataframe_number}"
            st.session_state[f'{name_of_dataframe}'] = chunk
            len_df_chunk += len(chunk)
        return [dataframe_number, len_df_chunk]  
    
    def get_date_of_file(_dataframe_) : #Cette fontion nous permet de récupérer avec le fichier la date et le mois concerné
        Date = str(_dataframe_.loc[2, 'FLX_ANN_MOI'])
        Date = Date[:4] +'-'+ Date[4:]
        Date = datetime.strptime(Date, '%Y-%m')
        Date = Date.strftime("%B") +' '+ Date.strftime("%Y")
        return Date
    
    def get_label_name(_columns_name_, excel_file):#Cette fontion nous permet de récupérer dans le fichier excel l'ensemble des noms de feuilles de calcul qui correspond aussi au noms des variables
        Lexique_open_DAMIR_sheet0 = pd.read_excel(excel_file, sheet_name = 0, engine = "xlrd")
        Lexique_open_DAMIR_sheet0.drop(Lexique_open_DAMIR_sheet0.columns[[list(range(2,5))]], axis = 1, inplace = True)
        Lexique_open_DAMIR_sheet0.rename(columns = {'Tables A à partir de 2015':'Nom de la Variable', 'Unnamed: 1':'Description'}, inplace = True)
        Lexique_open_DAMIR_sheet0 = Lexique_open_DAMIR_sheet0.set_index('Nom de la Variable')
        return Lexique_open_DAMIR_sheet0.loc[_columns_name_, 'Description']
    
    def valeur_maquante_BIG_DATA(_string_): # Cette fonction permet de récupérer la proportion de valeurs manquantes dans notre BIG DATA.
        liste_nan_value_in_dataframe = []
        for i in range(1, st.session_state.dataframe_number+1):
            name_of_dataframe= f"{_string_}_{i}"
            variable_intermédiaire_1 = pd.DataFrame(st.session_state[f"{name_of_dataframe}"].isna().sum())
            variable_intermédiaire_2 = variable_intermédiaire_1.loc[variable_intermédiaire_1[0] != 0].copy().rename(columns = {0:'Nombre de valeur nan dans la base dataframe'})
            liste_nan_value_in_dataframe.append(variable_intermédiaire_2)
        All_nan_values = pd.DataFrame()
        # liste_nan_value_in_dataframe[0] = liste_nan_value_in_dataframe[0].replace(np.nan, 0)
        df_intermédiaire_pour_somme = liste_nan_value_in_dataframe[0]
        for dataframe in liste_nan_value_in_dataframe[1:]:
            All_nan_values = df_intermédiaire_pour_somme.add(dataframe,  fill_value=0.000001)
            df_intermédiaire_pour_somme = All_nan_values
        return All_nan_values
    
    def get_label_list_from_excel_file(file_xls): # Cette fonction permet de retourner la liste de toutes les variables présente dans le fichier excel.
        Lexique_open_DAMIR = pd.ExcelFile(file_xls)
        Liste_des_variables_du_lexique = Lexique_open_DAMIR.sheet_names

        if Liste_des_variables_du_lexique[-1] == 'Feuil1' :
            Liste_of_sheet_name = Liste_des_variables_du_lexique[2:-1]
        else:
            Liste_of_sheet_name = Liste_des_variables_du_lexique[2:]
        return Liste_of_sheet_name

    Liste_of_sheet_name = get_label_list_from_excel_file(file_xls)
    
    def get_dict_with_dataframe(_dataframe_): # Cette fonction permet de retourner un dictionaire contenant la liste des valeur à encoder pour chaque variable du fichier excel
            return dict(zip(_dataframe_.iloc[:, 0], _dataframe_.iloc[:, 1]))
    
    def encoding_dataframe_for_nan_study(_dataframe_):# Cette fonction permet d'encoder chaque dataframe
        _dataframe_for_action = _dataframe_.copy()
        for columns in Liste_of_sheet_name : 
            # if not(columns.__contains__("REG")):
            dict_with_columns_values = get_dict_with_dataframe(pd.read_excel(file_xls, sheet_name = columns, engine = "xlrd"))
            if columns in list(_dataframe_for_action.columns):
                _dataframe_for_action[columns]= _dataframe_for_action[columns].map(dict_with_columns_values)
        return _dataframe_for_action
    
    def encoding_dataframe_for_analysis(_dataframe_):# Cette fonction permet d'encoder chaque dataframe en tachant de ne pas écraser les valeurs inexistantes dans le classeur excel
        _dataframe_for_action = _dataframe_.copy()
        for columns in Liste_of_sheet_name : 
            # if not(columns.__contains__("REG")):
            dict_with_columns_values = MyDict(get_dict_with_dataframe(pd.read_excel(file_xls, sheet_name = columns, engine = "xlrd")))
            if columns in list(_dataframe_for_action.columns):
                _dataframe_for_action[columns]= _dataframe_for_action[columns].map(dict_with_columns_values)
                _dataframe_for_action[columns] = _dataframe_for_action[columns].astype(str)
        return _dataframe_for_action
    
    def encoding_action(dataframe_final, dataframe_inital, encoding_fonction):  #Cette fonction permet de réaiser dynamiquement l'encodage pour tous les dataframes
        for i in range(1, st.session_state.dataframe_number +1) :
            name_of_dataframe_complet = f"{dataframe_final}_{i}"
            st.session_state[f'{name_of_dataframe_complet}'] = encoding_fonction(st.session_state[f"{dataframe_inital}_{i}"])
            
    def get_nan_dataframe(_dataframe_): #Cette fonction permet de récupérer des dataframes contenant les valeurs manquantes.
        df_nan_percent = (_dataframe_/st.session_state.len_df_chunk * 100).round(2).reset_index().rename(columns = {'index':'Variable avec les valeurs manquantes'})
        df_nan_percent_value = df_nan_percent.copy()
        df_nan_percent['Pourcentage de valeurs manquantes'] = df_nan_percent['Nombre de valeur nan dans la base dataframe'].astype(str) + '%'
        df_nan_percent['Label_description']=''
        for i in range(len(df_nan_percent)):
            df_nan_percent.loc[i, 'Label_description'] = get_label_name(df_nan_percent.loc[i, 'Variable avec les valeurs manquantes'], file_xls)
        df_nan_percent = pd.DataFrame(df_nan_percent, columns = ['Variable avec les valeurs manquantes', 'Label_description', 'Pourcentage de valeurs manquantes'])
        return [df_nan_percent, df_nan_percent_value]
    
    def get_values_absent_in_description(columns_name): # Cette fonction me permet de faire recouper entre les dataframe encodé et ceux non encodé afin d'obtenir la liste exact des valeurs qui manque dans le classeur excel
        list_ = []
        for i in range(1, st.session_state.dataframe_number+1):
            name_of_dataframe_complet = f"dataframe_complet_{i}"
            name_row_with_nan = f"index_dataframe_{i}_row_with_nan"
            name_of_initial_dataframe = f"dataframe_{i}"
            st.session_state[f"{name_row_with_nan}"] = st.session_state[f"{name_of_dataframe_complet}"][columns_name].index[st.session_state[f"{name_of_dataframe_complet}"][columns_name].isna()]
            list_ += list(pd.DataFrame(st.session_state[f"{name_of_initial_dataframe}"], columns = [columns_name], index = st.session_state[f"{name_row_with_nan}"])[columns_name].unique())
        return set(list_)
    
    def variable_exploration (__variable_name__): # Cette fonction me permet de faire recouper entre les dataframe encodé et ceux non encodé afin d'obtenir la liste exact des valeurs qui manque dans le classeur excel
        try : 
            if float(st.session_state.dataframe_complet_for_variable_exploration[st.session_state.dataframe_complet_for_variable_exploration['Variable avec les valeurs manquantes'] == __variable_name__]['Nombre de valeur nan dans la base dataframe']) != float(st.session_state.dataframe_for_variable_exploration[st.session_state.dataframe_for_variable_exploration['Variable avec les valeurs manquantes'] == __variable_name__]['Nombre de valeur nan dans la base dataframe']) :
                valeur_qui_ont_disparu = list(get_values_absent_in_description(__variable_name__))
                valeur_qui_ont_disparu = [i for i in valeur_qui_ont_disparu if i != 'nan']
                sentence_4 = 'Voici la liste exhaustive des valeurs absents dans le fichier lexique pour cette variable ' + str(valeur_qui_ont_disparu)
                st.write(sentence_4) 
            else :
                sentence_6 = "La présence des valeurs manquantes pour cette variable est du à l'absence de ces valeurs dans le fichiers original"
                st.write(sentence_6) 
        except :
            sentence_8 = 'Pour la variable ' + __variable_name__ + ' voici la liste des valeurs concernées'
            valeur_qui_ont_disparu = list(get_values_absent_in_description(__variable_name__))
            sentence_11 =  __variable_name__ + ' : ' + str(valeur_qui_ont_disparu)
            st.write(sentence_8) 
            st.write(sentence_11)
            
    def get_month_spent(dataframe) : # Cette fonction me permet pour un dataframe donné de calcule les KPI qui serviront à la visualisation. 
        ### La colonne qui regroupe les montants des remboursements est PRS_REM_MNT
        Dépense_global = sum(dataframe.PRS_REM_MNT)
        Dépense_par_secteur = dataframe.groupby(['PRS_PPU_SEC'])['PRS_REM_MNT'].sum().reset_index().rename(columns = {'PRS_PPU_SEC' : 'Secteur', 'PRS_REM_MNT': 'Montant total du rembourement'}).round(2).set_index('Secteur')
        Dépense_par_nature_dassurance = dataframe.groupby(['ASU_NAT'])['PRS_REM_MNT'].sum().reset_index().rename(columns = {'ASU_NAT' : "Type d'assurance", 'PRS_REM_MNT': 'Montant total du rembourement'}).round(2).set_index("Type d'assurance")
        # Dépense_par_nature_dassurance = (Dépense_par_nature_dassurance.set_index("Type d'assurance") / Dépense_global * 100).round(4)# à mettre en pourcentage
        Dépense_par_tranche_Age = dataframe.groupby(['AGE_BEN_SNDS'])['PRS_REM_MNT'].sum().reset_index().rename(columns = {'AGE_BEN_SNDS' : "Tranche d'âge du Bénéficiaire", 'PRS_REM_MNT': 'Montant total du rembourement'}).round(2).set_index("Tranche d'âge du Bénéficiaire") 
        Dépense_par_Accident_de_Travail = dataframe.groupby(['ATT_NAT'])['PRS_REM_MNT'].sum().reset_index().rename(columns = {'ATT_NAT' : "Nature de l'accident de Travail", 'PRS_REM_MNT': 'Montant total du rembourement'}).round(2).set_index("Nature de l'accident de Travail") 
        Dépense_par_Sex = dataframe.groupby(['BEN_SEX_COD'])['PRS_REM_MNT'].sum().reset_index().rename(columns = {'BEN_SEX_COD' : "Sex du Bénéficiaire", 'PRS_REM_MNT': 'Montant total du rembourement'}).round(2).set_index("Sex du Bénéficiaire")
        Dépense_par_Region = dataframe.groupby(['BEN_RES_REG'])['PRS_REM_MNT'].sum().reset_index().rename(columns = {'BEN_RES_REG' : "Région de Résidence du Bénéficiaire", 'PRS_REM_MNT': 'Montant total du rembourement'}).round(2).set_index("Région de Résidence du Bénéficiaire")
        return [Dépense_global, Dépense_par_secteur, Dépense_par_nature_dassurance, Dépense_par_tranche_Age, Dépense_par_Accident_de_Travail, Dépense_par_Sex, Dépense_par_Region]
    
    def to_excel(df, sheet_name): # Cette fonction permet créer un fichier excel à partir des dataframes que l'utilisateur pourra télécharger
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name = sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    # APPLICATION ENTRYPOINT
        
    if 'dataframe_1' not in st.session_state:
        st.session_state.liste = read_dataframe(file_gz)
        st.session_state.dataframe_number = st.session_state.liste[0]
        st.session_state.len_df_chunk = st.session_state.liste[1]
        st.session_state.Date = f"The file you downloaded contains all the data relating to the expenditure of the Primary Health Insurance Fund for the period of {get_date_of_file(st.session_state.dataframe_1)}"
        encoding_action('dataframe_complet', 'dataframe', encoding_dataframe_for_nan_study)
        st.session_state.nan_value_indataframe = valeur_maquante_BIG_DATA('dataframe')
        st.session_state.nan_value_indataframe_complet = valeur_maquante_BIG_DATA('dataframe_complet')
        st.session_state.df_nan_value_indataframe = get_nan_dataframe(st.session_state.nan_value_indataframe)
        st.session_state.df_nan_value_indataframe_complet = get_nan_dataframe(st.session_state.nan_value_indataframe_complet)
        st.session_state.dataframe_complet_for_variable_exploration = st.session_state.df_nan_value_indataframe_complet[1].copy()
        st.session_state.dataframe_for_variable_exploration = st.session_state.df_nan_value_indataframe[1].copy()
        encoding_action('dataframe_complet_analysis', 'dataframe', encoding_dataframe_for_analysis)
        
        for i in range(1, st.session_state.dataframe_number+1) :
            name_of_dataframe_complet = f"dataframe_complet_analysis_{i}"
            globals()[f'liste_spent_{i}'] = get_month_spent(st.session_state[f'{name_of_dataframe_complet}'])
            
        Dépense_global = 0
        for i in range(1, st.session_state.dataframe_number+1):
            Dépense_global += globals()[f'liste_spent_{i}'][0]
        st.session_state.Dépense_global = Dépense_global

        Dépense_par_secteur = liste_spent_1[1]
        for i in range(2, st.session_state.dataframe_number+1):
            Dépense_par_secteur = Dépense_par_secteur.add(globals()[f'liste_spent_{i}'][1])
        st.session_state.Dépense_par_secteur = Dépense_par_secteur.reset_index()

        Dépense_par_nature_dassurance = liste_spent_1[2]
        for i in range(2, st.session_state.dataframe_number+1):
            Dépense_par_nature_dassurance = Dépense_par_nature_dassurance.add(globals()[f'liste_spent_{i}'][2])
        st.session_state.Dépense_par_nature_dassurance = Dépense_par_nature_dassurance.reset_index()

        Dépense_par_tranche_Age = liste_spent_1[3]
        for i in range(2, st.session_state.dataframe_number+1):
            Dépense_par_tranche_Age = Dépense_par_tranche_Age.add(globals()[f'liste_spent_{i}'][3])
        st.session_state.Dépense_par_tranche_Age = Dépense_par_tranche_Age.reset_index()

        Dépense_par_Accident_de_Travail = liste_spent_1[4]
        for i in range(2, st.session_state.dataframe_number+1):
            Dépense_par_Accident_de_Travail = Dépense_par_Accident_de_Travail.add(globals()[f'liste_spent_{i}'][4])
        st.session_state.Dépense_par_Accident_de_Travail = Dépense_par_Accident_de_Travail.reset_index()

        Dépense_par_Sex = liste_spent_1[5]
        for i in range(2, st.session_state.dataframe_number+1):
            Dépense_par_Sex = Dépense_par_Sex.add(globals()[f'liste_spent_{i}'][5])
        st.session_state.Dépense_par_Sex = Dépense_par_Sex.reset_index()
        
        Dépense_par_Region = liste_spent_1[6]
        for i in range(2, st.session_state.dataframe_number+1):
            Dépense_par_Region = Dépense_par_Region.add(globals()[f'liste_spent_{i}'][6])
        st.session_state.Dépense_par_Region = Dépense_par_Region.reset_index()
        
    st.markdown("<h1 style='text-align: center; color: black;font-size:30px;'>Voici un échantillon de 50 lignes des fichiers que vous avez chargé</h1>", unsafe_allow_html=True)
    st.write(st.session_state.Date)
    st.write(st.session_state.dataframe_complet_3.head(50))
    
    st.markdown("<h1 style='text-align: center; color: green;font-size:35px;'>**********</h1>", unsafe_allow_html=True)
    st.write('WARNING')
    st.warning("L'encodage du fichier mensuel des dépenses (.gz) par le fichier excel indique l'absence de certaines valeurs de la Base de donnée dans le fichier excel")
    
    st.session_state.selectbox_1 = st.selectbox("Souhaitez vous connaître ces valeurs là ?", ['None' ,"Yes", "No"])
    
    if st.session_state.selectbox_1 == 'None' :
        st.write("Vous n'avez pas encore fait un choix...Veuillez sélectionner une valeur s'il vous plait")
        
    elif st.session_state.selectbox_1 == "Yes":
        
        list_variable_create_NaNvalues = [x for x in list(st.session_state.df_nan_value_indataframe_complet[0]['Variable avec les valeurs manquantes']) if x not in list(st.session_state.df_nan_value_indataframe[0]['Variable avec les valeurs manquantes'])]

        Nombre_de_variable_avec_nan_créer = len(list_variable_create_NaNvalues)

        sentence_1 = f"Voici les variables concernées : {str(set(list_variable_create_NaNvalues))}."
        st.write(sentence_1)

        list_selectbox_2 = list(set(list_variable_create_NaNvalues))
        list_selectbox_2.insert(0, 'None')
        st.session_state.selectbox_2 = st.selectbox("Veuillez sélectionner la variable pour découvrir les valeurs concernées", list_selectbox_2)
        
        if st.session_state.selectbox_2 == 'None' :
            st.write("Vous n'avez pas encore fait un choix...Veuillez sélectionner une valeur s'il vous plait")
        else :
            st.write(st.session_state.selectbox_2)
            variable_exploration(st.session_state.selectbox_2)
            st.info("Vous pouvez récupérer ces valeurs puis contacter l'administrateur da la Base de données via data gouv afin de rajouter les bonnes valeurs au fichier excel")
            markdown = st.markdown("""
                                    <style>
                                    div.stButton > button:hover {
                                        background-color: #0099ff;
                                        color:white;
                                        }
                                    </style>""", unsafe_allow_html=True)
            st.session_state.button_1 = st.button("Cliquez ici pour poursuivre l'analyse :)")
            
            if st.session_state.button_1 : 
                st.write("Parfait, poursuivons l'analyse")
                st.markdown("<h1 style='text-align: center; color: green;font-size:35px;'>**********</h1>", unsafe_allow_html=True)
                st.header("Analyse des dépenses Mensuelles")
                st.markdown("<h1 style='text-align: center; color: blue;font-size:35px;'>*******</h1>", unsafe_allow_html=True)

                st.subheader("Traitement des valeurs Manquantes et des Anomalies")

                st.write('***************************************************')
                st.markdown("<h1 style='text-align: left; color: green;font-size:18px;'>Voici la proportion de valeurs manquantes dans notre fichier. </h1>", unsafe_allow_html=True)
                st.write(st.session_state.df_nan_value_indataframe[0])
                st.write('Afin de ne pas biaiser nos calcul nous conserverons les lignes avec nan et ignorerons les variables en possédant dans notre analyse')
                st.write('***************************************************')

                st.markdown("<h1 style='text-align: center; color: blue;font-size:35px;'>*******</h1>", unsafe_allow_html=True)

                st.markdown("<h1 style='text-align: center; color: black;font-size:35px;'>DASHBOARD</h1>", unsafe_allow_html=True)

                st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par secteur</h1>", unsafe_allow_html=True)

                fig = px.pie(st.session_state.Dépense_par_secteur, values='Montant total du rembourement', names='Secteur', color='Secteur',
                     color_discrete_map={
                                         'PRIVE':'darkblue',
                                         'PUBLIC':'cyan'})

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par accident de travail</h1>", unsafe_allow_html=True)

                fig = px.pie(st.session_state.Dépense_par_Accident_de_Travail, values='Montant total du rembourement', names="Nature de l'accident de Travail", color="Nature de l'accident de Travail",
                     color_discrete_map={
                                         'ACCIDENT DU TRAJET':'darkblue',
                                         'ACCIDENT DU TRAVAIL':'cyan',
                                         'MALADIE PROFESSIONNELLE' : 'lightcyan',
                                         'SANS OBJET' : 'royalblue'})

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par tranche d'âge</h1>", unsafe_allow_html=True)

                fig = px.bar(st.session_state.Dépense_par_tranche_Age, x="Tranche d'âge du Bénéficiaire", y='Montant total du rembourement')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par région</h1>", unsafe_allow_html=True)
        
                fig = px.bar(st.session_state.Dépense_par_Region, x="Région de Résidence du Bénéficiaire", y='Montant total du rembourement')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par sexe</h1>", unsafe_allow_html=True)
                fig = px.pie(st.session_state.Dépense_par_Sex, values='Montant total du rembourement', names='Sex du Bénéficiaire')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par type d'assurance</h1>", unsafe_allow_html=True)
                fig = px.pie(st.session_state.Dépense_par_nature_dassurance, values='Montant total du rembourement', names="Type d'assurance")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<h1 style='text-align: center; color: blue;font-size:35px;'>*******</h1>", unsafe_allow_html=True)
        
                st.write('***************************************************')

                st.subheader("Si vous le souhaitez vous pouvez télécharger au format excel les résultats des analyses")

                date = get_date_of_file(st.session_state.dataframe_1)

                df_xlsx1 = to_excel(st.session_state.Dépense_par_secteur, 'Dépense par secteur')
                st.download_button(label=f"📥 Dépense par secteur {date}",
                                        data=df_xlsx1 ,
                                        file_name= f'Dépense par secteur {date}.xlsx')

                df_xlsx2 = to_excel(st.session_state.Dépense_par_tranche_Age, "Dépense par tranche d'âge")
                st.download_button(label=f"📥 Dépense par tranche d'âge {date}",
                                                data=df_xlsx2 ,
                                                file_name= f"Dépense par tranche d'âge {date}.xlsx")

                df_xlsx3 = to_excel(st.session_state.Dépense_par_Accident_de_Travail, "Dépense par accident de travail")
                st.download_button(label=f"📥 Dépense par accident de travail {date}",
                                                data=df_xlsx3 ,
                                                file_name= f"Dépense par accident de travail {date}.xlsx")

                df_xlsx4 = to_excel(st.session_state.Dépense_par_Region, "Dépense par région")
                st.download_button(label=f"📥 Dépense par région {date}",
                                                data=df_xlsx4 ,
                                                file_name= f"Dépense par région {date}.xlsx")

                df_xlsx5 = to_excel(st.session_state.Dépense_par_Sex, "Dépense par sex")
                st.download_button(label=f"📥 Dépense par sex {date}",
                                                data=df_xlsx5 ,
                                                file_name= f"Dépense par sex {date}.xlsx")

                df_xlsx5 = to_excel(st.session_state.Dépense_par_nature_dassurance, "Dépense par nature d'assurance")
                st.download_button(label=f"📥 Dépense par nature d'assurance {date}",
                                                data=df_xlsx5 ,
                                                file_name= f"Dépense par nature d'assurance {date}.xlsx")


                st.markdown("<h1 style='text-align: center; color: black;font-size:30px;'>END APP</h1>", unsafe_allow_html=True)
                st.write('***************************************************')
                st.markdown("<h1 style='text-align: center; color: green;font-size:50px;'>****************</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='color: black;font-size:18px;'>Créateur : Maxigor DEKADJEVI</h1>", unsafe_allow_html=True) 
                st.markdown("<h1 style='color: black;font-size:18px;'>Alternant Data Developpeur at Akane</h1>", unsafe_allow_html=True)
                st.markdown("<h1 style='color: black;font-size:18px;'>Profile linkedin : https://www.linkedin.com/in/maxigor-davidson-dekadjevi/ </h1>", unsafe_allow_html=True)
                st.markdown("<h1 style='color: black;font-size:18px;'>GitHub : https://github.com/maxigorD </h1>", unsafe_allow_html=True)
            
    else:
        st.write("Parfait, poursuivons l'analyse")
        st.markdown("<h1 style='text-align: center; color: green;font-size:35px;'>**********</h1>", unsafe_allow_html=True)
        st.header("Analyse des dépenses Mensuelles")
        st.markdown("<h1 style='text-align: center; color: blue;font-size:35px;'>*******</h1>", unsafe_allow_html=True)
        
        st.subheader("Traitement des valeurs Manquantes et des Anomalies")
        
        st.write('***************************************************')
        st.markdown("<h1 style='text-align: left; color: green;font-size:18px;'>Voici la proportion de valeurs manquantes dans notre fichier. </h1>", unsafe_allow_html=True)
        st.write(st.session_state.df_nan_value_indataframe[0])
        st.write('Afin de ne pas biaiser nos calcul nous conserverons les lignes avec nan et ignorerons les variables en possédant dans notre analyse')
        st.write('***************************************************')
        
        st.markdown("<h1 style='text-align: center; color: blue;font-size:35px;'>*******</h1>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: black;font-size:35px;'>DASHBOARD</h1>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par secteur</h1>", unsafe_allow_html=True)
        
        fig = px.pie(st.session_state.Dépense_par_secteur, values='Montant total du rembourement', names='Secteur', color='Secteur',
             color_discrete_map={
                                 'PRIVE':'darkblue',
                                 'PUBLIC':'cyan'})

        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par accident de travail</h1>", unsafe_allow_html=True)
        
        fig = px.pie(st.session_state.Dépense_par_Accident_de_Travail, values='Montant total du rembourement', names="Nature de l'accident de Travail", color="Nature de l'accident de Travail",
             color_discrete_map={
                                 'ACCIDENT DU TRAJET':'darkblue',
                                 'ACCIDENT DU TRAVAIL':'cyan',
                                 'MALADIE PROFESSIONNELLE' : 'lightcyan',
                                 'SANS OBJET' : 'royalblue'})

        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par tranche d'âge</h1>", unsafe_allow_html=True)
        
        fig = px.bar(st.session_state.Dépense_par_tranche_Age, x="Tranche d'âge du Bénéficiaire", y='Montant total du rembourement')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par région</h1>", unsafe_allow_html=True)
        
        fig = px.bar(st.session_state.Dépense_par_Region, x="Région de Résidence du Bénéficiaire", y='Montant total du rembourement')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par sexe</h1>", unsafe_allow_html=True)
        fig = px.pie(st.session_state.Dépense_par_Sex, values='Montant total du rembourement', names='Sex du Bénéficiaire')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h1 style='text-align: center; color: black;font-size:28px;'>Montant du remboursement par type d'assurance</h1>", unsafe_allow_html=True)
        fig = px.pie(st.session_state.Dépense_par_nature_dassurance, values='Montant total du rembourement', names="Type d'assurance")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<h1 style='text-align: center; color: blue;font-size:35px;'>*******</h1>", unsafe_allow_html=True)
        
        st.write('***************************************************')
        
        st.subheader("Si vous le souhaitez vous pouvez télécharger au format excel les résultats des analyses")
        
        date = get_date_of_file(st.session_state.dataframe_1)
        
        df_xlsx1 = to_excel(st.session_state.Dépense_par_secteur, 'Dépense par secteur')
        st.download_button(label=f"📥 Dépense par secteur {date}",
                                data=df_xlsx1 ,
                                file_name= f'Dépense par secteur {date}.xlsx')

        df_xlsx2 = to_excel(st.session_state.Dépense_par_tranche_Age, "Dépense par tranche d'âge")
        st.download_button(label=f"📥 Dépense par tranche d'âge {date}",
                                        data=df_xlsx2 ,
                                        file_name= f"Dépense par tranche d'âge {date}.xlsx")
        
        df_xlsx3 = to_excel(st.session_state.Dépense_par_Accident_de_Travail, "Dépense par accident de travail")
        st.download_button(label=f"📥 Dépense par accident de travail {date}",
                                        data=df_xlsx3 ,
                                        file_name= f"Dépense par accident de travail {date}.xlsx")
        
        df_xlsx4 = to_excel(st.session_state.Dépense_par_Region, "Dépense par région")
        st.download_button(label=f"📥 Dépense par région {date}",
                                        data=df_xlsx4 ,
                                        file_name= f"Dépense par région {date}.xlsx")
        
        df_xlsx5 = to_excel(st.session_state.Dépense_par_Sex, "Dépense par sex")
        st.download_button(label=f"📥 Dépense par sex {date}",
                                        data=df_xlsx5 ,
                                        file_name= f"Dépense par sex {date}.xlsx")
        
        df_xlsx5 = to_excel(st.session_state.Dépense_par_nature_dassurance, "Dépense par nature d'assurance")
        st.download_button(label=f"📥 Dépense par nature d'assurance {date}",
                                        data=df_xlsx5 ,
                                        file_name= f"Dépense par nature d'assurance {date}.xlsx")
        
                
        st.markdown("<h1 style='text-align: center; color: black;font-size:30px;'>END APP</h1>", unsafe_allow_html=True)
        st.write('***************************************************')
        st.markdown("<h1 style='text-align: center; color: green;font-size:50px;'>****************</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: black;font-size:18px;'>Créateur : Maxigor DEKADJEVI</h1>", unsafe_allow_html=True) 
        st.markdown("<h1 style='color: black;font-size:18px;'>Alternant Data Developpeur at Akane</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: black;font-size:18px;'>Profile linkedin : https://www.linkedin.com/in/maxigor-davidson-dekadjevi/ </h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: black;font-size:18px;'>GitHub : https://github.com/maxigorD </h1>", unsafe_allow_html=True)
        
else :
    st.markdown("<h1 style='text-align: center; color: green;font-size:50px;'>****************</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color: black;font-size:18px;'>Créateur : Maxigor DEKADJEVI</h1>", unsafe_allow_html=True) 
    st.markdown("<h1 style='color: black;font-size:18px;'>Alternant Data Developpeur at Akane</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: black;font-size:18px;'>Profile linkedin : https://www.linkedin.com/in/maxigor-davidson-dekadjevi/ </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: black;font-size:18px;'>GitHub : https://github.com/maxigorD </h1>", unsafe_allow_html=True)
    


        
        
        
        
    
