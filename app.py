import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.metrics import accuracy_score, precision_score ,confusion_matrix, recall_score,classification_report,f1_score

with open("best_model.pkl", "rb") as file:
        model=pickle.load(file)
        

# Titre de l'application
st.title("🌟 Prédiction de la Récidive Thyroïdienne")
# Description de l'application
st.markdown("""
Ce projet de machine learning vise à analyser la récidive des maladies thyroïdiennes, 
            en se basant sur un ensemble de données médicales comprenant des informations sur les antécédents des patients, les résultats des examens physiques, la pathologie et la réponse au traitement.
             L’objectif est de prédire la probabilité qu’une maladie thyroïdienne réapparaisse (récidive) chez les patients en utilisant des algorithmes de régression logistique,
             tout en optimisant les hyperparamètres pour améliorer les performances du modèle. 
            Ce projet se concentre également sur l’analyse des sous-groupes de patients et l’évaluation de l’équité du modèle pour éviter les biais"""
       )

# Nom de l'auteur dans la barre latérale
st.markdown("### 📌 Auteur : Ibrahima FALL")


st.sidebar.title("💡 Importation du jeux de donnees")
df = pd.read_csv('Thyroid_Diff.csv')
if st.sidebar.checkbox('afficher les donnees brutes', False):
  st.subheader("Jeu de donnees Thyroid_Diff ' : Echantillon des 15 premieres lignes")
  st.write(df.head(15))

st.sidebar.title("💡 Analyse et Exploration  du jeux de donnees") 
if st.sidebar.checkbox('Types de variables', False):
    st.subheader(" 🔎 Visualisation de la repartion des types de donnees")
    fig, ax = plt.subplots()
    df.dtypes.value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
    st.pyplot(fig)   
if st.sidebar.checkbox('Valeurs manquantes', False):
    st.subheader("🔎 Visualisation des valeurs manquantes")
    fig, ax = plt.subplots()
    sns.heatmap(df.isna(), cbar=False, ax=ax)
    st.pyplot(fig) 
if st.sidebar.checkbox('Distribution variable AGE', False):
    st.subheader("🔎 Histogramme de la variable AGE")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax)
    st.pyplot(fig)    
no_recurred=df[df['Recurred']=='No']
yes_recurred=df[df['Recurred']=='Yes']
df['Age_cat']=pd.cut(df['Age'],bins=[10,20,30,40,50,60,70,80])
df1=df.copy()
df1['Age']=df1['Age_cat']
if st.sidebar.checkbox('Relation Target/AGE', False):
    st.subheader("🔎 Relation Target/AGE")
    fig, ax = plt.subplots()
    sns.countplot(x='Age',hue='Recurred',data=df1,ax=ax)
    st.pyplot(fig)

cat=(df.drop(['Recurred'],axis=1).select_dtypes('object'))
target_var = 'Recurred' 
if st.sidebar.checkbox('Relation Target/Variable categorielle', False):
   if target_var not in df.columns: 
        st.error(f"La colonne '{target_var}' n'existe pas dans le dataset.")
   else: 
     # Sélectionner une variable à comparer avec 'recurred' 
     other_vars = [col for col in cat.columns if col != target_var]
     selected_var = st.selectbox("Sélectionnez une variable pour l'analyse de contingence :", other_vars)

     # Construire le tableau de contingence
     contingency_table = pd.crosstab(df[target_var], df[selected_var])
     st.write("### Tableau de contingence")
     st.write(contingency_table)

# Affichage sous forme de heatmap
fig, ax = plt.subplots()
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig) 

LE=LabelEncoder()
for col in df.columns:
    df[col]=LE.fit_transform(df[col])
df=df.drop('Age_cat',axis=1)
X=df.drop('Recurred',axis=1)
y=df['Recurred']
scaler=StandardScaler()
X_norm=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2,random_state=0)
y_pred_SVC = model.predict(X_test)

def metrics(y_test,y_pred):
    results={
    "Accuracy": accuracy_score(y_test,y_pred),
    "Recall":recall_score(y_test,y_pred),
    "Precision":precision_score(y_test,y_pred),
    "F1_Score":f1_score(y_test,y_pred),
     "Classification Report":classification_report(y_test,y_pred,output_dict=True)
    }
    return results

# Affichage des métriques du modèle
st.sidebar.title("💡 Mesure de performance")
metrics_result = None  # Initialisation pour éviter les erreurs

if st.sidebar.checkbox('Evaluation du modele', False):
    st.write("### 📊 Performance du Modèle")
    metrics_result = metrics(y_test, y_pred_SVC)

# Menu déroulant pour choisir une métrique
selected_metric = st.selectbox(
    "🔍 Sélectionnez une métrique à afficher :",
    ["Accuracy", "Recall", "Precision", "F1_Score", "Classification Report"]
)

# Vérification que metrics_result est bien défini avant de l'utiliser
if metrics_result:
    if selected_metric in ["Accuracy", "Recall", "Precision", "F1_Score"]:
        st.write(f"🔹 **{selected_metric} :** {metrics_result[selected_metric]:.3f}")
    elif selected_metric == "Classification Report":
        st.write("🔹 **Classification Report:**")
        st.json(metrics_result["Classification Report"])
else:
    st.warning("⚠️ Veuillez cocher 'Évaluation du modèle' pour voir les métriques.")



if st.sidebar.checkbox('Matrice de confusion', False):
   # Matrice de confusion
    st.write("### 🎯 Matrice de Confusion")
    cm=confusion_matrix(y_test, y_pred_SVC)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non Récidive", "Récidive"], yticklabels=["Non Récidive", "Récidive"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    st.pyplot(fig)  

if st.sidebar.checkbox(' Afficher la Learning Curve', False):  
    st.write("### 📈 Learning Curve")
    N, train_score, val_score=learning_curve(model,X_train,y_train,cv=5,
                            scoring='f1',train_sizes=np.linspace(0.1,1,10))
    fig, ax = plt.subplots()
    plt.plot(N,train_score.mean(axis=1),label='train score ')
    plt.plot(N,val_score.mean(axis=1),label='validation score ' ,linestyle='dashed')
    
    plt.xlabel("Training Size")
    plt.ylabel(f1_score)
    plt.title("Learning Curves ")
    plt.legend()
    plt.show() 
    st.pyplot(fig)   




st.sidebar.title("💡 Informations sur le patient")

# Autres variables numériques
Age = st.sidebar.number_input("Age du patient", value=10, step=1)

# Variable binaire sous forme de selectbox
Gender = st.sidebar.selectbox("Sexe du patient", [0, 1], format_func=lambda x: "homme" if x == 1 else "femme")

# Variable binaire sous forme de bouton radio
Smoking = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Hx_Smoking = st.sidebar.selectbox("HX Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Hx_Radiothreapy = st.sidebar.selectbox("Hx Radiothreapy", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Thyroid_Function=st.sidebar.radio("Thyroid function",[0,1,2,3,4],format_func=lambda x: ['Euthyroid', 'Clinical Hyperthyroidism' ,'Clinical Hypothyroidism',
 'Subclinical Hyperthyroidism' ,'Subclinical Hypothyroidism'][x])
Physical_Examination=st.sidebar.radio("Physical Examination",[0,1,2,3,4],format_func=lambda x: ['Single nodular goiter-left', 'Multinodular goiter',
 'Single nodular goiter-right', 'Normal' ,'Diffuse goiter'][x])
Adenopathy=st.sidebar.radio("Adenopathy",[0,1,2,3,4,5],format_func=lambda x:   ['No', 'Right', 'Extensive', 'Left' ,'Bilateral' ,'Posterior'][x])
Pathology=st.sidebar.radio("Pathology",[0,1,2,3],format_func=lambda x:['Micropapillary', 'Papillary', 'Follicular' ,'Hurthel cell'][x])
Focality=st.sidebar.selectbox("Focality", [0, 1], format_func=lambda x: 'Uni-Focal'  if x == 1 else 'Multi-Focal')
Risk=st.sidebar.radio("Risk",[0,1,2],format_func=lambda x:  ['Low' ,'Intermediate' ,'High'][x])
T=st.sidebar.radio("T",[0,1,2,3,4,5,6],format_func=lambda x: ['T1a', 'T1b' ,'T2', 'T3a' ,'T3b' ,'T4a', 'T4b'][x])
N=st.sidebar.radio("N",[0,1,2,],format_func=lambda x: ['N0', 'N1b', 'N1a'][x])
M=st.sidebar.selectbox("M", [0, 1], format_func=lambda x: 'M0'  if x == 1 else 'M1')
Stage=st.sidebar.radio("Stage",[0,1,2,3,4],format_func=lambda x:      ['I', 'II' ,'IVB' ,'III' ,'IVA'][x])
Response=st.sidebar.radio("Response",[0,1,2,3],format_func=lambda x:['Indeterminate', 'Excellent' ,'Structural Incomplete',
 'Biochemical Incomplete'][x])


# Bouton pour lancer la prédiction
if st.sidebar.button("🔍 Prédire"):
    features = np.array([[Age, Gender, Smoking, Hx_Smoking, Hx_Radiothreapy,
   Thyroid_Function, Physical_Examination, Adenopathy, Pathology,
   Focality, Risk, T, N, M, Stage, Response]])
    prediction = model.predict(features)
     # Convertir la prédiction en modalités (par exemple "Oui" pour 0, "Non" pour 1)
    st.write("### Prediction")
    result = "Yes" if prediction[0] == 1 else "No"
    st.success(f"✨ Valeur predite: {prediction[0]} ({result})")