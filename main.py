import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Chargement du modèle avec gestion des erreurs
try:
    with open("best_model_RF.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Style de la page
st.set_page_config(page_title="Prédiction ML", page_icon="🤖", layout="centered")

# En-tête avec un encadré stylisé
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'> Prédiction de la Récidive Thyroïdienne </h1>", 
    unsafe_allow_html=True
)
st.write("Utilisez ce modèle pour la prédiction de la Récidive Thyroïdienne .")



# Séparation visuelle
st.markdown("---")

# *📌 Entrée des données*
with st.container():
    st.subheader("📥 Entrez les informations du patient")

row1_col1,row1_col2=st.columns(2)

# Autres variables numériques
with row1_col1:
    Age = st.number_input("Age du patient", min_value=1, step=1)
with row1_col2:    
    Gender = st.selectbox("Sexe du patient", [0, 1], format_func=lambda x: "homme" if x == 1 else "femme")

row2_col1,row2_col2=st.columns(2)
with row2_col1:
   Smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with row2_col2:   
   Hx_Smoking = st.selectbox("HX Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
# Variable binaire sous forme de bouton radio
row3_col1,row3_col2=st.columns(2)
with row3_col1:
   Hx_Radiothreapy = st.selectbox("Hx Radiothreapy", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No") 
with row3_col2:   
   Thyroid_Function=st.selectbox("Thyroid function",[0,1,2,3,4],format_func=lambda x: ['Euthyroid', 'Clinical Hyperthyroidism' ,'Clinical Hypothyroidism',
 'Subclinical Hyperthyroidism' ,'Subclinical Hypothyroidism'][x])
row4_col1,row4_col2=st.columns(2)   
with row4_col1: 
    Physical_Examination=st.selectbox("Physical Examination",[0,1,2,3,4],format_func=lambda x: ['Single nodular goiter-left', 'Multinodular goiter',
 'Single nodular goiter-right', 'Normal' ,'Diffuse goiter'][x])  
with row4_col2:    
    Adenopathy=st.selectbox("Adenopathy",[0,1,2,3,4,5],format_func=lambda x:   ['No', 'Right', 'Extensive', 'Left' ,'Bilateral' ,'Posterior'][x])
row5_col1,row5_col2=st.columns(2)      
with row5_col1:   
   Pathology=st.selectbox("Pathology",[0,1,2,3],format_func=lambda x:['Micropapillary', 'Papillary', 'Follicular' ,'Hurthel cell'][x])
with row5_col2:   
   Focality=st.selectbox("Focality", [0, 1], format_func=lambda x: 'Uni-Focal'  if x == 1 else 'Multi-Focal')
row6_col1,row6_col2=st.columns(2)   
with row6_col1:   
   Risk=st.selectbox("Risk",[0,1,2],format_func=lambda x:  ['Low' ,'Intermediate' ,'High'][x])
with row6_col2:   
   T=st.selectbox("T",[0,1,2,3,4,5,6],format_func=lambda x: ['T1a', 'T1b' ,'T2', 'T3a' ,'T3b' ,'T4a', 'T4b'][x])
row7_col1,row7_col2=st.columns(2)   
with row7_col1: 
    N=st.selectbox("N",[0,1,2,],format_func=lambda x: ['N0', 'N1b', 'N1a'][x])  
with row7_col2:    
    M=st.selectbox("M", [0, 1], format_func=lambda x: 'M0'  if x == 1 else 'M1')
row8_col1,row8_col2=st.columns(2)     
with row8_col1:   
    Stage=st.selectbox("Stage",[0,1,2,3,4],format_func=lambda x:      ['I', 'II' ,'IVB' ,'III' ,'IVA'][x])
with row8_col2:    
    Response=st.selectbox("Response",[0,1,2,3],format_func=lambda x:['Indeterminate', 'Excellent' ,'Structural Incomplete',
 'Biochemical Incomplete'][x])
  
# Séparation visuelle
st.markdown("---")
  

# Bouton de prédiction
if st.button("🔮 Prédire", use_container_width=True):
    try:
        features = np.array([[Age, Gender, Smoking, Hx_Smoking, Hx_Radiothreapy,
   Thyroid_Function, Physical_Examination, Adenopathy, Pathology,
   Focality, Risk, T, N, M, Stage, Response]])
        prediction = model.predict(features)[0]
        
        

        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(features)[0]
            st.subheader("📊 Résultats de la prédiction")
            st.success(f"✅ Valeur prédite : *{prediction}*")
             # Affichage des probabilités sous forme de barres
            st.write("### 🔢 Probabilités des classes :")
            prob_df = pd.DataFrame({"Classe": range(len(prediction_proba)), "Probabilité": prediction_proba})
            st.bar_chart(prob_df.set_index("Classe"))

           # st.dataframe(prob_df)
        else:
            st.warning("⚠️ Le modèle ne prend pas en charge les probabilités de classe.")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
   
# Pied de page
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>✨ Développé avec Streamlit</p>", unsafe_allow_html=True)
