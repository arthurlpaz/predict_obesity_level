import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import locale

# Esconder o menu padrão

hide_streamlit_style = """ 
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden}
        </style>
        """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Obesity Prediction')
st.sidebar.title('Provide the data')

# variaveis
atributos = ['Gender', 'Age', 'Height', 'Weight',
                'family_history_with_overweight', 'FAVC',
                'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH20',
                'SCC', 'FAF', 'CALC', 'MTRANSP']

# prever nível de obesidade "Bobeyesdad"

dict_categorias = {
    'Gender': {'Female': 0, 'Male': 1},
    'family_history_with_overweight': {'no': 0, 'yes': 1},
    'FAVC': {'no': 0, 'yes': 1},
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'SMOKE': {'no': 0, 'yes': 1},
    'SCC': {'no': 0, 'yes': 1},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'MTRANS': {'Public_Transportation': 0,
                'Automobile': 1,
                'Walking': 2,
                'Motorbike': 3,
                'Bike': 4},
}

dict_niveis = {
    'NObeyesdad': {
        'Insufficient_Weight': 0,
        'Normal_Weight': 1,
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I': 4,
        'Obesity_Type_II': 5,
        'Obesity_Type_III ': 6}
}

with st.sidebar:
    with st.form(key='my_form'):
        cat_Gender = st.selectbox('Gender', options=list(dict_categorias['Gender'].keys()))
        Gender = dict_categorias['Gender'][cat_Gender]

        Age = st.number_input('Age', min_value=0, max_value=130, step=1, value=30)

        Height = st.number_input('Height', min_value=1.45, max_value=2.50, value=1.70, step=0.01)

        Weight = st.number_input('Weight', min_value=30.0, max_value=190.0, value=80.0, step=0.01)

        cat_family_with_history_overweight = st.selectbox('family_history_with_overweight', options=list(dict_categorias['family_history_with_overweight'].keys()))
        family_with_history_overweight = dict_categorias['family_history_with_overweight'][cat_family_with_history_overweight]

        cat_FAVC = st.selectbox('Consumption of high-calorie foods', options=list(dict_categorias['FAVC'].keys()))
        FAVC = dict_categorias['FAVC'][cat_FAVC]

        FCVC = st.number_input('Frequency of vegetable consumption', min_value=1, max_value=3, value=1, step=1)

        NCP = st.number_input('Number of main meals', min_value=1, max_value=4, value=3, step=1)

        cat_CAEC = st.selectbox('Food consumption between meals', options=list(dict_categorias['CAEC'].keys()))
        CAEC = dict_categorias['CAEC'][cat_CAEC]

        cat_SMOKE = st.selectbox('Smoke', options=list(dict_categorias['SMOKE'].keys()))
        SMOKE = dict_categorias['SMOKE'][cat_SMOKE]

        CH20 = st.number_input('Daily water consumption (liters)', min_value=1, max_value=3, value=1, step=1)

        cat_SCC = st.selectbox('Monitors calorie consumption', options=list(dict_categorias['SCC'].keys()))
        SCC = dict_categorias['SCC'][cat_SCC]

        FAF = st.number_input('Frequency of physical activity', min_value=0, max_value=3, value=1, step=1)

        cat_CALC = st.selectbox('Alcohol consumption', options=list(dict_categorias['CALC'].keys()))
        CALC = dict_categorias['CALC'][cat_CALC]

        cat_MTRANS = st.selectbox('Type of transport used', options=list(dict_categorias['MTRANS'].keys()))
        MTRANS = dict_categorias['MTRANS'][cat_MTRANS]

        predict_button = st.form_submit_button(label='PREDICT LEVEL')

## Pagina principal
arquivo_modelo = 'modelo.pkl'

with open(arquivo_modelo, 'rb') as f:
    modelo = pickle.load(f)

def mapear_nivel_obesidade(valor):
    valor_arredondado = round(valor.item()) if isinstance(valor, np.ndarray) else round(valor)

    for chave, categoria in dict_niveis['NObeyesdad'].items():
        if categoria == valor_arredondado:
            return chave
    return "UNIDENTIFIED LEVEL"




def previsao_obesidade(modelo, Gender, Age, Height, Weight, family_with_history_overweight,
                        FAVC, FCVC, NCP, CAEC, SMOKE, CH20, SCC, FAF, CALC, MTRANS):
    
    new_X = np.array([Gender, Age, Height, Weight, family_with_history_overweight,
                        FAVC, FCVC, NCP, CAEC, SMOKE, CH20, SCC, FAF, CALC, MTRANS])
    
    nivel_obesidade = modelo.predict(new_X.reshape(1, -1))[0]

    return nivel_obesidade

imagem_obesidade = 'obesidade.jpg'

image = Image.open(imagem_obesidade)
st.image(image, width=400)

if predict_button:
    nivel_obesidade = previsao_obesidade(modelo, Gender, Age, Height, Weight, family_with_history_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH20, SCC, FAF, CALC, MTRANS)

    str_obesidade = mapear_nivel_obesidade(nivel_obesidade)

    st.markdown(f'## Obesity Level: {str_obesidade}')

