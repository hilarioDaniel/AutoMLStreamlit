import streamlit as st
import pandas as pd
import pycaret
from pycaret.classification import setup, compare_models, pull, save_model
import os 

# Título e descrição da página inicial
st.title("Analise dos dados usando Modelos de Machine Learning")
st.subheader("Treinamento de modelos para a classificacao de pacientes com a DP usando registros de voz")

# Buscando o arquivo enviado para análise
@st.cache_data
def carregar_dataframe():
    if os.path.exists("dados.csv"):
        df_data = pd.read_csv("dados.csv", index_col=None)
        st.session_state["data"] = df_data
        return df_data

df = carregar_dataframe()

## Variaveis a serem apresentadas a funcao Pycaret
with st.form("Basic Settings"):
    variavelTarget = st.selectbox("Selecione a variavel alvo: ", df.columns, index = 2)
    foldsNumber = st.number_input('Numero de Folds', min_value = 2, max_value = 10, value = 5)
    testing = st.slider('Escolha o percentual para os dados de teste:', min_value=0.05, max_value=1.00, step=0.05, value = 0.20)
    training = (1-testing)
    # Botao para dar inicio a configuracao e treinamento
    trainingModels = st.form_submit_button("Treinar os Modelos")
    
    if trainingModels:
        ## Pycaret
        # Funcao Setup
        setup(data = df, target = variavelTarget, session_id=1245, 
            normalize= True, remove_multicollinearity=True,
            multicollinearity_threshold=0.85, fold = foldsNumber, train_size=training)
        setup_df = pull()
        st.info("Parametros utilizados para treinar os modelos")
        st.dataframe(setup_df, use_container_width=True)

        # Modelos Treinamento
        best_model = compare_models()
        compare_df = pull()
        st.info("Dataframe com o resultado para os modelos treinados")
        st.dataframe(compare_df, use_container_width=True)

        st.info("Os hiperparametros para o melhor modelo sao dados a seguir:")
        st.json(best_model.get_params())