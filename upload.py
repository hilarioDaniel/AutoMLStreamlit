import streamlit as st
import pandas as pd

# Título e descrição da página de envio do arquivo
st.markdown("<h1 style='text-align: center;'> Upload dos dados para modelagem </h1>", unsafe_allow_html=True)

df = None # Inicializando o df como None

with st.form("Upload"):
    st.write('Upload do arquivo no formato csv')
    file = st.file_uploader("Upload dos seus dados aqui")
    send_file = st.form_submit_button("Exibir Dataframe")
    
    if send_file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dados.csv", index = None)
        
        st.markdown("<h3 style='text-align:center;'> Dataframe para a modelagem</h3>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=False)
