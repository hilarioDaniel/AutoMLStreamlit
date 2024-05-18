## Importando as bibliotecas principais
# import streamlit as st
# import pandas as pd
# import os
# # Importando profilling
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report

# # Título e descrição da página inicial
# st.title("Analise Exploratoria dos Dados (EDA)")
# st.subheader("EDA - Automatizada")

# if os.path.exists("dados.csv"):
#     df = pd.read_csv("dados.csv", index_col=None)

# # Análise de dados usando o pandas profiling
# relatorio = df.profile_report()
# st_profile_report(relatorio)

import pandas as pd
import pandas_profiling
import streamlit as st
from pandas_profiling import ProfileReport

from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
pr = df.profile_report()

st_profile_report(pr)