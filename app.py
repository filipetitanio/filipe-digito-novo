import streamlit as st
import numpy as np
from PIL import Image
import joblib
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import plotly.express as px

# Configura página
st.set_page_config(page_title="Reconhecimento de Dígitos - Filipe Tchivela", layout="wide")

# Aplica CSS personalizado
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Arquivo style.css não encontrado. Estilização padrão será aplicada.")

# Verifica modelo
model_path = 'mnist_model_final_rbf.pkl'
if not os.path.exists(model_path):
    st.error(f"Erro: Arquivo '{model_path}' não encontrado.")
    st.stop()

# Carrega modelo
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {str(e)}")
    st.stop()

# Carrega subconjunto de teste
data_path = 'mnist_test_subset.csv'
if os.path.exists(data_path):
    test_data = pd.read_csv(data_path)
    X_test = test_data.drop('label', axis=1).values / 255.0
    y_test = test_data['label'].values
else:
    st.warning(f"Arquivo '{data_path}' não encontrado. Visualizações limitadas.")
    X_test = None
    y_test = None

# Menu lateral
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione uma página:", ["Início", "Prever Dígito", "Resultados", "Sobre o Projeto", "Sobre Mim"])

# Página Início
if page == "Início":
    st.title("Reconhecimento de Dígitos Manuscritos")
    st.markdown("""
    Bem-vindo à aplicação de reconhecimento de dígitos manuscritos! Este projeto utiliza um modelo SVM com kernel RBF, treinado no dataset MNIST, alcançando **97% de acurácia**. Navegue pelo menu para desenhar dígitos, carregar imagens, visualizar resultados ou saber mais sobre o projeto.
    """)
    try:
        st.image("imagem_do_projecto.jpg", caption="Visão geral do projeto", width=600)
    except FileNotFoundError:
        st.warning("Imagem do projeto não encontrada.")

# Página Prever Dígito
elif page == "Prever Dígito":
    st.title("Prever Dígito")
    st.markdown("Desenhe um dígito ou carregue uma imagem para testar o modelo.")

    # Abas para Desenhar e Carregar
    tab1, tab2 = st.tabs(["Desenhar Dígito", "Carregar Imagem"])

    with tab1:
        st.subheader("Desenhar Dígito")
        st.markdown("Desenhe um dígito com pincel branco sobre fundo preto.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=150,
            width=150,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image).reshape(1, -1) / 255.0
            prediction = model.predict(image_array)[0]
            try:
                probabilities = model.predict_proba(image_array)[0]
            except AttributeError:
                probabilities = np.ones(10) / 10  # Fallback se predict_proba não disponível

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Dígito Desenhado", width=100)
                st.markdown(f"**Dígito Previsto: {prediction}**")
            with col2:
                fig = px.bar(x=range(10), y=probabilities, labels={'x': 'Dígito', 'y': 'Probabilidade'},
                             title="Probabilidades de Classe", color=probabilities, color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Carregar Imagem")
        st.markdown("Carregue uma imagem 28x28 (PNG/JPG) em escala de cinza.")
        uploaded_file = st.file_uploader("Escolha a imagem", type=["png", "jpg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image).reshape(1, -1) / 255.0
            prediction = model.predict(image_array)[0]
            try:
                probabilities = model.predict_proba(image_array)[0]
            except AttributeError:
                probabilities = np.ones(10) / 10  # Fallback

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Imagem Carregada", width=100)
                st.markdown(f"**Dígito Previsto: {prediction}**")
            with col2:
                fig = px.bar(x=range(10), y=probabilities, labels={'x': 'Dígito', 'y': 'Probabilidade'},
                             title="Probabilidades de Classe", color=probabilities, color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)

# Página Resultados
elif page == "Resultados":
    st.title("Resultados do Modelo")
    st.markdown("Explore a matriz de confusão e exemplos de previsões do modelo SVM (97% de acurácia).")

    if X_test is not None and y_test is not None:
        # Matriz de confusão
        st.subheader("Matriz de Confusão")
        try:
            st.image("confusion_matrix.png", caption="Matriz de Confusão", width=600)
        except FileNotFoundError:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel('Rótulo Previsto')
            ax.set_ylabel('Rótulo Verdadeiro')
            ax.set_title('Matriz de Confusão')
            st.pyplot(fig)

        # Exemplos de previsões
        st.subheader("Exemplos de Previsões")
        st.markdown("Cinco imagens do conjunto de teste com rótulos verdadeiros e previstos:")
        try:
            st.image("predictions.png", caption="Exemplos de Previsões", width=600)
        except FileNotFoundError:
            cols = st.columns(5)
            for i in range(5):
                image = X_test[i].reshape(28, 28)
                true_label = y_test[i]
                pred_label = model.predict(X_test[i].reshape(1, -1))[0]
                with cols[i]:
                    st.image(image, caption=f"Verd.: {true_label}\nPrev.: {pred_label}", width=80)

# Página Sobre o Projeto
elif page == "Sobre o Projeto":
    st.title("Sobre o Projeto")
    st.markdown("""
    Este projeto foi desenvolvido para a disciplina de Engenharia do Conhecimento na UFMG. Utiliza o dataset MNIST, com 70.000 imagens de dígitos manuscritos, para treinar um modelo SVM com kernel RBF, alcançando **97% de acurácia**. A aplicação permite:
    - Desenhar dígitos em um canvas interativo.
    - Carregar imagens para previsão.
    - Visualizar resultados como matriz de confusão e exemplos de previsões.
    """)
    try:
        st.image("project_image.png", caption="Reconhecimento de Dígitos", width=600)
    except FileNotFoundError:
        st.warning("Imagem do projeto não encontrada.")

# Página Sobre Mim
elif page == "Sobre Mim":
    st.title("Sobre Mim")
    st.markdown("""
    - **Nome**: Filipe Tchivela
    - **Número de Estudante**: 202346
    - **Curso**: Ciência da Informação
    - **Instituição**: UFMG
    - **Contacto**: +946715031
    - **E-mail**: filipetischio@gmail.com
    """)

# Criar style.css
%%writefile style.css
/* Fundo estilizado com gradiente */
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: #ffffff;
    font-family: 'Arial', sans-serif;
}

/* Estiliza o título principal */
h1 {
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    font-size: 2.5em;
    text-align: center;
}

/* Estiliza cabeçalhos */
h2, h3 {
    color: #e0e0e0;
}

/* Estiliza a barra lateral */
.stSidebar {
    background-color: #0f1e3d;
    padding: 20px;
}

.stSidebar .stRadio > label {
    color: #ffffff;
    font-weight: bold;
}

/* Estiliza botões */
button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}

button:hover {
    background-color: #45a049;
}

/* Estiliza abas */
.stTabs button {
    background-color: #1e3c72;
    color: #ffffff;
    border-radius: 5px 5px 0 0;
}

.stTabs button:hover {
    background-color: #2a5298;
}

/* Estiliza o canvas */
canvas {
    border: 2px solid #ffffff;
    border-radius: 5px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* Estiliza imagens */
img {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}

/* Texto destacado */
strong {
    color: #4CAF50;
}

# Criar requirements.txt
%%writefile requirements.txt
streamlit==1.36.0
numpy==1.26.4
pillow==10.4.0
joblib==1.4.2
scikit-learn==1.5.0
streamlit-drawable-canvas==0.9.3
pandas==2.2.2
seaborn==0.13.2
matplotlib==3.9.0
plotly==5.22.0

# Criar project_image.png
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.text(0.5, 0.5, "Reconhecimento de Dígitos\nMNIST", fontsize=20, ha='center', va='center')
plt.axis('off')
plt.savefig('project_image.png')
