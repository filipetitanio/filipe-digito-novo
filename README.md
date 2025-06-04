# filipe-digito-novo
Reconhecimento de Dígitos Manuscritos - Projeto MNIST
Descrição
Este projeto é uma aplicação de aprendizado de máquina desenvolvida para a disciplina de Engenharia do Conhecimento na UMN-ISPH. Utiliza o dataset MNIST, com 70.000 imagens de dígitos manuscritos (0-9), para treinar um modelo SVM (Máquina de Vetores de Suporte) com kernel RBF, alcançando uma acurácia de 97%. A aplicação, construída com Streamlit, permite:

Desenhar dígitos em um canvas interativo.
Carregar imagens de dígitos para previsão.
Visualizar a matriz de confusão e exemplos de previsões.

Objetivo
Classificar dígitos manuscritos com alta precisão, demonstrando a aplicação de técnicas de aprendizado de máquina supervisionado em visão computacional.
Tecnologias Utilizadas

Python: Linguagem principal.
Scikit-learn: Treinamento do modelo SVM.
Streamlit: Interface web interativa.
Matplotlib/Seaborn: Geração de gráficos.
Pillow: Manipulação de imagens.
Git/GitHub: Controle de versão e hospedagem.

Pré-requisitos

Python 3.11+
Dependências listadas em requirements.txt.

Instalação

Clone o repositório:git clone https://github.com/filipetitanio/filipe-digito-novo.git
cd filipe-digito-novo


Crie um ambiente virtual:python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows


Instale as dependências:pip install -r requirements.txt



Uso

Execute a aplicação:streamlit run app.py


Acesse a interface no navegador (ex.: http://localhost:8501).
Navegue pelas abas:
Início: Visão geral do projeto.
Prever Dígito: Desenhe ou carregue imagens 28x28 para prever dígitos.
Resultados: Veja a matriz de confusão e exemplos de previsões.
Sobre o Projeto: Detalhes do desenvolvimento.
Sobre Mim: Informações do autor.



Deploy
A aplicação está hospedada no Streamlit Cloud: https://filipe-digito-novo.streamlit.app.
Estrutura do Repositório

app.py: Código principal da aplicação Streamlit.
style.css: Estilização personalizada.
requirements.txt: Dependências do projeto.
mnist_model_final_rbf.pkl: Modelo treinado (97% de acurácia).
mnist_test_subset.csv: Subconjunto de teste para visualizações.
confusion_matrix.png: Matriz de confusão.
predictions.png: Exemplos de previsões.
new_project_image.png: Imagem do projeto.
filipe_tchivela.jpg: Foto do autor.

Créditos

Autor: Filipe Tchivela (2022142110), estudante de Ciência da Computação na UMN-ISPH.
Orientador: Prof. Abel Zacarias.
Data: Junho de 2025.

Licença
Este projeto é de uso acadêmico e não possui licença comercial.

Última atualização: 04/06/2025, 01:19 AM WAT
