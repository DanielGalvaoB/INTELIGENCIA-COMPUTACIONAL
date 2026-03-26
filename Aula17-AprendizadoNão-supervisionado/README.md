# Descobrindo Padrões com Aprendizado Não-Supervisionado

Este repositório contém a implementação de uma atividade prática acadêmica focada em **Aprendizado Não-Supervisionado**. O objetivo do projeto é aplicar técnicas de clusterização para descobrir padrões em um conjunto de dados não rotulado e, posteriormente, validar a eficácia do agrupamento.

## 🎯 Sobre o Projeto

O experimento utiliza o clássico **Dataset Iris**. Para simular um cenário não supervisionado, as classes (espécies das flores) foram "escondidas" do algoritmo. 

Foi utilizado o algoritmo **K-means** para tentar agrupar as flores com base apenas nas medidas de suas pétalas e sépalas. O código testa a divisão dos dados em $k=2$, $k=3$ e $k=4$ grupos. Ao final, o gabarito é revelado para o cenário $k=3$, e o código calcula uma Matriz de Confusão e a Porcentagem de Correspondência (Acurácia) para verificar se o algoritmo conseguiu redescobrir as espécies reais sozinho.

## 📂 Estrutura de Arquivos

* `codigo.py`: Script principal contendo o carregamento de dados, o modelo K-means, a geração dos gráficos e o cálculo das métricas.
* `relatorio.pdf`: Relatório completo com a fundamentação teórica, metodologia, resultados experimentais e discussão.
* `clusters_kmeans.png`: Gráficos de dispersão (scatter plots) comparando os clusters gerados para $k=2$, $k=3$ e $k=4$.
* `requirements.txt`: Lista de dependências e bibliotecas Python necessárias.

## 🚀 Como Executar

* `python -m pip install -r requirements.txt`\
* `python codigo.py`