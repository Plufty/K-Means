import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[:, :-1].values  #A última coluna foi removida pois era a classe    
    normalized_data = (data - data.mean(axis=0)) / data.std(axis=0) # Normalização dos dados
    return normalized_data

def input_int(message): #Garante que o K informado pelo usuário sempre será um número inteiro
    while True:
        try:
            value = int(input(message))
            return value
        except ValueError:
            print("Por favor, insira um número inteiro válido.")

def initialize_centroids(data, K):
    np.random.seed(42) #Seed para que seja possível replicar os resultados (42 a resposta para qualquer pergunta no universo rs.)
    centroids = data[np.random.choice(data.shape[0], K, replace=False)] #inicializando o centroide de forma aleatória
    return centroids


class KMeans:
    def __init__(self, n_clusters, max_iter=100): #O laço de repetição executado até a convergência do algoritmo terá limite máximo de 100 iterações
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, plot_progress=False): #Execução do algoritmo
        self.centroids = initialize_centroids(X, self.n_clusters)
        
        if plot_progress: #Gerando o pdfpara salvar a execução
            fig, ax = plt.subplots()
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pp = PdfPages(f'kmeans_progress_{current_time}.pdf')
        
        for i in range(self.max_iter):
            #Verificar distância e colocar no centróide mais próximo
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2)) #i.	Medida de proximidade: distância Euclidiana
            labels = np.argmin(distances, axis=0) #Casos de empate na associação de um elemento ao centróide: escolher o primeiro.
            
            #Atualizar os Centroides
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            #Verificar se alcançou a convergência
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
                
            if plot_progress: #Plotar cada iteração para observar a execução do algoritmo
                ax.clear()
                ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
                ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
                ax.set_title(f"Iteração {i+1}")
                pp.savefig(fig)
        
        if plot_progress:
            pp.close()
                
        self.labels_ = labels

# Processamento dos dados da base Iris
file_path = "iris.data" 
iris_df = preprocess_data(file_path)

# Definindo o número de clusters
k = input_int("Digite a quantidade de clusters para o agrupamento:")

# Execução do Algoritmo
kmeans = KMeans(n_clusters=k)
kmeans.fit(iris_df, plot_progress=True)

#Salvando a Execução
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_csv_file = f"resultado_kmeans_iris_{current_time}.csv"
result_pdf_file = f"kmeans_progress_{current_time}.pdf"

result_df = pd.read_csv(file_path)
header_row = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
result_df.columns = header_row
result_df['cluster'] = kmeans.labels_
result_df.to_csv(result_csv_file, index=False, header=True)

print(f"Resultados salvos em '{result_csv_file}' e '{result_pdf_file}'")
