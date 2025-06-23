import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly.express as px
import warnings
import time
warnings.filterwarnings('ignore')

CSV_FILE_PATH = r"c:\Users\Sodiq Abdullah\OneDrive\Documents\KULIAH\Semester 6\Kecerdasan Komputasional\Pertemuan 6\Praktikum\TestbedSatJun12Flows.csv"

try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("Dataset berhasil dimuat.")
    df = df.head(1000)
    print(f"Dataset dibatasi menjadi {len(df)} baris untuk pengujian.")
except FileNotFoundError:
    print(f"File {CSV_FILE_PATH} tidak ditemukan. Pastikan path file benar.")
    exit(1)

def preprocess_data(df):
    numerical_cols = ['totalSourceBytes', 'totalDestinationBytes', 'totalDestinationPackets', 'totalSourcePackets']
    categorical_cols = ['appName', 'direction', 'protocolName']
    
    df[numerical_cols] = df[numerical_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna('Tidak Diketahui')
    
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, numerical_cols + categorical_cols

class GeneticAlgorithm:
    def __init__(self, features, population_size=10, generations=5, mutation_rate=0.01):
        self.features = features
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
    
    def initialize_population(self):
        return np.random.choice([0, 1], size=(self.population_size, len(self.features)))
    
    def fitness_function(self, individual, X, y):
        start_time = time.time()
        selected_features = [self.features[i] for i in range(len(individual)) if individual[i] == 1]
        if not selected_features:
            return 0
        X_subset = X[selected_features]
        model = IsolationForest(contamination=0.1, random_state=42, n_jobs=1, n_estimators=50)
        model.fit(X_subset)
        scores = -model.decision_function(X_subset)
        elapsed_time = time.time() - start_time
        print(f"Fitness function selesai dalam {elapsed_time:.2f} detik untuk fitur: {selected_features}")
        return np.mean(scores)
    
    def select_parents(self, fitness_scores):
        probabilities = fitness_scores / fitness_scores.sum()
        parents_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        return self.population[parents_indices]
    
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(self.features) - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child
    
    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual
    
    def run(self, X, y):
        for generation in range(self.generations):
            print(f"Memproses generasi {generation + 1}/{self.generations}...")
            fitness_scores = np.array([self.fitness_function(ind, X, y) for ind in self.population])
            parents = self.select_parents(fitness_scores)
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                next_population.append(self.mutate(child1))
                next_population.append(self.mutate(child2))
            self.population = np.array(next_population[:self.population_size])
        best_individual = self.population[np.argmax([self.fitness_function(ind, X, y) for ind in self.population])]
        return [self.features[i] for i in range(len(best_individual)) if best_individual[i] == 1]

def detect_anomalies(df, selected_features):
    if not selected_features:
        print("Peringatan: Tidak ada fitur yang dipilih. Menggunakan semua fitur numerik default.")
        selected_features = ['totalSourceBytes', 'totalDestinationBytes', 'totalSourcePackets', 'totalDestinationPackets']
    X = df[selected_features]
    model = IsolationForest(contamination=0.1, random_state=42, n_jobs=1, n_estimators=50)
    model.fit(X)
    df['anomaly_score'] = -model.decision_function(X)
    df['is_anomaly'] = model.predict(X)
    df['is_anomaly'] = df['is_anomaly'].map({1: 'Normal', -1: 'Anomali'})
    return df

def visualize_anomalies(df, selected_features, output_pdf, output_html):
    with PdfPages(output_pdf) as pdf:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df[df['is_anomaly'] == 'Normal'], x='anomaly_score', bins=30, color='blue', alpha=0.5, label='Normal', kde=False)
        sns.histplot(data=df[df['is_anomaly'] == 'Anomali'], x='anomaly_score', bins=30, color='red', alpha=0.5, label='Anomali', kde=False)
        plt.title('Distribusi Skor Anomali', fontsize=14)
        plt.xlabel('Skor Anomali', fontsize=12)
        plt.ylabel('Jumlah', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 6))
        anomaly_counts = df['is_anomaly'].value_counts()
        sns.barplot(x=anomaly_counts.index, y=anomaly_counts.values, palette=['blue', 'red'])
        plt.title('Jumlah Data Normal vs Anomali', fontsize=14)
        plt.xlabel('Status', fontsize=12)
        plt.ylabel('Jumlah', fontsize=12)
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()

    numeric_features = [f for f in selected_features if f in ['totalSourceBytes', 'totalDestinationBytes', 'totalDestinationPackets', 'totalSourcePackets']]
    if len(numeric_features) >= 2:
        x_feature = numeric_features[0]
        y_feature = numeric_features[1]
    else:
        x_feature = 'totalSourceBytes'
        y_feature = 'totalDestinationBytes'

    fig = px.scatter(df, x=x_feature, y=y_feature, color='is_anomaly',
                     color_discrete_map={'Normal': 'blue', 'Anomali': 'red'},
                     title=f'Scatter Plot: {x_feature} vs {y_feature}',
                     labels={x_feature: x_feature, y_feature: y_feature},
                     hover_data=['anomaly_score'])
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(showlegend=True)
    fig.write_html(output_html)
    print(f"Scatter plot disimpan ke '{output_html}'.")

def main():
    start_time_total = time.time()
    print("Memulai proses deteksi anomali dengan berbagai uji coba...")
    
    df_processed, features = preprocess_data(df.copy())
    
    # Eksperimen 1: Variasi Ukuran Populasi
    print("\n=== Eksperimen 1: Variasi Ukuran Populasi ===")
    population_sizes = [10, 30, 50]
    generations = [5]
    mutation_rates = [0.01]
    for pop_size in population_sizes:
        start_time = time.time()
        print(f"\nUji Coba - Populasi: {pop_size}, Generasi: {generations[0]}, Mutasi: {mutation_rates[0]}")
        ga = GeneticAlgorithm(features=features, population_size=pop_size, generations=generations[0], mutation_rate=mutation_rates[0])
        selected_features = ga.run(df_processed[features], df_processed.get('Label', pd.Series(['Normal'] * len(df_processed))))
        print(f"Fitur Terpilih: {selected_features}")
        df_results = detect_anomalies(df_processed, selected_features)
        output_csv = f'results_exp1_pop{pop_size}_gen{generations[0]}_mut{mutation_rates[0]}.csv'
        output_pdf = f'report_exp1_pop{pop_size}_gen{generations[0]}_mut{mutation_rates[0]}.pdf'
        output_html = f'scatter_exp1_pop{pop_size}_gen{generations[0]}_mut{mutation_rates[0]}.html'
        df_results.to_csv(output_csv, index=False)
        print(f"Hasil disimpan ke '{output_csv}'")
        visualize_anomalies(df_results, selected_features, output_pdf, output_html)
        elapsed_time = time.time() - start_time
        print(f"Uji coba selesai dalam {elapsed_time:.2f} detik.")
    
    # Eksperimen 2: Variasi Jumlah Generasi
    print("\n=== Eksperimen 2: Variasi Jumlah Generasi ===")
    population_sizes = [10]
    generations = [5, 15, 25]
    mutation_rates = [0.01]
    for gen in generations:
        start_time = time.time()
        print(f"\nUji Coba - Populasi: {population_sizes[0]}, Generasi: {gen}, Mutasi: {mutation_rates[0]}")
        ga = GeneticAlgorithm(features=features, population_size=population_sizes[0], generations=gen, mutation_rate=mutation_rates[0])
        selected_features = ga.run(df_processed[features], df_processed.get('Label', pd.Series(['Normal'] * len(df_processed))))
        print(f"Fitur Terpilih: {selected_features}")
        df_results = detect_anomalies(df_processed, selected_features)
        output_csv = f'results_exp2_pop{population_sizes[0]}_gen{gen}_mut{mutation_rates[0]}.csv'
        output_pdf = f'report_exp2_pop{population_sizes[0]}_gen{gen}_mut{mutation_rates[0]}.pdf'
        output_html = f'scatter_exp2_pop{population_sizes[0]}_gen{gen}_mut{mutation_rates[0]}.html'
        df_results.to_csv(output_csv, index=False)
        print(f"Hasil disimpan ke '{output_csv}'")
        visualize_anomalies(df_results, selected_features, output_pdf, output_html)
        elapsed_time = time.time() - start_time
        print(f"Uji coba selesai dalam {elapsed_time:.2f} detik.")
    
    # Eksperimen 3: Variasi Tingkat Mutasi
    print("\n=== Eksperimen 3: Variasi Tingkat Mutasi ===")
    population_sizes = [10]
    generations = [5]
    mutation_rates = [0.01, 0.05, 0.10]
    for mut_rate in mutation_rates:
        start_time = time.time()
        print(f"\nUji Coba - Populasi: {population_sizes[0]}, Generasi: {generations[0]}, Mutasi: {mut_rate}")
        ga = GeneticAlgorithm(features=features, population_size=population_sizes[0], generations=generations[0], mutation_rate=mut_rate)
        selected_features = ga.run(df_processed[features], df_processed.get('Label', pd.Series(['Normal'] * len(df_processed))))
        print(f"Fitur Terpilih: {selected_features}")
        df_results = detect_anomalies(df_processed, selected_features)
        output_csv = f'results_exp3_pop{population_sizes[0]}_gen{generations[0]}_mut{mut_rate}.csv'
        output_pdf = f'report_exp3_pop{population_sizes[0]}_gen{generations[0]}_mut{mut_rate}.pdf'
        output_html = f'scatter_exp3_pop{population_sizes[0]}_gen{generations[0]}_mut{mut_rate}.html'
        df_results.to_csv(output_csv, index=False)
        print(f"Hasil disimpan ke '{output_csv}'")
        visualize_anomalies(df_results, selected_features, output_pdf, output_html)
        elapsed_time = time.time() - start_time
        print(f"Uji coba selesai dalam {elapsed_time:.2f} detik.")
    
    elapsed_time_total = time.time() - start_time_total
    print(f"Proses total selesai dalam {elapsed_time_total:.2f} detik.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Proses dihentikan oleh pengguna.")
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")