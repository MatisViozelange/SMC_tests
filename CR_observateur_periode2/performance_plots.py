import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Lire les données depuis un fichier CSV
system = "basic_system"  # Remplacez par le nom du système souhaité
file_name = f'results_{system}.csv'
df = pd.read_csv(file_name)

# Créer la fenêtre principale
root = tk.Tk()
root.title("Visualisation des résultats")
root.geometry("900x600")

# Fonction pour tracer les erreurs en fonction du nombre de neurones pour un gamma fixe
def plot_error_vs_neurons(gamma_values, error_type):
    fig, ax = plt.subplots()
    for gamma_value in gamma_values:
        filtered_df = df[df["gamma"] == gamma_value]
        ax.plot(filtered_df["n_neurons"], filtered_df[error_type], marker='o', label=f"Gamma={gamma_value}")
    ax.set_xlabel("Nombre de neurones")
    ax.set_ylabel(error_type)
    ax.set_title(f"{error_type} en fonction du nombre de neurones")
    ax.legend()
    show_plot(fig)

# Fonction pour tracer les erreurs en fonction de gamma pour un nombre de neurones fixe
def plot_error_vs_gamma(neurons_values, error_type):
    fig, ax = plt.subplots()
    for neurons_value in neurons_values:
        filtered_df = df[df["n_neurons"] == neurons_value]
        ax.plot(filtered_df["gamma"], filtered_df[error_type], marker='o', label=f"Neurones={neurons_value}")
    ax.set_xlabel("Gamma")
    ax.set_ylabel(error_type)
    ax.set_title(f"{error_type} en fonction de Gamma")
    ax.legend()
    show_plot(fig)

# Fonction pour afficher le graphique dans la GUI
def show_plot(fig):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Interface utilisateur pour la sélection des paramètres
control_frame = ttk.LabelFrame(root, text="Contrôles")
control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)

plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Widgets pour choisir Gamma, Nombre de Neurones et type d'erreur
ttk.Label(control_frame, text="Gamma:").pack(pady=5)
gamma_combobox = ttk.Combobox(control_frame, values=sorted(df["gamma"].unique().tolist()))
gamma_combobox.pack(pady=5)

ttk.Label(control_frame, text="Nombre de neurones:").pack(pady=5)
neurons_combobox = ttk.Combobox(control_frame, values=sorted(df["n_neurons"].unique().tolist()))
neurons_combobox.pack(pady=5)

ttk.Label(control_frame, text="Type d'erreur:").pack(pady=5)
error_type_combobox = ttk.Combobox(control_frame, values=[
    "mean_quad_perturbation_approximation_error",
    "perturbation_approximation_std_deviation",
    "perturbation_approximation_correlation_coefficient"
])
error_type_combobox.pack(pady=5)
error_type_combobox.current(0)

# Boutons pour tracer les graphiques
plot_gamma_button = ttk.Button(
    control_frame,
    text="Tracer Erreur vs Neurones",
    command=lambda: plot_error_vs_neurons([float(val) for val in gamma_combobox.get().split(',')], error_type_combobox.get()),
)
plot_gamma_button.pack(pady=5)

plot_neurons_button = ttk.Button(
    control_frame,
    text="Tracer Erreur vs Gamma",
    command=lambda: plot_error_vs_gamma([int(val) for val in neurons_combobox.get().split(',')], error_type_combobox.get()),
)
plot_neurons_button.pack(pady=5)

# Lancer la boucle principale de la fenêtre
root.mainloop()
