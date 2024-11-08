import tkinter as tk
from tkinter import ttk
import pandas as pd

class DataGUI(tk.Tk):
    def __init__(self, df):
        super().__init__()

        # Set window title
        self.title("Data GUI")

        # Set window dimensions
        self.geometry("900x600")

        # Create a DataFrame
        self.df = df
        
        # Curves to plot
        self.curves = []
        
        # Zone de selection des données à visualser 
        self.control_frame = ttk.LabelFrame(self, text="Paramètres de visualisation")
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Zone de visualisation des données
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Widgets pour choisir le type de system
        self.system_combobox = ttk.Combobox(self.control_frame, values=['basic_system', 'pendule'])
        self.system_combobox.pack(pady=5)

        # Widgets pour choisir de ploter par rapport à Gamma ou Nombre de Neurones 
        self.X_combobox = ttk.Combobox(self.control_frame, values=['gamma', 'n_neurons'])
        self.X_combobox.pack(pady=5)
        
        # Widgets pour choisir le type d'erreur
        self.error_type_combobox = ttk.Combobox(
            self.control_frame, 
            values=[
                "mean_quad_perturbation_approximation_error",
                "perturbation_approximation_std_deviation",
                "perturbation_approximation_correlation_coefficient"
            ]
        )
        
        # Widgets pour choisir gamma et nombre de neurones
        self.gamma_combobox = ttk.Combobox(
            self.control_frame, 
            values=sorted(self.df["gamma"].unique().tolist()), 
            state='disabled'
        )
        self.gamma_combobox.pack(pady=5)
        
        self.neurons_combobox = ttk.Combobox(
            self.control_frame, 
            values=sorted(self.df["n_neurons"].unique().tolist()), 
            state='disabled'
        )
        self.neurons_combobox.pack(pady=5)
        
        # Boutons pour tracer les graphiques
        self.plot_curve_button = ttk.Button(
            self.control_frame,
            text="PLot Curve",
            command=self.plot_curve,
        )
        self.plot_curve_button.pack(pady=5)
        
        # Bouton pour afficher le DataFrame
        self.show_dataframe_button = ttk.Button(
            self.control_frame,
            text="Show DataFrame",
            command=self.show_dataframe,
        )
        self.show_dataframe_button.pack(pady=5)
        
    def plot_curve(self):
        # Get the data to plot
        
        
        # if the data is the same, add the new curve to the curve to plot collection
        
        
        # else, clear the plot and plot the new curve
        
        
        # plot the curve(s)
        
        pass
    
    def show_dataframe(self):
        # Tkinter Window for displaying the DataFrame
        
        pass
        
        

# Initialize and run the GUI
if __name__ == "__main__":
    system = 'basic_system'
    df = pd.read_csv(f'results_{system}.csv')
    app = DataGUI()
    app.mainloop()
