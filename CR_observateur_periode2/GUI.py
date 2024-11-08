import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt

class DataGUI(tk.Tk):
    def __init__(self, df=None):
        super().__init__()

        # Set window title
        self.title("Data GUI")

        # Set window dimensions
        self.geometry("1200x800")
        self.configure(bg='#f0f0f0')

        # Create a DataFrame
        self.df = df if df is not None else pd.DataFrame({})
        
        # Curves to plot
        self.curves = []
        
        # Zone de selection des données à visualiser 
        self.control_frame = ttk.LabelFrame(self, text="Paramètres de visualisation", padding=(20, 10))
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=20, pady=20)
        
        # Widgets pour choisir le type de system
        ttk.Label(self.control_frame, text="Type de Système", font=("Helvetica", 12)).pack(pady=10)
        self.system_combobox = ttk.Combobox(self.control_frame, values=['basic_system', 'pendule'], font=("Helvetica", 12), state='readonly')
        self.system_combobox.pack(pady=10, ipady=5, ipadx=10)
        self.system_combobox.bind('<<ComboboxSelected>>', self.update_dataframe)

        # Widgets pour choisir de ploter par rapport à Gamma ou Nombre de Neurones 
        ttk.Label(self.control_frame, text="Axe X", font=("Helvetica", 12)).pack(pady=10)
        self.X_combobox = ttk.Combobox(self.control_frame, values=['gamma', 'n_neurons'], font=("Helvetica", 12), state='readonly')
        self.X_combobox.pack(pady=10, ipady=5, ipadx=10)
        self.X_combobox.bind('<<ComboboxSelected>>', self.update_combobox_states)
        
        # Widgets pour choisir le type d'erreur
        ttk.Label(self.control_frame, text="Type d'Erreur", font=("Helvetica", 12)).pack(pady=10)
        self.error_type_combobox = ttk.Combobox(
            self.control_frame, 
            values=[
                "mean_quad_perturbation_approximation_error",
                "perturbation_approximation_std_deviation",
                "perturbation_approximation_correlation_coefficient"
            ],
            font=("Helvetica", 12),
            state='readonly'
        )
        self.error_type_combobox.pack(pady=10, ipady=5, ipadx=10)
        self.error_type_combobox.bind('<<ComboboxSelected>>', self.update_error_combobox_states)
        
        # Frame for values selection and buttons
        self.values_frame = ttk.LabelFrame(self, text="Sélection des valeurs et Actions", padding=(20, 10))
        self.values_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=20, pady=20)
        
        # Widgets pour choisir gamma et nombre de neurones
        ttk.Label(self.values_frame, text="Valeur de Gamma", font=("Helvetica", 12)).pack(pady=10)
        self.gamma_combobox = ttk.Combobox(
            self.values_frame, 
            values=[], 
            state='disabled',
            font=("Helvetica", 12)
        )
        self.gamma_combobox.pack(pady=10, ipady=5, ipadx=10)
        
        ttk.Label(self.values_frame, text="Nombre de Neurones", font=("Helvetica", 12)).pack(pady=10)
        self.neurons_combobox = ttk.Combobox(
            self.values_frame, 
            values=[], 
            state='disabled',
            font=("Helvetica", 12)
        )
        self.neurons_combobox.pack(pady=10, ipady=5, ipadx=10)
        
        # Boutons pour tracer les graphiques
        self.plot_curve_button = ttk.Button(
            self.values_frame,
            text="Tracer la Courbe",
            command=self.plot_curve,
            style='TButton'
        )
        self.plot_curve_button.pack(pady=20, ipady=10, ipadx=10)
        
        # Bouton pour afficher le DataFrame
        self.show_dataframe_button = ttk.Button(
            self.values_frame,
            text="Afficher le DataFrame",
            command=self.show_dataframe,
            style='TButton'
        )
        self.show_dataframe_button.pack(pady=10, ipady=10, ipadx=10)
        
        # CLear curves
        self.clear_curves_button = ttk.Button(
            self.values_frame,
            text="Clear",
            command=self.clear,
            style='TButton'
        )
        self.clear_curves_button.pack(pady=10, ipady=10, ipadx=10)

        # key bindings
        self.bind("<Return>", lambda event: self.plot_curve())
        self.bind("<Escape>", lambda event: self.clear())

        # Style customization
        style = ttk.Style()
        style.configure('TButton', font=("Helvetica", 12), padding=10)
        style.configure('TCombobox', padding=10)
        style.configure('TLabel', padding=10)

    def update_dataframe(self, event):
        # Update DataFrame based on system selection
        system = self.system_combobox.get()
        self.clear()
        try:
            self.df = pd.read_csv(f'results_{system}.csv')
            # Update gamma and neurons combobox values
            self.gamma_combobox.config(values=sorted(self.df["gamma"].unique().tolist()))
            self.neurons_combobox.config(values=sorted(self.df["n_neurons"].unique().tolist()))
        except FileNotFoundError:
            print(f"Error: Data file for system '{system}' not found.")
            self.df = pd.DataFrame({})
            self.gamma_combobox.config(values=[])
            self.neurons_combobox.config(values=[])

    def update_combobox_states(self, event):
        x_axis = self.X_combobox.get()
        self.clear()
        if x_axis == 'gamma':
            self.gamma_combobox.set('')
            self.gamma_combobox.config(state='disabled')
            self.neurons_combobox.config(state='normal')
        elif x_axis == 'n_neurons':
            self.gamma_combobox.config(state='normal')
            self.neurons_combobox.set('')
            self.neurons_combobox.config(state='disabled')
        else:
            self.gamma_combobox.config(state='disabled')
            self.neurons_combobox.config(state='disabled')
            
    def update_error_combobox_states(self, event):
        self.clear()

    def plot_curve(self):
        plt.close('all')
        
        # Get current selections from comboboxes
        x_axis = self.X_combobox.get()
        error_type = self.error_type_combobox.get()
        gamma_value = self.gamma_combobox.get()
        neurons_value = self.neurons_combobox.get()

        # Validate the user selections
        if not x_axis or not error_type:
            print("Error: Please select X-axis and error type before plotting.")
            return

        # Enable appropriate combobox based on selected x_axis
        if x_axis == 'gamma':
            if not neurons_value:
                print("Error: Please select a number of neurons value for gamma analysis.")
                return
            filtered_df = self.df[self.df['n_neurons'] == int(neurons_value)]
            x_data = filtered_df['gamma']
            constant_value = 'n_neurons'
        elif x_axis == 'n_neurons':
            if not gamma_value:
                print("Error: Please select a gamma value for neuron analysis.")
                return
            filtered_df = self.df[self.df['gamma'] == float(gamma_value)]
            x_data = filtered_df['n_neurons']
            constant_value = 'gamma'
        else:
            print("Error: Invalid selection for X-axis.")
            return

        # Get y-axis data
        y_data = filtered_df[error_type]

        # Add to curves to plot
        self.curves.append((x_data, y_data, f"{constant_value} = {gamma_value if x_axis == 'n_neurons' else neurons_value}"))

        # Plot curves with actual points of values
        plt.figure(figsize=(10, 6))
        for curve in self.curves:
            plt.plot(curve[0], curve[1], label=curve[2], marker='o')
            
        plt.xlabel(x_axis)
        plt.ylabel(error_type)
        plt.legend()
        plt.title(f"{error_type} vs {x_axis}")
        plt.grid(True)
        plt.show()

    def show_dataframe(self):
        # Tkinter Window for displaying the DataFrame
        dataframe_window = tk.Toplevel(self)
        dataframe_window.title("DataFrame View")
        dataframe_window.geometry("800x600")

        # Convert DataFrame to text for display
        text_widget = tk.Text(dataframe_window, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, self.df.to_string())

    def clear(self):
        self.curves = []

        # Close all open plots
        plt.close('all')
        
        # Clear the comboboxes
        # self.system_combobox.set('')
        # self.X_combobox.set('')
        # self.error_type_combobox.set('')

    
# Initialize and run the GUI
if __name__ == "__main__":
    app = DataGUI()
    app.mainloop()