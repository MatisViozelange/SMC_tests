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
        self.system = None
        
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
        
        # Frame for values selection 
        self.values_frame = ttk.LabelFrame(self, text="Sélection des valeurs", padding=(20, 10))
        self.values_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Frame for buttons
        self.buttons_frame = ttk.LabelFrame(self, text="Actions", padding=(20, 10))
        self.buttons_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=20, pady=20)
        
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
            self.buttons_frame,
            text="Tracer la Courbe",
            command=self.plot_curve,
            style='TButton'
        )
        self.plot_curve_button.pack(pady=20, ipady=10, ipadx=10)
        
        self.plot_curve_all_systems_button = ttk.Button(
            self.buttons_frame,
            text='Tracer Pour Tous Les Systèmes',
            command=self.plot_curve_all_systems,
            style='TButton'
        )
        self.plot_curve_all_systems_button.pack(pady=20, ipady=10, ipadx=10)
        
        # Bouton pour afficher le DataFrame
        self.show_dataframe_button = ttk.Button(
            self.buttons_frame,
            text="Afficher le DataFrame",
            command=self.show_dataframe,
            style='TButton'
        )
        self.show_dataframe_button.pack(pady=10, ipady=10, ipadx=10)
        
        # CLear curves
        self.clear_curves_button = ttk.Button(
            self.buttons_frame,
            text="Clear",
            command=self.clear,
            style='TButton'
        )
        self.clear_curves_button.pack(pady=10, ipady=10, ipadx=10)
        
        # Button to compute the mean error for the given parameters
        self.plot_error_data_button = ttk.Button(
            self.buttons_frame,
            text="Plot Error Data",
            command=self.plot_error_data,
            style='TButton'
        )
        self.plot_error_data_button.pack(pady=10, ipady=10, ipadx=10)
        
        #Button to plot the error mean, median and std deviation for each const value
        self.plot_all_error_data_button = ttk.Button(
            self.buttons_frame,
            text="Show error data for all const values",
            command=self.plot_all_error_data,
            style='TButton'
        )
        self.plot_all_error_data_button.pack(pady=10, ipady=10, ipadx=10)

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
        self.system = self.system_combobox.get()
        # self.clear()
        try:
            self.df = pd.read_csv(f'results_{self.system}.csv')
            # Update gamma and neurons combobox values
            self.gamma_combobox.config(values=sorted(self.df["gamma"].unique().tolist()))
            self.neurons_combobox.config(values=sorted(self.df["n_neurons"].unique().tolist()))
        except FileNotFoundError:
            print(f"Error: Data file for system '{self.system}' not found.")
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
        self.curves.append((
            x_data, 
            y_data, 
            f"{constant_value} = {gamma_value if x_axis == 'n_neurons' else neurons_value}, {self.system}"
        ))

        # Plot curves with actual points of values
        plt.figure(figsize=(10, 6))
        for curve in self.curves:
            plt.plot(curve[0], curve[1], label=curve[2], marker='o')
            
        plt.xlabel(x_axis)
        plt.ylabel(error_type)
        plt.legend()
        plt.title(f"{error_type} vs {x_axis} fot {self.system}")
        plt.grid(True)
        plt.show()

    def plot_curve_all_systems(self):
        plt.close('all')
        
        systems = ['basic_system', 'pendule']
        
        for system in systems:
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
            self.curves.append((
                x_data, 
                y_data, 
                f"{constant_value} = {gamma_value if x_axis == 'n_neurons' else neurons_value}, {system}"
            ))

        # Plot curves with actual points of values
        plt.figure(figsize=(10, 6))
        for curve in self.curves:
            plt.plot(curve[0], curve[1], label=curve[2], marker='o')
            
        plt.xlabel(x_axis)
        plt.ylabel(error_type)
        plt.legend()
        plt.title(f"{error_type} vs {x_axis} for all systems")
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

    def plot_error_data(self):
        self.clear()
        # Get current selections from comboboxes
        x_axis = self.X_combobox.get()
        error_type = self.error_type_combobox.get()
        gamma_value = self.gamma_combobox.get()
        neurons_value = self.neurons_combobox.get()

        if x_axis == 'gamma':
            if not neurons_value:
                print("Error: Please select a number of neurons value for gamma analysis.")
                return

            # Select rows where the neurons value matches the selected one
            df_filtered = self.df[self.df['n_neurons'] == int(neurons_value)]
            if df_filtered.empty:
                print("Error: No data available for the selected neurons value.")
                return

            # Calculate statistics
            mean_error = df_filtered[error_type].mean()
            median_error = df_filtered[error_type].median()
            std_error = df_filtered[error_type].std()

            # Plot gamma on the x-axis and error type on the y-axis
            plt.plot(df_filtered['gamma'], df_filtered[error_type], marker='o', linestyle='-', color='b', label=f'{error_type}')
            plt.axhline(mean_error, color='g', linestyle='--', label=f'Mean: {mean_error:.2f}')
            plt.axhline(median_error, color='r', linestyle='-.', label=f'Median: {median_error:.2f}')
            plt.axhline(mean_error + std_error, color='y', linestyle=':', label=f'Mean ± Std Dev\nStd Dev: {std_error:.2f}')
            plt.xlabel('Gamma')
            plt.ylabel(f'{error_type}')
            plt.title(f'{error_type} vs Gamma (Neurons = {neurons_value}) for {self.system}')
            plt.legend()
            

        else:  # x_axis == 'n_neurons'
            if not gamma_value:
                print("Error: Please select a gamma value for neuron analysis.")
                return

            # Select rows where the gamma value matches the selected one
            df_filtered = self.df[self.df['gamma'] == float(gamma_value)]
            if df_filtered.empty:
                print("Error: No data available for the selected gamma value.")
                return

            # Calculate statistics
            mean_error = df_filtered[error_type].mean()
            median_error = df_filtered[error_type].median()
            std_error = df_filtered[error_type].std()

            # Plot neurons on the x-axis and error type on the y-axis
            plt.plot(df_filtered['n_neurons'], df_filtered[error_type], marker='o', linestyle='-', color='r', label='Error')
            plt.axhline(mean_error, color='g', linestyle='--', label=f'Mean: {mean_error:.2f}')
            plt.axhline(median_error, color='r', linestyle='-.', label=f'Median: {median_error:.2f}')
            plt.axhline(mean_error + std_error, color='y', linestyle=':', label=f'Mean ± Std Dev\nStd Dev: {std_error:.2f}')
            plt.xlabel('Neurons Value')
            plt.ylabel(f'{error_type}')
            plt.title(f'{error_type} vs Neurons (Gamma = {gamma_value}) for {self.system}')
            plt.legend()

        plt.grid(True)
        plt.show()
    
    def plot_all_error_data(self):
        self.clear()

        # Get current selections from comboboxes
        x_axis = self.X_combobox.get()
        error_type = self.error_type_combobox.get()
        
        if not x_axis or not error_type:
            print("Error: Please select X-axis and error type before plotting.")
            return
        
        # Group the dataframe by n_neurons
        grouped_values = self.df.groupby(f'{x_axis}')

        # Initialize lists to store mean, median, and standard deviation for each value of x_axis
        const_values = []
        mean_values = []
        median_values = []
        std_values = []

        # Iterate over each group of n_neurons
        for value, group in grouped_values:
            error_values = group[error_type]

            # Compute mean, median, and standard deviation for the curve
            mean_value = error_values.mean()
            median_value = error_values.median()
            std_value = error_values.std()

            # Store the computed values
            const_values.append(value)
            mean_values.append(mean_value)
            median_values.append(median_value)
            std_values.append(std_value)
            
        # Plot the data
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        axes[0].plot(const_values, mean_values, marker='o', linestyle='-', label='Mean')
        axes[0].set_title(f'{error_type} Mean vs {x_axis} for {self.system}')
        axes[0].set_ylabel('Mean')
        axes[0].legend()

        axes[1].plot(const_values, median_values, marker='o', linestyle='-', label='Median')
        axes[1].set_title(f'{error_type} Median vs {x_axis}')
        axes[1].set_ylabel('Median')
        axes[1].legend()

        axes[2].plot(const_values, std_values, marker='o', linestyle='-', label='Standard Deviation')
        axes[2].set_title(f'{error_type} Standard Deviation vs {x_axis}')
        axes[2].set_xlabel(f'{x_axis}')
        axes[2].set_ylabel('Standard Deviation')
        axes[2].legend()

        plt.tight_layout()
        plt.show()
    
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