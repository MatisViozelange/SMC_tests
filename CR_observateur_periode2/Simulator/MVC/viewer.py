import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

class MVCView:
    def __init__(self, model):
        self.model = model
        self.model.view = self
        self.root = tk.Tk()
        self.root.title("Control System Simulator")

        self.system_label = ttk.Label(self.root, text="Select System:")
        self.system_label.pack()

        self.system_combobox = ttk.Combobox(self.root, values=list(model.systems.keys()))
        self.system_combobox.pack()
        self.system_combobox.bind("<<ComboboxSelected>>", self.on_system_selected)
        
        self.run_button = ttk.Button(self.root, text="Run Simulation", command=self.model.run_simulation)
        self.run_button.pack()
        
        self.performance_button = ttk.Button(self.root, text="Compute Performance", command=self.model.compute_performance)
        self.performance_button.pack()

        self.plot_button = ttk.Button(self.root, text="Plot Results", command=self.plot_results)
        self.plot_button.pack()

        self.show_table_button = ttk.Button(self.root, text="Show Performance Table", command=self.show_results_table)
        self.show_table_button.pack()

    def on_system_selected(self, event):
        system_name = self.system_combobox.get()
        self.model.set_system(system_name)
        print(f"Selected System: {system_name}")

    def plot_results(self):
        real_perturbation = self.model.current_system.get_perturbation(self.model.times, self.model.controlers['ASTWC'].x1, self.model.controlers['ASTWC'].x2)
        real_perturbation_nn = self.model.current_system.get_perturbation(self.model.times, self.model.controlers['NN_based_ASTWC'].controler.x1, self.model.controlers['NN_based_ASTWC'].controler.x2)

        # Now compute the derivatives after smoothing
        dot_perturbation = np.gradient(real_perturbation, self.model.Te)
        dot_perturbation_nn = np.gradient(real_perturbation_nn, self.model.Te)

        d_max = np.max(np.abs(dot_perturbation))
        d_max_nn = np.max(np.abs(dot_perturbation_nn))

        fig, axs = plt.subplots(3, 3, num=1, figsize=(12, 12), sharex=True, sharey=False)
        (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = axs.flat

        # ASTWC Plots
        ax1.set_title('ASTWC - States')
        ax1.plot(self.model.controlers['ASTWC'].times, self.model.controlers['ASTWC'].x1[:-1], label='x1')
        ax1.plot(self.model.controlers['ASTWC'].times, self.model.controlers['ASTWC'].x2[:-1], label='x2')
        ax1.plot(self.model.controlers['ASTWC'].times, self.model.controlers['ASTWC'].y_ref, label='ref')
        ax1.set_ylabel('x1, x2')
        ax1.legend()

        ax2.set_title('ASTWC - Perturbation Derivatives')
        ax2.plot(self.model.times, np.abs(dot_perturbation), label='|d_dot|')
        ax2.plot(self.model.times, self.model.controlers['ASTWC'].k[:-1], label='k')
        ax2.axhline(y=d_max, color='r', linestyle='--', label=f'Delta_d = {d_max:.2f}')
        ax2.set_ylabel('Perturbation Derivative')
        ax2.legend()

        ax3.set_title('ASTWC - Sliding Variable')
        ax3.plot(self.model.controlers['ASTWC'].times, self.model.controlers['ASTWC'].s, label='s')
        ax3.axhline(y=self.model.controlers['ASTWC'].epsilon, color='r', linestyle='--', label=f'epsilon = {self.model.controlers['ASTWC'].epsilon:.2f}')
        ax3.axhline(y=-self.model.controlers['ASTWC'].epsilon, color='r', linestyle='--')
        ax3.set_ylabel('Sliding Variable')
        ax3.legend()

        ax8.set_title('ASTWC - Inputs')
        ax8.plot(self.model.times, self.model.controlers['ASTWC'].u, label='u')
        ax8.plot(self.model.times, self.model.controlers['ASTWC'].v_dot, label='v_dot')
        ax8.set_xlabel('Time')
        ax8.set_ylabel('Inputs')
        ax8.legend()

        # NN-based ASTWC Plots
        ax4.set_title('NN-based ASTWC - States')
        ax4.plot(self.model.controlers['NN_based_ASTWC'].controler.times, self.model.controlers['NN_based_ASTWC'].controler.x1[:-1], label='x1')
        ax4.plot(self.model.controlers['NN_based_ASTWC'].controler.times, self.model.controlers['NN_based_ASTWC'].controler.x2[:-1], label='x2')
        ax4.plot(self.model.controlers['NN_based_ASTWC'].controler.times, self.model.controlers['NN_based_ASTWC'].controler.y_ref, label='ref')
        ax4.set_ylabel('x1, x2')
        ax4.legend()

        ax5.set_title('NN-based ASTWC - Perturbation Derivatives')
        ax5.plot(self.model.times, np.abs(dot_perturbation_nn), label='|d_dot|')
        ax5.plot(self.model.times, self.model.controlers['NN_based_ASTWC'].controler.k[:-1], label='k')
        ax5.axhline(y=d_max_nn, color='r', linestyle='--', label=f'Delta_d = {d_max_nn:.2f}')
        ax5.set_ylabel('Perturbation Derivative')
        ax5.legend()

        ax6.set_title('NN-based ASTWC - Sliding Variable')
        ax6.plot(self.model.controlers['NN_based_ASTWC'].controler.times, self.model.controlers['NN_based_ASTWC'].controler.s, label='s')
        ax6.axhline(y=self.model.controlers['NN_based_ASTWC'].controler.epsilon, color='r', linestyle='--', label=f'epsilon = {self.model.controlers['NN_based_ASTWC'].controler.epsilon:.2f}')
        ax6.axhline(y=-self.model.controlers['NN_based_ASTWC'].controler.epsilon, color='r', linestyle='--')
        ax6.set_ylabel('Sliding Variable')
        ax6.legend()

        ax7.set_title('NN-based ASTWC - Perturbation Approximation')
        ax7.plot(self.model.times, real_perturbation_nn, label='d')
        ax7.plot(self.model.times, self.model.controlers['NN_based_ASTWC'].perturbation[:-1], label='NN d approx')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Perturbation')
        ax7.legend()

        ax9.set_title('NN-based ASTWC - Inputs')
        ax9.plot(self.model.times, self.model.controlers['NN_based_ASTWC'].controler.u, label='u')
        ax9.plot(self.model.times, self.model.controlers['NN_based_ASTWC'].controler.v_dot, label='v_dot')
        ax9.set_xlabel('Time')
        ax9.set_ylabel('Inputs')
        ax9.legend()

        plt.tight_layout()
        plt.show()

    def show_results_table(self):
        if self.model.results_df is None:
            print("Please run the simulation first.")
            return

        root = tk.Tk()
        root.title("Approximation Performance Metrics")

        frame = ttk.Frame(root)
        frame.pack(expand=True, fill='both')

        tree = ttk.Treeview(frame, columns=list(self.model.results_df.columns), show='headings')
        for col in self.model.results_df.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor='center', width=150)

        for index, row in self.model.results_df.iterrows():
            tree.insert("", "end", values=list(row))

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        tree.pack(expand=True, fill='both')

        style = ttk.Style()
        style.configure("Treeview", rowheight=30)

        root.mainloop()