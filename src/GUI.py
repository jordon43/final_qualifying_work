import tkinter as tk
from tkinter import messagebox, Checkbutton, IntVar
from simulation import Simulation
import plotting

simulation = Simulation()

def run_program():
    try:
        num_points = int(num_points_entry.get())
        gps_noise = float(gps_noise_entry.get())
        vo_noise = float(vo_noise_entry.get())

        use_kf = kf_var.get()
        use_pf = pf_var.get()
        use_ukf = ukf_var.get()

        rmse = simulation.run(num_points, gps_noise, vo_noise, use_kf, use_pf, use_ukf)
        messagebox.showinfo("RMSE Values", f"KF RMSE: {rmse['kf']}\n"
                                           f"PF RMSE: {rmse['pf']}\n"
                                           f"UKF RMSE: {rmse['ukf']}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def show_plots():
    if simulation.true_xyz is None or simulation.vo_xyz is None:
        messagebox.showerror("Error", "Выполните симуляцию перед показом графиков.")
        return
    plotting.plot_trajectory(
                        simulation.true_xyz,
                        simulation.vo_xyz,
                    gps_xyz=simulation.gps_xyz,
                    kf_results=simulation.kf_results,
                    pf_results=simulation.pf_results,
                    ukf_results=simulation.ukf_results)


window = tk.Tk()
window.title("Simulation")


kf_var = IntVar()
pf_var = IntVar()
ukf_var = IntVar()


kf_checkbox = Checkbutton(window, text="Use Kalman Filter", variable=kf_var)
kf_checkbox.grid(row=5, column=0, sticky="W")
kf_checkbox.select()

pf_checkbox = Checkbutton(window, text="Use Particle Filter", variable=pf_var)
pf_checkbox.grid(row=6, column=0, sticky="W")

ukf_checkbox = Checkbutton(window, text="Use UKF", variable=ukf_var)
ukf_checkbox.grid(row=7, column=0, sticky="W")


tk.Label(window, text="Number of Points:").grid(row=0, column=0)
num_points_entry = tk.Entry(window)
num_points_entry.grid(row=0, column=1)
num_points_entry.insert(0, "50")

tk.Label(window, text="GPS Noise:").grid(row=1, column=0)
gps_noise_entry = tk.Entry(window)
gps_noise_entry.grid(row=1, column=1)
gps_noise_entry.insert(0, "0.3")

tk.Label(window, text="VO Noise:").grid(row=2, column=0)
vo_noise_entry = tk.Entry(window)
vo_noise_entry.grid(row=2, column=1)
vo_noise_entry.insert(0, "0.15")

run_button = tk.Button(window, text="Run Simulation", command=run_program)
run_button.grid(row=3, column=0, columnspan=2)

plot_button = tk.Button(window, text="Show Plots", command=show_plots)
plot_button.grid(row=4, column=0, columnspan=2)

window.mainloop()