import tkinter as tk
from tkinter import messagebox, Label, Entry, Button, Checkbutton, IntVar, StringVar, OptionMenu
from simulation import Simulation
import plotting

def select_trajectory(*args):
    selected_trajectory = trajectory_var.get()
    simulation.set_trajectory(selected_trajectory)
    messagebox.showinfo("Траектория", f"Выбрана траектория: {selected_trajectory}")


def run_program():
    try:
        use_coords = coord_var.get()
        if use_coords:
            num_points = int(num_points_entry.get())
            gps_noise = float(gps_noise_entry.get())
            vo_noise = float(vo_noise_entry.get())

            use_kf = kf_var.get()
            use_pf = pf_var.get()
            use_ukf = ukf_var.get()

            rmse = simulation.run(num_points, gps_noise, vo_noise, use_kf, use_pf, use_ukf, mode='file')
            print("RMSE Values", f"KF RMSE: {rmse['kf']}\n"
                                               f"PF RMSE: {rmse['pf']}\n"
                                               f"UKF RMSE: {rmse['ukf']}")
            messagebox.showinfo("RMSE Values", f"KF RMSE: {rmse['kf']}\n"
                                               f"PF RMSE: {rmse['pf']}\n"
                                               f"UKF RMSE: {rmse['ukf']}")

        else:
            num_points = int(num_points_entry.get())
            gps_noise = float(gps_noise_entry.get())
            vo_noise = float(vo_noise_entry.get())

            use_kf = kf_var.get()
            use_pf = pf_var.get()
            use_ukf = ukf_var.get()
            rmse = simulation.run(num_points, gps_noise, vo_noise, use_kf, use_pf, use_ukf)
            print("RMSE Values", f"EKF RMSE: {rmse['kf']}\n"
                                               f"PF RMSE: {rmse['pf']}\n"
                                               f"UKF RMSE: {rmse['ukf']}\n"
                                               f"GPS RMSE: {rmse['gps']}\n"
                                               f"VO RMSE: {rmse['vo']}")
            messagebox.showinfo("RMSE Values", f"EKF RMSE: {rmse['kf']}\n"
                                               f"PF RMSE: {rmse['pf']}\n"
                                               f"UKF RMSE: {rmse['ukf']}\n"
                                               f"GPS RMSE: {rmse['gps']}\n"
                                               f"VO RMSE: {rmse['vo']}")
    except ValueError as e:
        messagebox.showerror("Error", str(e))



def show_plots():
    if simulation.gps_xyz is None or simulation.vo_xyz is None:
        messagebox.showerror("Error", "Выполните симуляцию перед показом графиков.")
        return
    plotting.plot_trajectory(
                    use_kf=simulation.use_kf,
                    use_pf=simulation.use_pf,
                    use_ukf=simulation.use_ukf,
                    true_xyz=simulation.true_xyz,
                    vo_xyz=simulation.vo_xyz,
                    gps_xyz=simulation.gps_xyz,
                    kf_results=simulation.kf_results,
                    pf_results=simulation.pf_results,
                    ukf_results=simulation.ukf_results,
                    mode=simulation.mode
    )


def show_help():
    help_text = (
        "Инструкция по использованию симулятора:\n\n"
        "Use Kalman Filter: Использование Калмановского фильтра в симуляции.\n"
        "Use Particle Filter: Использование фильтра частиц.\n"
        "Use UKF: Использование фильтра Калмана без запаха (Unscented Kalman Filter).\n"
        "Run Simulation: Запуск симуляции с заданными параметрами.\n"
        "Show Plots: Отображение результатов симуляции.\n"
        "Industrial building – Parrot Bebop 2:\n"
        "\t Данные VO - получены при помощи mono vision odometry\n"
        "\t Данные GPS - получены из метаданных фотографии\n"
    )
    messagebox.showinfo("Справка", help_text)


def enable_entries():
    state = tk.NORMAL if coord_var.get() == 0 else tk.DISABLED
    num_points_entry.config(state=state)
    gps_noise_entry.config(state=state)
    vo_noise_entry.config(state=state)
    trajectory_menu.config(state=state)



simulation = Simulation()

window = tk.Tk()
window.title("Simulation")

trajectory_var = StringVar(window)
trajectory_var.set("Random Walk")
trajectory_var.trace("w", select_trajectory)



coord_var = IntVar(value=0)

coord_checkbox = Checkbutton(window, text="Использовать данные Industrial building – Parrot Bebop 2 (см. Справка)",
                             variable=coord_var, command=enable_entries)
coord_checkbox.grid(row=0, column=0, sticky="W")

kf_var = IntVar(value=1)
pf_var = IntVar()
ukf_var = IntVar()

kf_checkbox = Checkbutton(window, text="Use Kalman Filter", variable=kf_var)
kf_checkbox.grid(row=6, column=0, sticky="W")

pf_checkbox = Checkbutton(window, text="Use Particle Filter", variable=pf_var)
pf_checkbox.grid(row=7, column=0, sticky="W")

ukf_checkbox = Checkbutton(window, text="Use UKF", variable=ukf_var)
ukf_checkbox.grid(row=8, column=0, sticky="W")

Label(window, text="Выбрать траекторию:").grid(row=1, column=0, sticky='w')
# OptionMenu(window, trajectory_var, "Random Walk", "Sinusoidal", "Circular", "Figure-Eight").grid(row=1, column=0)
trajectory_menu = OptionMenu(window, trajectory_var, "Random Walk", "Sinusoidal", "Circular", "Figure-Eight")
trajectory_menu.grid(row=1, column=0)



tk.Label(window, text="Number of Points:").grid(row=3, column=0)
num_points_entry = tk.Entry(window)
num_points_entry.grid(row=3, column=1)
num_points_entry.insert(0, "50")

tk.Label(window, text="GPS Noise:").grid(row=4, column=0)
gps_noise_entry = tk.Entry(window)
gps_noise_entry.grid(row=4, column=1)
gps_noise_entry.insert(0, "0.3")

tk.Label(window, text="VO Noise:").grid(row=5, column=0)
vo_noise_entry = tk.Entry(window)
vo_noise_entry.grid(row=5, column=1)
vo_noise_entry.insert(0, "0.15")

run_button = tk.Button(window, text="Run Simulation", command=run_program)
run_button.grid(row=6, column=0, columnspan=2)

plot_button = tk.Button(window, text="Show Plots", command=show_plots)
plot_button.grid(row=7, column=0, columnspan=2)


help_button = tk.Button(window, text="Справка", command=show_help)
help_button.grid(row=8, column=0, columnspan=2)

window.mainloop()