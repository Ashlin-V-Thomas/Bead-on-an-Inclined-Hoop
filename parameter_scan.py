#!/usr/bin/env python
from multiprocessing import Pool
import os, signal
import numpy as np
import h5py
from tqdm import tqdm
from time_evolution import velocity_verlet
import storage_setup


def compute_poincare(params):
    alpha, omega, theta0, p0 = params

    T = 2*np.pi/omega
    dt = T/100  # To get a good resolution of the dynamics
    discard = 100*T  # Discard initial transient behaviour
    t_fin = discard + 300*T

    t, theta, p = velocity_verlet(
            theta0, p0, t_fin,
            omega, alpha,
            discard_initial_time = discard,
            strob=True,
            strob_time=T,
            dt=dt
            )

    return alpha, omega, theta, p, dt

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    theta0 = np.deg2rad(0.1)
    p0 = 0

    #alphas_deg = np.arange(0,15,0.01)
    alphas_deg = np.array([0])
    alphas_rad = np.deg2rad(alphas_deg)
    #omegas = [i for i in range(1,11)]
    omegas = np.arange(0.01,10, 0.01)

    param_list = [(a,w, theta0, p0) for a in alphas_rad for w in omegas]

    pool = Pool(os.cpu_count()-1, initializer=init_worker)

    try:
        with h5py.File("Data/poincare_trajectories.h5", "a") as file:

            for alpha, omega, theta, p, dt in tqdm(
                    pool.imap_unordered(compute_poincare, param_list),
                    total=len(param_list),
                    desc="Computing Stroboscopic trajectories"
                    ):
                alpha_grp = storage_setup.get_or_create_group(file, f"alpha{np.rad2deg(alpha):05.2f}", attrs={"alpha":alpha})
                omega_grp = storage_setup.get_or_create_group(alpha_grp, f"omega{omega:06.3f}", attrs={"omega":omega})
                init_grp = storage_setup.get_or_create_group(omega_grp, f"init0_short_time", attrs={
                    "theta0": theta0,
                    "p0": p0,
                    "dt": dt
                    })

                storage_setup.create_or_overwrite_dataset(
                        init_grp, "theta", theta
                        )
                storage_setup.create_or_overwrite_dataset(
                        init_grp, "p", p
                        )

            pool.close()
            pool.join()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating workers...")
        pool.terminate()
        pool.join()
        print("Workers terminated cleanly.")
        exit()

if __name__ == "__main__":
    main()
