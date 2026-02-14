#!/usr/bin/env python
from time_evolution import velocity_verlet
import storage_setup
import numpy as np
import h5py

initial_conditions = [
        (np.deg2rad(0.1), 0),
        ]

t_fin = 10_000

with h5py.File("Data/poincare_trajectories.h5", "a") as file:
    for alpha_grp in file.values():
        for omega_grp in alpha_grp.values():
            for i, init in enumerate(initial_conditions):
                init_grp = storage_setup.get_or_create_group(omega_grp, f"init{i}", attrs={"theta0": init[0], "p0": init[1]})
                omega = init_grp.attrs["omega"]
                alpha = init_grp.attrs["alpha"]
                t, theta, p = velocity_verlet(
                        init[0], init[1], t_fin,
                        omega, alpha,
                        discard_initial_time = 1000,
                        strob=True,
                        strob_time=2*np.pi/omega
                        )

                storage_setup.create_or_overwrite_dataset(
                        init_grp, "theta", theta
                        )
                storage_setup.create_or_overwrite_dataset(
                        init_grp, "p", p
                        )
                
                print("-"*50)
                print(f"Finished alpha={np.rad2deg(alpha):.2f} deg, omega={omega:.2f} rad/s")
