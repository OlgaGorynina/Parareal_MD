Description:
============

This page contains the code from the paper 

"Combining machine-learned and classical force fields with the
parareal algorithm: application to the diffusion of atomistic
defects"

by Olga Gorynina, Frédéric Legoll, Tony Lelièvre and Danny Perez

Dependencies:
=============

-Python 3.8+
-Compile LAMMPS as a shared library with python support:
https://docs.lammps.org/Python_install.html

Running: 
========

```
python run.py [options]
```

--N: Number of time steps in parareal algorithm (default value 2000)
--fine: choice of potential for fine propagator (default value 14, corresponds to SNAP-205 potential)
	availible potentials: 
	14 - SNAP-205
	12 - SNAP-141
	10 - SNAP-92
	8 -  SNAP-56
	6 -  SNAP-31
	4 -  SNAP-15
	2 -  SNAP-6
	0 -  EAM potential
--coarse: choice of potential for coarse propagator (default value 0, corresponds to EAM potential)
--dc: convergence parameter (default value 10^(-5))
--de: explosion threshold parameter, de > dc (default value 0.35)

