""" 
2022 Olga Gorynina (Ecole Des Ponts, Paris)

Adaptive parareal for MD, fitsnap/lammps realisation
This example will use 2(coarse and fine) snap/eam potentials
"""

from __future__ import print_function
import sys
import numpy as np 
from random import seed, randint
import time
import os
import shutil
import glob
from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM
from propagation import Propagation
from adaptive_parareal import Parareal

# Set default initial data
N_steps = 2000 # Number of time steps (N)
N_points = 1 # Number of subintervals in every time-step (L)
N_particles = 129 # Number of particles with interstiail atom
fine_propagator = 4 # Choice of  potential for fine propagator
coarse_propagator = 0 # Choice of  potential for coarse propagator, value 0 corresponds to eam potential
tdamp = 1.0 # Damping parameter - 1/gamma
equilibration_flag = 1 # 0 if no need in equilibration
equilibration_steps = 10000 # Number of equilibration steps, runs sequantially
test_case = "in.interstitial." # Self interstitial atom simulation
flag_only_reference = 0 # 1 if only reference solution needed
delta_conv = 10**(-3) # Convergence parameter
delta_expl = 0.3 # Explosion treshold parameter

# Parse command line
if("--N" in  sys.argv):
    N_steps = int(sys.argv[sys.argv.index("--N") + 1]) 
if("--fine" in  sys.argv):
    fine_propagator = int(sys.argv[sys.argv.index("--fine") + 1]) 
if("--coarse" in  sys.argv):
    coarse_propagator = int(sys.argv[sys.argv.index("--coarse") + 1]) 
if("--dc" in  sys.argv):
    delta_conv = float(sys.argv[sys.argv.index("--dc") + 1]) 
if("--de" in  sys.argv):
    delta_expl = float(sys.argv[sys.argv.index("--de") + 1]) 

# Create output directories
folder_name = test_case + '-N_steps_'+ str(N_steps) + '-N_points_'+ str(N_points) + '-delta_expl_'+ str(delta_expl) + '-delta_conv_'+ str(delta_conv) + '-fine(coarse)_'+ str(fine_propagator) + '(' + str(coarse_propagator) + ')'
if os.path.exists(folder_name):
	shutil.rmtree(folder_name)
os.mkdir(folder_name)

path = os.path.dirname(os.path.abspath(__file__))
dst_folder_parareal = path + '/' +  folder_name +"/parareal_solution/"
os.mkdir(dst_folder_parareal)
dst_folder_reference = path + '/' +  folder_name +"/reference_solution/"
os.mkdir(dst_folder_reference)

# Random number generator
np.random.seed(1)
white_noise = np.random.randint(1, 900000000, size = N_steps+1)

####################################################################

# Set lammps parameters (initial conditions + equlibration run)
lmp = lammps()
lmp.command('log '+ folder_name +'/log.lammps.dat')
lmp.file(test_case + "simulation_box")
lmp.command('variable seed equal ' + str(white_noise[0]))
lmp.command('variable tdamp equal ' + str(tdamp))

# Set propagator for equlibration run
lmp.command('variable propagator equal ' + str(fine_propagator))
if fine_propagator == 0: 
	lmp.file(test_case +"potential_EAM")
else: 
	lmp.file(test_case +"potential")

# Intital configuration
lmp.command('variable Tcorrected equal ${temp}')
lmp.command('dump firstdump all custom 10 '+ folder_name +'/dump_output_0.dat id type xs ys zs ix iy iz') # to dump initial configuration
lmp.command('run 1')
lmp.command('undump firstdump') # to dump initial configuration

# Add interstitial atom
lmp.command('create_atoms 1 single 0 1.59015 0 # units box')
lmp.command('min_style cg')
lmp.command('minimize 1e-15 1e-15 5000 5000')

# Perform equilibration run
if equilibration_flag == 1:
	t0 = time.process_time() 
	lmp.command('run '+str(equilibration_steps))
	t0 = time.process_time() - t0
	print('time per equlibration steps : '+str(t0))	

# Store initial state for all parareal iterations
temp1 = lmp.numpy.extract_atom("x")
temp2 = lmp.numpy.extract_atom("v")
q_initial = np.array(temp1)
v_initial = np.array(temp2)

##################################################################################

# Generate temperature at the given step in order to correct lammps scheme
temperature_correction = np.zeros(N_points+1)
temperature_correction[0] = 2.0
for i in range(1, N_points+1):
	temperature_correction[i] = 0.25*(-np.sqrt(temperature_correction[i-1]) + np.sqrt(temperature_correction[i-1] + 8))**2
for i in range(0, N_points+1):
	lmp.command('variable Tcorrected_' + str(i) + ' equal ' + str(temperature_correction[i]))

lmp.command('variable N_points equal ' + str(N_points))       
lmp.command('variable time_step equal "step"')   
mixed_beta = 'variable Tcorrected equal v_Tcorrected_0*${temp}*(v_time_step%${N_points}==0)'
for i in range(1, N_points+1):
	mixed_beta += '+v_Tcorrected_' + str(i) + '*${temp}*(v_time_step%${N_points}==' + str(i) + ')'
lmp.command(mixed_beta)

###################################################################################

# Compute reference solution
start_time = time.time()
reference_solution = Propagation(N_steps, N_particles, N_points, "fine", fine_propagator, test_case, "reference", folder_name)
reference_solution.run(0, N_steps, q_initial, v_initial, white_noise, lmp)

# Write cpu time
with open(folder_name + '/cpu_for_reference.txt', 'w') as outfile:
	outfile.write('# cpu for reference solution_' + str(fine_propagator)+ '_' + str(N_steps*N_points) + ':' +str((time.time()-start_time)))

q_reference, v_reference = reference_solution.return_solution()
reference_solution.write_output(folder_name)
path = os.path.dirname(os.path.abspath(__file__))
src_folder = path

# Output for 0 iteration
files = glob.glob(src_folder +  '/' +  folder_name + "/dump_output_0.dat")
for file in files:
    file_name = os.path.basename(file)
    shutil.copy2(file, dst_folder_reference + "/dump_output_reference_0.dat")
    shutil.move(file, dst_folder_parareal + file_name)
    
if flag_only_reference == 1:
	exit()

##############################################################################

# Compute parareal solution
with open(folder_name + '/parareal_history_' + str(N_steps) + '_' + str(N_points) +'.txt', 'w') as outfile:
    outfile.write('# O parareal iteration: {0}\n')

adaptive_algo = Parareal(test_case, N_particles, N_steps, N_points, q_initial, v_initial, delta_conv, delta_expl, q_reference, v_reference, fine_propagator, coarse_propagator, white_noise, lmp)
adaptive_algo.run(folder_name)
adaptive_algo.output(folder_name)

##############################################################################

print('Normal end of execution')




