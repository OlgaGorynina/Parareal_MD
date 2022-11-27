from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM
import numpy as np 

class Propagation:
	"""Propagation with lammps"""
	
	def __init__(self, N_steps, N_particles, N_points, integrator_indicator, integrator_value, test_case, output_name, folder_name):
		"""Initialization of the scheme"""
		self.integrator_indicator = integrator_indicator #
		self.integrator_value = integrator_value
		self.test_case = test_case
		self.output_name = output_name

		self.N_particles = N_particles
		self.N_steps = N_steps
		self.N_points = N_points

		self.q_solution = np.zeros((self.N_steps+1, self.N_particles, 3)) #
		self.v_solution = np.zeros((self.N_steps+1, self.N_particles, 3)) #
		self.folder_name = folder_name

	def run_one_step(self, index, q_previous, v_previous, white_noise, lmp):
		"""Run one step of MD scheme"""

		print(self.integrator_indicator + ' propagation iteration number : '+str(index))
		lmp.command('clear')
		lmp.file(self.test_case + "simulation_box")
		lmp.command('create_atoms 1 single 0 1.59015 0 # units box')
		lmp.command('variable seed equal ' + str(white_noise[index]))
		print('current white noise value : ' + str(int(white_noise[index])))
		lmp.command('print ${seed}')
		# potential propagation setting
		lmp.command('variable propagator equal ' + str(self.integrator_value))
		#to check if we work with EAM potential
		
		if self.integrator_value == 0: 
			lmp.file(self.test_case +"potential_EAM")
		else: 
			lmp.file(self.test_case +"potential")
		lmp.command('print ${Tcorrected}')

		
		if self.integrator_indicator == "fine" and self.output_name == "parareal":
			#lmp.command('variable index equal ' + str(index+1))
			lmp.command('dump mydump all custom 10 '+ self.folder_name + '/' + self.output_name + '_solution/dump_output_'+ str(index+1).zfill(10) + '.dat id type xs ys zs ix iy iz')
		if self.integrator_indicator == "fine" and self.output_name == "reference":
			#lmp.command('variable index equal ' + str(index+1))
			lmp.command('dump mydump all custom 10 '+ self.folder_name + '/' + self.output_name +'_solution/dump_output_reference_'+ str(index+1).zfill(10) + '.dat id type xs ys zs ix iy iz')
		
		q_f = lmp.extract_atom("x",3)
		for j in range (0, self.N_particles):
			for l in range (0, 3):
				q_f[j][l] = q_previous[j][l]
				
		v_f = lmp.extract_atom("v",3)
		for j in range (0, self.N_particles):
			for l in range (0, 3):
				v_f[j][l] = v_previous[j][l]
 
		lmp.command('run '+str(self.N_points))
			
		self.q_solution[index + 1] = np.array(lmp.numpy.extract_atom("x"))
		self.v_solution[index + 1] = np.array(lmp.numpy.extract_atom("v"))

	def run(self, step_index_begin, step_index_end, q_previous, v_previous, white_noise, lmp):
		"""Run MD scheme from setp_index_begon to step_index_end"""
		self.q_solution[step_index_begin] = q_previous
		self.v_solution[step_index_begin] = v_previous
		
		for i in range(step_index_begin, step_index_end):
			self.run_one_step(i, self.q_solution[i], self.v_solution[i], white_noise, lmp)

	def return_solution(self):
		return self.q_solution, self.v_solution

	def write_output(self, folder_name):
		"""write output"""
		# Write q array to disk
		with open(folder_name + '/' + self.output_name +'_solution_q_adp.txt', 'w') as outfile:
			outfile.write('# Solution shape: {0}\n'.format(self.q_solution.shape))
			
			for data_slice in self.q_solution:

				np.savetxt(outfile, data_slice, fmt='%-7.5f')
				outfile.write('#\n Next popagator point\n')

		# Write v array to disk
		with open(folder_name + '/' + self.output_name +'_solution_v_adp.txt', 'w') as outfile:
			outfile.write('# Solution shape: {0}\n'.format(self.v_solution.shape))
			
			for data_slice in self.v_solution:

				np.savetxt(outfile, data_slice, fmt='%-7.5f')
				outfile.write('#\n Next popagator point\n')

