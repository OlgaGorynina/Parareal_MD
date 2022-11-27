from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM
import numpy as np 
from propagation import Propagation

class Parareal:
	"""Run Adaptive Parareal"""
		
	def __init__(self, test_case, N_particles, N_steps, N_points, q_initial, v_initial, delta_conv, delta_expl, q_reference, v_reference, fine_propagator, coarse_propagator, white_noise, lmp):
		""" Create parareal algorithm """
		self.test_case = test_case
		self.N_particles = N_particles
		self.N_steps = N_steps
		self.N_points = N_points
		self.q_initial = q_initial
		self.v_initial = v_initial
		self.q_reference = q_reference
		self.v_reference = v_reference
		self.white_noise = white_noise
		self.lmp = lmp
		self.fine_propagator = fine_propagator
		self.coarse_propagator = coarse_propagator
		
		# adaptive parameters
		self.N_init = 0
		self.N_final = N_steps
		self.N_expl = self.N_final
		self.N_adaptations = 0 #counter of adaptations

		self.delta_conv = delta_conv
		self.delta_expl = delta_expl
		self.delta = .5*(self.delta_conv + self.delta_expl)
		self.cost = 0
		self.k = 0

		# total parareal solution
		self.q_total = list()
		self.v_total = list()
		
		# output information
		self.delta_total = list()
		self.delta_total_n = list()
		self.delta_total_d = list()

		self.delta_f_total = list()
		self.size_array = list()
		self.slab_array = list()
		self.cost_array = list()


	def run(self,folder_name):
		""" Run adaptive parareal  """
		print('Zero parareal iteration ')	

		iteration_0_solution = Propagation(self.N_steps, self.N_particles, self.N_points, "coarse", self.coarse_propagator, self.test_case, "parareal", folder_name)
		iteration_0_solution.run(0, self.N_steps, self.q_initial, self.v_initial, self.white_noise, self.lmp)

		q_parareal_0, v_parareal_0 = iteration_0_solution.return_solution()

		self.q_total.append(q_parareal_0)
		self.v_total.append(v_parareal_0)

		while self.N_init < self.N_steps:
			while self.delta_conv < self.delta and self.delta < self.delta_expl:
				print('Parareal iteration number: ' +str(self.k+1))
				self.cost += 1 

				q_previous = np.zeros((self.N_steps+1, self.N_particles, 3))
				v_previous = np.zeros((self.N_steps+1, self.N_particles, 3))

				q_previous = np.array(self.q_total[-1])
				v_previous = np.array(self.v_total[-1])

				q_parareal = q_previous
				v_parareal = v_previous
				
				delta_array = np.zeros(self.N_steps) #to store delta
				delta_array_n = np.zeros(self.N_steps)
				delta_array_d = np.zeros(self.N_steps)
				delta_f_array = np.zeros(self.N_steps) #to store delta
				delta_f_array_n = np.zeros(self.N_steps) #to store delta
				delta_f_array_d = np.zeros(self.N_steps) #to store delta
				size = np.zeros(self.N_steps)

				find = False # to do double break (final break from current while)

				delta_numerator = np.zeros(self.N_final) # numerator of delta 
				delta_denominator = np.zeros(self.N_final) # denominator of delta
				delta_f_numerator = np.zeros(self.N_final) # numerator of delta_f
				delta_f_denominator = np.zeros(self.N_final) # denominator of delta_f

				#fine propagation
				q_fine = np.zeros((self.N_steps+1, self.N_particles, 3))
				v_fine = np.zeros((self.N_steps+1, self.N_particles, 3))
				
				fine_solution = Propagation(self.N_steps, self.N_particles, self.N_points, "fine", self.fine_propagator, self.test_case, "parareal", folder_name)

				for i in range(self.N_init, self.N_final):
					fine_solution.run_one_step(i, q_previous[i], v_previous[i], self.white_noise, self.lmp)
				q_fine, v_fine = fine_solution.return_solution()

				#coarse propagation (old data)
				q_coarse = np.zeros((self.N_steps+1, self.N_particles, 3))
				v_coarse = np.zeros((self.N_steps+1, self.N_particles, 3))

				coarse_solution = Propagation(self.N_steps, self.N_particles, self.N_points, "coarse", self.coarse_propagator, self.test_case, "parareal", folder_name)
					
				for i in range(self.N_init,self.N_final):
					coarse_solution.run_one_step(i, q_previous[i], v_previous[i], self.white_noise, self.lmp)
				q_coarse, v_coarse = coarse_solution.return_solution()

				#coarse propagation (new data)
				q_coarse_new = np.zeros((self.N_steps+1, self.N_particles, 3))
				v_coarse_new = np.zeros((self.N_steps+1, self.N_particles, 3))

				coarse_new_solution = Propagation(self.N_steps, self.N_particles, self.N_points, "coarse", self.coarse_propagator, self.test_case, "parareal", folder_name)

				for i in range(self.N_init, self.N_final):
					coarse_new_solution.run_one_step(i, q_parareal[i], v_parareal[i], self.white_noise, self.lmp)
					q_coarse_new, v_coarse_new = coarse_new_solution.return_solution()
					
					q_parareal[i+1] = q_fine[i+1] - q_coarse[i+1] + q_coarse_new[i+1]
					v_parareal[i+1] = v_fine[i+1] - v_coarse[i+1] + v_coarse_new[i+1]

					# update of delta
					delta_numerator[i] = np.sum(np.linalg.norm(q_parareal[i+1]- self.q_total[-1][i+1], axis = 1))
					delta_denominator[i] = np.sum(np.linalg.norm(self.q_total[-1][i+1], axis = 1))
					delta_f_numerator[i] = np.sum(np.linalg.norm(q_parareal[i+1]- self.q_reference[i+1], axis = 1))
					delta_f_denominator[i] = np.sum(np.linalg.norm(self.q_reference[i+1], axis = 1))

					delta_array[i] = np.sum(delta_numerator)/np.sum(delta_denominator)
					self.delta = delta_array[i]
					
					delta_array_n[i] = np.sum(delta_numerator)
					delta_array_d[i] = np.sum(delta_denominator)
					delta_f_array[i] = np.sum(delta_f_numerator)/np.sum(delta_f_denominator)
					delta_f_array_n[i] = np.sum(delta_f_numerator)
					delta_f_array_d[i] = np.sum(delta_f_denominator)
					size[i] = 1
					
					### log output
					self.lmp.command('print "delta "')
					self.lmp.command('print '+str(delta_array[i]))
					self.lmp.command('print "delta_f "')
					self.lmp.command('print '+str(delta_f_array[i]))
					self.lmp.command('print "cost "')
					self.lmp.command('print '+str(self.cost))
					
					with open(folder_name + '/parareal_history_' + str(self.N_steps) + '_' + str(self.N_points) +'.txt', 'a') as outfile:
						outfile.write('\n # '+ str(self.k+1) + ' parareal iteration, time window : '+str(i))
						outfile.write('\n   # current delta : '+str(self.delta))
						outfile.write('\n   # N_init : '+str(self.N_init))
						outfile.write('\n   # N_final : '+str(self.N_final))

					if self.delta > self.delta_expl:
						self.N_expl = i + 1 
						find = True
						break 
					print('current delta : ', self.delta)

				self.k += 1 #still go to new iteration

				self.q_total.append(q_parareal)
				self.v_total.append(v_parareal)

				self.delta_total.append(delta_array)
				self.delta_total_n.append(delta_array_n)
				self.delta_total_d.append(delta_array_d)
				self.size_array.append(size)
				
				if find: 
					break
			if self.delta > self.delta_expl:
				#N_adaptations += 1 
				self.N_final = self.N_expl

				### log output
				self.lmp.command('print "N_init "')
				self.lmp.command('print '+str(self.N_init))
				self.lmp.command('print "N_final "')
				self.lmp.command('print '+str(self.N_final))
				
			else:
				self.slab_array.append(self.N_final - self.N_init)
				self.cost_array.append(self.k)

				self.N_init = self.N_final

				### log output
				self.lmp.command('print "N_init "')
				self.lmp.command('print '+str(self.N_init))
				self.lmp.command('print "N_steps "')
				self.lmp.command('print '+str(self.N_steps))
				self.lmp.command('print "cost "')
				self.lmp.command('print '+str(self.cost))
				
				self.N_final = self.N_steps
				if self.N_init >= self.N_steps:
					break
					
				# coarse propagator
				self.N_adaptations += 1 

				### log output
				self.lmp.command('print "N_adaptations just been changed too"')
				self.lmp.command('print '+str(self.N_adaptations))

				q_parareal = self.q_total[-1]
				v_parareal = self.v_total[-1]
				self.k = 0

				print('Zero parareal iteration ')	

				iteration_renew_0_solution = Propagation(self.N_steps, self.N_particles, self.N_points, "coarse", self.coarse_propagator, self.test_case, "parareal", folder_name)
				iteration_renew_0_solution.run(self.N_init, self.N_final, q_parareal[self.N_init], v_parareal[self.N_init], self.white_noise, self.lmp)
				q_parareal_new, v_parareal_new = iteration_renew_0_solution.return_solution()
				q_parareal[self.N_init:] = q_parareal_new[self.N_init:]
				v_parareal[self.N_init:] = v_parareal_new[self.N_init:]

				self.q_total.append(q_parareal)
				self.v_total.append(v_parareal)

			self.delta = .5*(self.delta_conv + self.delta_expl)
		print('N_adaptations ', self.N_adaptations)


	def output(self, folder_name):
		# Write q array to disk
		with open(folder_name + '/' + 'check_parareal_solution_q_adp.txt', 'w') as outfile:
			outfile.write('# Solution shape: {0}\n'.format(np.shape(self.q_total[-1])))
			
			for data_slice in self.q_total[-1]:

				np.savetxt(outfile, data_slice, fmt='%-7.5f')
				outfile.write('#\n Next popagator point\n')

		# Write total q total array to disk
		with open(folder_name + '/' + 'check_parareal_solution_total_q_adp.txt', 'w') as outfile:
			outfile.write('# Solution shape: {0}\n'.format(np.shape(self.q_total)))
			
			for data_slice in self.q_total:
				for smaller_slice in data_slice:
				    np.savetxt(outfile, smaller_slice, fmt='%-7.5f')
				    outfile.write('#\n Next popagator point\n')
				outfile.write('#\n Next parareal iteration \n')

		# Write v array to disk
		with open(folder_name + '/' + 'check_parareal_solution_v_adp.txt', 'w') as outfile:
			outfile.write('# Solution shape: {0}\n'.format(np.shape(self.v_total[-1])))
			
			for data_slice in self.v_total[-1]:

				np.savetxt(outfile, data_slice, fmt='%-7.5f')
				outfile.write('#\n Next popagator point\n')
			  
		# Write total v total array to disk
		with open(folder_name + '/' + 'check_parareal_solution_total_v_adp.txt', 'w') as outfile:
			outfile.write('# Solution shape: {0}\n'.format(np.shape(self.v_total)))
			
			for data_slice in self.v_total:
				for smaller_slice in data_slice:
				    np.savetxt(outfile, smaller_slice, fmt='%-7.5f')
				    outfile.write('#\n Next popagator point\n')
				outfile.write('#\n Next parareal iteration \n')

		# Write delta to disk
		with open(folder_name + '/' + 'check_parareal_solution_delta_adp.txt', 'w') as outfile:
			outfile.write('# Number of adaptations: {0}\n'.format(self.N_adaptations))
			outfile.write('# Total cost: {0}\n'.format(self.cost))
			outfile.write('# Number of steps: {0}\n'.format(self.N_steps))
			
			for data_slice_1, data_slice_2, data_slice_3 in zip(self.delta_total, self.delta_total_n, self.delta_total_d):
				outfile.write('#\n delta \n')
				np.savetxt(outfile, data_slice_1, fmt='%-14.12f')
				outfile.write('#\n delta_n \n')
				np.savetxt(outfile, data_slice_2, fmt='%-14.12f')
				outfile.write('#\n delta_d \n')
				np.savetxt(outfile, data_slice_3, fmt='%-14.12f')
				outfile.write('#\n Next parareal iteration\n')
				#outfile.write('#\n Next adaptation slice \n')
				
		# Write delta_f to disk
		with open(folder_name + '/' + 'check_parareal_solution_delta_f_adp.txt', 'w') as outfile:
			outfile.write('# Number of adaptations: {0}\n'.format(self.N_adaptations))
			outfile.write('# Total cost: {0}\n'.format(self.cost))
			
			for data_slice_1 in self.delta_f_total:
				outfile.write('#\n delta \n')
				np.savetxt(outfile, data_slice_1, fmt='%-14.12f')
				outfile.write('#\n Next parareal iteration\n')

		# Write all delta to plot it
		with open(folder_name + '/' + 'check_delta_plot_' + str(self.N_steps) + '_' + str(self.N_points) + '.txt', 'w') as outfile:
			for data_slice_1, data_slice_2 in zip(self.delta_total, self.delta_f_total):
				outfile.write(str(i+1)+' ')
				outfile.write(str(data_slice_1[-1])+' ')
				outfile.write(str(data_slice_2[-1])+' ')
				outfile.write('\n')

		# Write size to disk
		with open(folder_name + '/' + 'check_parareal_solution_size_adp.txt', 'w') as outfile:
			outfile.write('# Number of adaptations: {0}\n'.format(self.N_adaptations))
			outfile.write('# Total cost: {0}\n'.format(self.cost))
			
			for data_slice in self.size_array:
				np.savetxt(outfile, data_slice, fmt='%-14.12f')
				outfile.write('#\n Next parareal iteration\n')

		# Write slabs length to disk
		with open(folder_name + '/' + 'check_parareal_solution_slab_length.txt', 'w') as outfile:
			outfile.write('# Number of adaptations: {0}\n'.format(self.N_adaptations))
			outfile.write('# Total cost: {0}\n'.format(self.cost))
			outfile.write('# Slabs: {0}\n')
			np.savetxt(outfile, self.slab_array, fmt='%-14.12f')
			outfile.write('# Costs: {0}\n')
			np.savetxt(outfile, self.cost_array, fmt='%-14.12f')

