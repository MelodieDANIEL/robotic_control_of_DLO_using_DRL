import sys
import pybullet as p
import numpy as np
from numpy import random
import math

class Database_Frite:
	def __init__(self, path_load, load_name, path_generate, generate_name, nb_x=10, nb_y=30, nb_z=10):
		
		self.load_name = load_name
		self.generate_name = generate_name
		self.path_load = path_load
		self.path_generate = path_generate
		
		self.nb_x = nb_x
		self.nb_y = nb_y
		self.nb_z = nb_z
		
		self.nb_lines = 0
		self.nb_deformations = 0
		
		self.env = None
		self.data = None
		self.nb_points = None
		
		print("nb_x={}, nb_y={}, nb_z={}".format(self.nb_x,self.nb_y,self.nb_z))
	
	def set_env(self, env):
		self.env = env
		self.nb_points = len(self.env.id_frite_to_follow)
	
	
	def print_config(self):
		self.init_spaces()
		
		d_x = self.goal_high[0] - self.goal_low[0]
		d_y = self.goal_high[1] - self.goal_low[1]
		d_z = self.goal_high[2] - self.goal_low[2]
		
		print("****** CONFIG DATABASE ***************")
		print("nb_x={}, nb_y={}, nb_z={}".format(self.nb_x,self.nb_y,self.nb_z))
		print("d_x={}, d_y={}, d_z={}".format(d_x,d_y,d_z))
		print("step_x={}, step_y={}, step_z={}".format(self.step_x,self.step_y,self.step_z))
		print("range_x={}, range_y={}, range_z={}".format(self.range_x,self.range_y,self.range_z))
		print("delta_x={}, delta_y={}".format(self.delta_x,self.delta_y))
		print("**************************************")
		
		
	
	def init_spaces(self):
		
		self.gripper_position = self.env.get_gripper_position()
		self.goal_low = self.env.goal_space.low
		self.goal_high = self.env.goal_space.high
		
		#self.delta_x = math.ceil(((self.gripper_position[0]-self.goal_low[0])/(self.goal_high[0]-self.goal_low[0]))*self.nb_x) + 1
		#self.delta_y = math.ceil(((self.gripper_position[1]-self.goal_low[1])/(self.goal_high[1]-self.goal_low[1]))*self.nb_y) + 1
		
		
		self.step_x = float((self.goal_high[0]-self.goal_low[0])/(self.nb_x +1))
		self.step_y = float((self.goal_high[1]-self.goal_low[1])/(self.nb_y + 1))
		self.step_z = float((self.goal_high[2]-self.goal_low[2])/(self.nb_z + 1))
		
		
		self.delta_x = math.ceil((self.gripper_position[0]-self.goal_low[0])/self.step_x)
		self.delta_y = math.ceil((self.gripper_position[1]-self.goal_low[1])/self.step_y)
		
		print("delta_x={}, delta_y={}".format(self.delta_x,self.delta_y))

		self.range_x = self.nb_x + 1
		self.range_y = self.nb_y + 1
		self.range_z = self.nb_z + 1
		
		print("step_x={}, step_y={}, step_z={}".format(self.step_x,self.step_y,self.step_z))
			

	def debug_point(self, pt, offset = 0.1, width = 3.0, color = [1, 0, 0]):
		
		#print(pt)
		p.addUserDebugLine(lineFromXYZ          = [pt[0]+offset, pt[1], pt[2]]  ,
						   lineToXYZ            = [pt[0]-offset, pt[1], pt[2]],
						   lineColorRGB         = color  ,
						   lineWidth            = width        ,
						   lifeTime             = 0          )
						   
		p.addUserDebugLine(lineFromXYZ          = [pt[0], pt[1]+offset, pt[2]]  ,
						   lineToXYZ            = [pt[0], pt[1]-offset, pt[2]],
						   lineColorRGB         = color  ,
						   lineWidth            = width        ,
						   lifeTime             = 0          )
						   
	
	
	def debug_all_points(self):
		for i in range(self.nb_deformations):
			for j in range(self.nb_points):
				#for k in range(self.
				self.debug_point(pt=self.data[i,j], offset=0.01)
				p.stepSimulation()
			
	
	
	def debug_all_random_points(self, nb):
		for i in range(nb):
			a_pt = self.get_random_targets()
			for j in range(self.nb_points):
				self.debug_point(pt = a_pt[j], offset =0.01, color = [0, 0, 1])
		
			
	def get_random_targets(self):
		if self.data is not None:
			index = random.randint(self.nb_deformations-1)
			
			return self.data[index]
		else:
			return None
	
	def load(self):
		f = open(self.path_load + self.load_name)
		total_list = []
		for line in f.readlines():
			self.nb_lines+=1
			line_split = line.split()
			total_list.append(float(line_split[1])) #x
			total_list.append(float(line_split[2])) #y
			total_list.append(float(line_split[3]))	#z

		
		self.nb_deformations = int(self.nb_lines/self.nb_points)
		
		print("nb lines = {}, nb points = {}, nb_deformations = {}".format(self.nb_lines, self.nb_points,self.nb_deformations))
		self.data = np.array(total_list).reshape(self.nb_deformations, self.nb_points, 3)
		
		
		print("shape = {}".format(self.data.shape))	
		print(self.data[0,0])
			
		#print("id={}, x={}, y={}, z={}".format(int(line_split[0]), float(line_split[1]), float(line_split[2]), float(line_split[3])))
		
		
	def write_floats(self, f):
		list_id_frite = self.env.get_position_id_frite()
		ids_frite = self.env.id_frite_to_follow
		
		for k in range(len(list_id_frite)):
			f.write("{} {:.3f} {:.3f} {:.3f}\n".format(ids_frite[k], list_id_frite[k][0],  list_id_frite[k][1], list_id_frite[k][2]))
	
	def go_to_corner(self):
		# go to corner
		d_x_y_z= [-self.step_x, 0.0, 0.0]
		for x in range(self.delta_x):
			self.env.set_action_cartesian(d_x_y_z)
			p.stepSimulation()

		d_x_y_z= [0.0, -self.step_y, 0.0]
		for y in range(self.delta_y):
			self.env.set_action_cartesian(d_x_y_z)
			p.stepSimulation()

	
	def generate(self):
		self.init_spaces()
		self.go_to_corner()
		f = open(self.path_generate + self.generate_name, "w+")
   
		for j in range(self.range_z):
			
			print("plan {} / {}".format(j, self.range_z))
        
			for i in range(math.ceil(self.range_y/2)):
				# advance to x max
				d_x_y_z= [0.0, 0.0, 0.0]
				d_x_y_z[0] = self.step_x
				for x in range(self.range_x):
					self.env.set_action_cartesian(d_x_y_z)
					self.env.draw_gripper_position()
					p.stepSimulation()
					self.write_floats(f)
                        
            
				# 1 shift Y
				d_x_y_z = [0.0, 0.0, 0.0]
				d_x_y_z[1] = self.step_y
				self.env.set_action_cartesian(d_x_y_z)
				self.env.draw_gripper_position()
				p.stepSimulation()
				self.write_floats(f)
        
				# advance to x min
				d_x_y_z= [0.0, 0.0, 0.0]
				d_x_y_z[0] = -self.step_x
				for x in range(self.range_x):
					self.env.set_action_cartesian(d_x_y_z)
					self.env.draw_gripper_position()
					p.stepSimulation()
					self.write_floats(f)
        
				# 1 shift Y
				d_x_y_z = [0.0, 0.0, 0.0]
				d_x_y_z[1] = self.step_y
				self.env.set_action_cartesian(d_x_y_z)
				self.env.draw_gripper_position()
				p.stepSimulation()
				self.write_floats(f)
  
			# 1 shift z
			d_x_y_z= [0.0, 0.0, 0.0]
			d_x_y_z[2] = -self.step_z
        
			self.step_y*=-1
        
		f.close()
		
