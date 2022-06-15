"""
Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Mélodie HANI DANIEL ZAKARIA
PhD Student
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.hani_daniel_zakaria@uca.fr
"""

import sys
import pybullet as p
import numpy as np
from numpy import random
import math

class Database_Frite:
	def __init__(self, path_load, load_name):
		
		self.load_name = load_name
		self.path_load = path_load
		
		self.nb_lines = 0
		self.nb_deformations = 0
		
		self.env = None
		self.data = None
		self.nb_points = None
		
	
	def set_env(self, env):
		self.env = env
		self.nb_points = len(self.env.id_frite_to_follow)
	
	
	def print_config(self):
		print("****** CONFIG DATABASE ***************")
		print("nb lines = {}, nb points = {}, nb_deformations = {}".format(self.nb_lines, self.nb_points,self.nb_deformations))
		print("shape = {}".format(self.data.shape))	
		print("**************************************")
		
		
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
		
