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

import pybullet as p

class Debug_Gui:
	
	def __init__(self, env = None):
		self.env = env
		self.dic_id = {}
	
	
	def draw_text(self, a_name, a_text = "", a_pos = [0,0,0], a_size = 1.5, a_color = [1, 0, 0]):
		if ( str(a_name) in self.dic_id.keys() ):
			p.removeUserDebugItem(self.dic_id[str(a_name)])
		self.dic_id[str(a_name)] = p.addUserDebugText(a_text, a_pos,textColorRGB=a_color,textSize=a_size)
		p.stepSimulation()
		
	def draw_cross(self, name, a_pos = [0,0,0], a_size = 0.1, a_color = [1, 0, 0], a_width = 3.0, a_time = 0):
		
		if ( str(name)+"_x" in self.dic_id.keys() ):
			p.removeUserDebugItem(self.dic_id[str(name)+"_x"])
			
		if ( str(name)+"_y" in self.dic_id.keys() ):
			p.removeUserDebugItem(self.dic_id[str(name)+"_y"])	
		
		if ( str(name)+"_z" in self.dic_id.keys() ):
			p.removeUserDebugItem(self.dic_id[str(name)+"_z"])	
		
		self.dic_id[str(name)+"_x"] = p.addUserDebugLine(lineFromXYZ      = [a_pos[0]+a_size, a_pos[1], a_pos[2]]  ,
			   lineToXYZ            = [a_pos[0]-a_size, a_pos[1], a_pos[2]],
			   lineColorRGB         = a_color,
			   lineWidth            = a_width,
			   lifeTime             = a_time
				  )
			   
		self.dic_id[str(name)+"_y"] = p.addUserDebugLine(lineFromXYZ      = [a_pos[0], a_pos[1]+a_size, a_pos[2]]  ,
			   lineToXYZ            = [a_pos[0], a_pos[1]-a_size, a_pos[2]],
			   lineColorRGB         = a_color  ,
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )
					 
		self.dic_id[str(name)+"_z"] = p.addUserDebugLine(lineFromXYZ      = [a_pos[0], a_pos[1], a_pos[2]+a_size]  ,
			   lineToXYZ            = [a_pos[0], a_pos[1], a_pos[2]-a_size],
			   lineColorRGB         = a_color  ,
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )
			   
		p.stepSimulation()


	def draw_box(self, low , high , color = [0, 0, 1]):
		low_array = low
		high_array = high

		p1 = [low_array[0], low_array[1], low_array[2]] # xmin, ymin, zmin
		p2 = [high_array[0], low_array[1], low_array[2]] # xmax, ymin, zmin
		p3 = [high_array[0], high_array[1], low_array[2]] # xmax, ymax, zmin
		p4 = [low_array[0], high_array[1], low_array[2]] # xmin, ymax, zmin

		p.addUserDebugLine(p1, p2, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p2, p3, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p3, p4, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p4, p1, lineColorRGB=color, lineWidth=2.0, lifeTime=0)


		p5 = [low_array[0], low_array[1], high_array[2]] # xmin, ymin, zmax
		p6 = [high_array[0], low_array[1], high_array[2]] # xmax, ymin, zmax
		p7 = [high_array[0], high_array[1], high_array[2]] # xmax, ymax, zmax
		p8 = [low_array[0], high_array[1], high_array[2]] # xmin, ymax, zmax

		p.addUserDebugLine(p5, p6, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p6, p7, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p7, p8, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p8, p5, lineColorRGB=color, lineWidth=2.0, lifeTime=0)

		p.addUserDebugLine(p1, p5, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p2, p6, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p3, p7, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		p.addUserDebugLine(p4, p8, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
		
		p.stepSimulation()
