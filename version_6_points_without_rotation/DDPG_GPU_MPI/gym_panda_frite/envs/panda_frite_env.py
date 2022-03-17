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

import os, inspect
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

import math

import threading
import time

from gym_panda_frite.envs.debug_gui import Debug_Gui

class PandaFriteEnv(gym.Env):
	
	def __init__(self, database = None, distance_threshold = None, gui = None, reset_env = True):
		
		self.reset_env = reset_env
		self.database = database
		self.debug_lines_gripper_array = [0, 0, 0, 0]
		
		self.gui = gui
		
		# bullet paramters
		#self.timeStep=1./240
		self.timeStep = 0.003
		self.n_substeps = 20
		self.dt = self.timeStep*self.n_substeps
		self.max_vel = 1
		self.max_gripper_vel = 20
		self.startOrientation = p.getQuaternionFromEuler([0,0,0])
		
		self.id_debug_gripper_position = None
		self.id_debug_joints_values = None
		self.id_debug_frite_list = None
		self.id_debug_marker_frite_list = None
		self.id_save_button = None
		
		self.id_frite_to_follow = [53, 129, 101, 179, 21, 165]
		#self.debug_id_frite_to_follow = [[None,None],[None,None]]  # draw 2 lines (a cross) per id frite to follow
		
		self.debug_gui = Debug_Gui(env = self)
		
		self.distance_threshold=distance_threshold
		
		#print("PandaFriteEnv distance_threshold = {}".format(self.distance_threshold))
		
		self.seed()
		
		if self.gui == True:
			# connect bullet
			p.connect(p.GUI) #or p.GUI (for test) or p.DIRECT (for train) for non-graphical version
		else:
			p.connect(p.DIRECT)
		
		# switch tool to convert a numeric value to string value
		self.switcher_type_name = {
			p.JOINT_REVOLUTE: "JOINT_REVOLUTE",
			p.JOINT_PRISMATIC: "JOINT_PRISMATIC",
			p.JOINT_SPHERICAL: "JOINT SPHERICAL",
			p.JOINT_PLANAR: "JOINT PLANAR",
			p.JOINT_FIXED: "JOINT FIXED"
		}
		
		# effector index : 
		# i=7, name=panda_joint8, type=JOINT FIXED, lower=0.0, upper=-1.0, effort=0.0, velocity=0.0
		# i=8, name=panda_hand_joint, type=JOINT FIXED, lower=0.0, upper=-1.0, effort=0.0, velocity=0.0
		# i=9, name=panda_finger_joint1, type=JOINT_PRISMATIC, lower=0.0, upper=0.04, effort=20.0, velocity=0.2
		self.panda_end_eff_idx = 7
		
		# i=6, name=panda_joint7, type=JOINT_REVOLUTE, lower=-2.9671, upper=2.9671, effort=12.0, velocity=2.61
		self.panda_last_revolute_joint = 6
		
		self.database.set_env(self)
		self.database.load()
		
		self.initial_reset()
		
		self.panda_list_lower_limits, self.panda_list_upper_limits, self.panda_list_joint_ranges, self.panda_list_initial_poses = self.get_panda_joint_ranges()
			 
		# show sliders
		self.panda_joint_name_to_slider={}
		#self.show_sliders()
		
		#self.show_cartesian_sliders()
		
		p.stepSimulation()
		
		if self.gui == True:
			self.draw_thread = threading.Thread(target=self.loop_update_cross)
			self.draw_thread.start()


	def loop_update_cross(self):
		while True:
			self.draw_id_to_follow()
			
	def initial_reset(self):
		#p.resetSimulation()
		
		# Set Gravity to the environment
		#p.setGravity(0, 0, 0)
		
		# reset pybullet to deformable object
		p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

		# bullet setup
		# add pybullet path
		currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		#print("currentdir = {}".format(currentdir))
		p.setAdditionalSearchPath(currentdir)
		
		#p.setAdditionalSearchPath(pd.getDataPath())
		
		#p.setTimeStep(0.003)
		p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
		#p.setPhysicsEngineParameter(numSubSteps = self.n_substeps, fixedTimeStep = 0.0003, numSolverIterations = 500, useSplitImpulse = 1, erp = 0.1, solverResidualThreshold = 0.001, sparseSdfVoxelSize = 0.25)
		#p.setPhysicsEngineParameter(numSubSteps = self.n_substeps, fixedTimeStep = self.timeStep, numSolverIterations = 500)
		#p.setPhysicsEngineParameter(numSubSteps = self.n_substeps, fixedTimeStep = self.timeStep, numSolverIterations = 500, useSplitImpulse = 1, erp = 0.1, solverResidualThreshold = 0.001, sparseSdfVoxelSize = 0.25)
		p.setTimeStep(self.timeStep)

		# Set Gravity to the environment
		p.setGravity(0, 0, -9.81)
		#p.setGravity(0, 0, 0)
		
		# load plane
		self.load_plane()
		p.stepSimulation()

		# load table
		self.load_table()
		p.stepSimulation()
		
		#load panda
		self.load_panda()
		p.stepSimulation()
		
		# set panda joints to initial positions
		self.set_panda_initial_joints_positions()
		p.stepSimulation()
		
		# set gym spaces
		self.set_gym_spaces()
		
		# load frite
		self.load_frite()
		p.stepSimulation()
	
		# anchor frite to gripper
		self.create_anchor_panda()
		p.stepSimulation()
		
		# close gripper
		#self.close_gripper()
		p.stepSimulation()
	
	def draw_all_ids_mesh_frite(self):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		for i in range(data[0]):
			pos = data[1][i]
			if i in self.id_frite_to_follow:
				p.addUserDebugText(str(i), pos, textColorRGB=[1,0,0])
				#self.debug_gui.draw_cross("id_frite_"+str(i) , a_color = [0, 0, 1], a_pos = pos)
			else:
				p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
			
	def draw_ids_mesh_frite(self, a_from=0, a_to=0):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		for i in range(data[0]):
			if (i>=a_from and i<=a_to):
				pos = data[1][i]
				if i in self.id_frite_to_follow:
					p.addUserDebugText(str(i), pos, textColorRGB=[1,0,0])
				else:
					p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
	
	def get_gripper_position(self):
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		return gripper_pos
		
	def draw_text_joints_values(self):
		joints_values=[]
		for joint_name in self.panda_joint_name_to_ids.keys():
			if (joint_name != 'panda_finger_joint1' and joint_name != 'panda_finger_joint2'):
				joint_index = self.panda_joint_name_to_ids[joint_name]
				joints_values.append(p.getJointState(self.panda_id, joint_index)[0])
				
		str_joints_values = ""
		for i in range(len(joints_values)):
			str_joints_values+="q{}={:.3f}, ".format(i+1,joints_values[i])
			
		self.debug_gui.draw_text(a_name="joint_values" , a_text=str_joints_values, a_pos = [-1.0,0,0.5])

	def draw_text_gripper_position(self):
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		gripper_pos_str = "x={:.3f}, y={:.3f}, z={:.3f}".format(gripper_pos[0], gripper_pos[1], gripper_pos[2])
		self.debug_gui.draw_text(a_name="gripper_pos" , a_text=gripper_pos_str, a_pos = [-1.0,0,0.1])

	def draw_env_box(self):
		self.debug_gui.draw_box(self.pos_space.low, self.pos_space.high, [0, 0, 1])
		self.debug_gui.draw_box(self.goal_space.low, self.goal_space.high, [1, 0, 0])

	def draw_goal(self):
		for i in range(self.goal.shape[0]):
			self.debug_gui.draw_cross("goal_"+str(i) , a_pos = self.goal[i])
			
			
	def draw_gripper_position(self):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		self.debug_gui.draw_cross("gripper", a_pos = cur_pos, a_color = [0, 0, 1])
	
	
	def draw_id_to_follow(self):
		list_positions = self.get_position_id_frite()
		for i in range(len(self.id_frite_to_follow)):
			self.debug_gui.draw_cross("id_frite_"+str(i), a_pos = list_positions[i], a_color = [0, 0, 1])
			p.stepSimulation()
			
	def sample_goal(self):
		return self.database.get_random_targets()
	
	
	def show_cartesian_sliders(self):
		self.list_slider_cartesian = []
		self.list_slider_cartesian.append(p.addUserDebugParameter("VX", -1, 1, 0))
		self.list_slider_cartesian.append(p.addUserDebugParameter("VY", -1, 1, 0))
		self.list_slider_cartesian.append(p.addUserDebugParameter("VZ", -1, 1, 0))
		
				
	
	def apply_cartesian_sliders(self):
		action = np.empty(3, dtype=np.float64)
		
		for i in range(3):
			action[i] = p.readUserDebugParameter(self.list_slider_cartesian[i])
		
		self.set_action(action)
		p.stepSimulation()

	
	def set_gym_spaces(self):
		panda_eff_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		"""
		# LARGE
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-4*low_marge
		low_y_up = panda_eff_state[0][1]+4*low_marge
		
		
		z_low_marge = 0.3
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		"""
		# MEDIUM
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-3*low_marge
		low_y_up = panda_eff_state[0][1]+3*low_marge
		
		
		z_low_marge = 0.25
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		"""
		
		# SMALL
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.0*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-2.5*low_marge
		low_y_up = panda_eff_state[0][1]+2.5*low_marge
		
		
		z_low_marge = 0.25
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		
		
		self.goal_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		#print("frite env goal space = {}".format(self.goal_space))
		
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-2*low_marge
		low_x_up = panda_eff_state[0][0]+low_marge

		low_y_down = panda_eff_state[0][1]-5*low_marge
		low_y_up = panda_eff_state[0][1]+5*low_marge

		z_low_marge = 0.3
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		
		self.pos_space = spaces.Box(low=np.array([low_x_down, low_y_down ,low_z_down]), high=np.array([low_x_up, low_y_up ,low_z_up]))
		
		# action_space = cartesian world velocity (vx, vy, vz)  = 3 float
		self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
		
		# observation = 42 float -> see function _get_obs
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(42,), dtype=np.float32)

		
	def get_panda_joint_ranges(self):
		list_lower_limits, list_upper_limits, list_joint_ranges, list_initial_poses = [], [], [], []

		for item in self.panda_joint_name_to_ids.items():
			joint_name = item[0]
			joint_index = item[1] 
			joint_info = p.getJointInfo(self.panda_id, joint_index)

			a_lower_limit, an_upper_limit = joint_info[8:10]
			#print(item, ' ', a_lower_limit, ' ' , an_upper_limit)

			a_range = an_upper_limit - a_lower_limit

			# For simplicity, assume resting state = initial position
			an_inital_pose = self.panda_initial_positions[joint_name]

			list_lower_limits.append(a_lower_limit)
			list_upper_limits.append(an_upper_limit)
			list_joint_ranges.append(a_range)
			list_initial_poses.append(an_inital_pose)

		return list_lower_limits, list_upper_limits, list_joint_ranges, list_initial_poses
	
	def printPandaAllInfo(self):
		print("=================================")
		print("All Panda Robot joints info")
		num_joints = p.getNumJoints(self.panda_id)
		print("=> num of joints = {0}".format(num_joints))
		for i in range(num_joints):
			joint_info = p.getJointInfo(self.panda_id, i)
			#print(joint_info)
			joint_name = joint_info[1].decode("UTF-8")
			joint_type = joint_info[2]
			child_link_name = joint_info[12].decode("UTF-8")
			link_pos_in_parent_frame = p.getLinkState(self.panda_id, i)[0]
			link_orien_in_parent_frame = p.getLinkState(self.panda_id, i)[1]
			joint_type_name = self.switcher_type_name.get(joint_type,"Invalid type")
			joint_lower_limit, joint_upper_limit = joint_info[8:10]
			joint_limit_effort = joint_info[10]
			joint_limit_velocity = joint_info[11]
			print("i={0}, name={1}, type={2}, lower={3}, upper={4}, effort={5}, velocity={6}".format(i,joint_name,joint_type_name,joint_lower_limit,joint_upper_limit,joint_limit_effort,joint_limit_velocity))
			print("child link name={0}, pos={1}, orien={2}".format(child_link_name,link_pos_in_parent_frame,link_orien_in_parent_frame))
		print("=================================")

		
	def load_frite(self):
		panda_eff_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)

		self.frite_startPos = [panda_eff_state[0][0], panda_eff_state[0][1], 0.0]
		self.frite_startOrientation = self.startOrientation

		self.frite_id = p.loadSoftBody("vtk/tetra_cylinder_6.vtk", basePosition = self.frite_startPos, baseOrientation=self.frite_startOrientation, mass = 0.2, useNeoHookean = 1, NeoHookeanMu = 961500, NeoHookeanLambda = 1442300, NeoHookeanDamping = 0.01, useSelfCollision = 1, collisionMargin = 0.001, frictionCoeff = 0.5, scale=1.0)
		#p.changeVisualShape(self.frite_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)

	def load_plane(self):
		self.plane_id = p.loadURDF("urdf/plane.urdf", basePosition=[0,0,0], useFixedBase=True)
		
	def load_table(self):
		# load table
		self.table_startPos = [0, 0, 0.81]
		self.table_id = p.loadURDF("urdf/table/table.urdf", self.table_startPos, useFixedBase=True)
		self.table_height = 0.625

	def load_panda(self):
		# load panda
		self.panda_startPos = [0.7, 0, self.table_startPos[2]+self.table_height]
		self.panda_startOrientation = self.startOrientation

		self.panda_id = p.loadURDF("urdf/franka_panda/panda.urdf",
								   basePosition=self.panda_startPos, baseOrientation=self.panda_startOrientation, useFixedBase=True)

		self.panda_num_joints = p.getNumJoints(self.panda_id) # 12 joints

		#print("panda num joints = {}".format(self.panda_num_joints))

	def set_panda_initial_joints_positions(self, init_gripper = True):
		self.panda_joint_name_to_ids = {}
		
		# Set intial positions
		"""
		self.panda_initial_positions = {
		'panda_joint1': 0.0, 'panda_joint2': math.pi/4., 'panda_joint3': 0.0,
		'panda_joint4': -math.pi/2., 'panda_joint5': 0.0, 'panda_joint6': 3*math.pi/4,
		'panda_joint7': -math.pi/4., 'panda_finger_joint1': 0.08, 'panda_finger_joint2': 0.08,
		} 
		"""
		
		self.panda_initial_positions = {
		'panda_joint1': 0.0, 'panda_joint2': 0.287, 'panda_joint3': 0.0,
		'panda_joint4': -2.392, 'panda_joint5': 0.0, 'panda_joint6': 2.646,
		'panda_joint7': -0.799, 'panda_finger_joint1': 0.08, 'panda_finger_joint2': 0.08,
		}
		
		for i in range(self.panda_num_joints):
			joint_info = p.getJointInfo(self.panda_id, i)
			joint_name = joint_info[1].decode("UTF-8")
			joint_type = joint_info[2]
			
			if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
				assert joint_name in self.panda_initial_positions.keys()

				joint_type_name = self.switcher_type_name.get(joint_type,"Invalid type")
                    
				#print("joint {0}, type:{1} -> {2}".format(joint_name,joint_type,joint_type_name))
				self.panda_joint_name_to_ids[joint_name] = i
				
				if joint_name == 'panda_finger_joint1' or joint_name == 'panda_finger_joint2':
					if init_gripper:
						p.resetJointState(self.panda_id, i, self.panda_initial_positions[joint_name])
				else:
					p.resetJointState(self.panda_id, i, self.panda_initial_positions[joint_name])


	def show_sliders(self, prefix_name = '', joint_values=None):
		index = 0
		for item in self.panda_joint_name_to_ids.items():
			joint_name = item[0]
			if (joint_name != 'panda_finger_joint1' and joint_name != 'panda_finger_joint2'):
				joint_index = item[1]
				ll = self.panda_list_lower_limits[index]
				ul = self.panda_list_upper_limits[index]
				if joint_values != None:
					joint_value = joint_values[index]
				else:   
					joint_value = self.panda_initial_positions[joint_name]
				slider = p.addUserDebugParameter(prefix_name + joint_name, ll, ul, joint_value) # add a slider for that joint with the limits
				self.panda_joint_name_to_slider[joint_name] = slider
			index = index + 1 

	def apply_sliders(self):
		for joint_name in self.panda_joint_name_to_ids.keys():
			if (joint_name != 'panda_finger_joint1' and joint_name != 'panda_finger_joint2'):
				slider = self.panda_joint_name_to_slider[joint_name] # get the slider of that joint name
				slider_value = p.readUserDebugParameter(slider) # read the slider value
				joint_index = self.panda_joint_name_to_ids[joint_name]
				p.setJointMotorControl2(self.panda_id, joint_index, p.POSITION_CONTROL,
											targetPosition=slider_value,
											positionGain=0.2)
				p.stepSimulation()


	def close_gripper(self):
		id_finger_joint1  = self.panda_joint_name_to_ids['panda_finger_joint1']
		id_finger_joint2  = self.panda_joint_name_to_ids['panda_finger_joint2']

		p.setJointMotorControl2(self.panda_id, id_finger_joint1, 
								p.POSITION_CONTROL,targetPosition=0.023,
								positionGain=0.1)
								
		p.setJointMotorControl2(self.panda_id, id_finger_joint2 , 
								p.POSITION_CONTROL,targetPosition=0.023,
								positionGain=0.1)
								
    
	def create_anchor_panda(self):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		pose_frite_down = list(data[1][0])  # n° 0 down
		pose_frite_up = list(data[1][1])  # n° 1 up
		
		uid_down = p.addUserDebugText("*", pose_frite_down, textColorRGB=[0,0,0])
		uid_up = p.addUserDebugText("*", pose_frite_up, textColorRGB=[0,0,0])
		
		gripper_pos = p.getLinkState(self.panda_id, self.panda_end_eff_idx)[0]
		
		
		# mesh id = 0 -> down
		# mesh id = 1 -> up
		# -1, -1 -> means anchor to the plane
		p.createSoftBodyAnchor(self.frite_id, 0, self.plane_id , -1, [gripper_pos[0],gripper_pos[1],0])
		#p.createSoftBodyAnchor(self.frite_id, 0, -1 , -1)
		
		# panda urdf index=8
		# i=8, name=panda_hand_joint, type=JOINT FIXED, lower=0.0, upper=-1.0, effort=0.0, velocity=0.0
		p.createSoftBodyAnchor(self.frite_id, 1, self.panda_id , self.panda_end_eff_idx, [0,0,-(gripper_pos[2]-1.55)])
		

	def get_position_id_frite(self):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		
		pos_list = []

		for i in range(len(self.id_frite_to_follow)):
			pos_list.append(data[1][self.id_frite_to_follow[i]])
		
		return pos_list

		
	def set_action_cartesian(self, action):
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = cur_pos + np.array(action)
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=10 * 240.)
			
		
	def set_action(self, action):
		assert action.shape == (3,), 'action shape error'
		
		cur_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		cur_pos = np.array(cur_state[0])
		cur_orien = np.array(cur_state[1])
		
		new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		
		jointPoses = p.calculateInverseKinematics(self.panda_id, self.panda_end_eff_idx, new_pos, cur_orien)[0:7]
		
		for i in range(len(jointPoses)):
			p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, jointPoses[i],force=10 * 240.)
	
		if self.gui == True:	
			time.sleep(5)
	
	def get_obs(self):
		eff_link_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx, computeLinkVelocity=1)
		gripper_link_pos = np.array(eff_link_state[0]) # gripper cartesian world position = 3 float (x,y,z) = achieved goal
		gripper_link_vel = np.array(eff_link_state[6]) # gripper cartesian world velocity = 3 float (vx, vy, vz)
		
		id_frite_to_follow_pos = np.array(self.get_position_id_frite()).flatten()
		
		# self.goal = len(self.id_frite_to_follow = [53, 129, 101, 179, ?, ?]) x 3 values (x,y,z) cartesian world position = 18 floats
		# observation = 
		#  3 floats (x,y,z) gripper link cartesian world position  [0,1,2]
		# + 3 float (vx, vy, vz) gripper link cartesian world velocity [3,4,5]
		# + current cartesian world position of id frite to follow (18 floats) [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
		# + self.goal cartesian world position of id frite to reach (18 floats) [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
		# observation = 42 floats
		
		#print("goal = {}".format(self.goal))
		#print("goal flat = {}, id pos flat = {}".format(self.goal.flatten(), id_frite_to_follow_pos))
		
		obs = np.concatenate((
					gripper_link_pos, gripper_link_vel, id_frite_to_follow_pos, self.goal.flatten()
		))
		
		return obs
	
	def is_success(self, d):
		return (d < self.distance_threshold).astype(np.float32)
		
	def step(self, action):
		action = np.clip(action, self.action_space.low, self.action_space.high)
		new_gripper_pos = self.set_action(action)
		
		p.stepSimulation()
		
		obs = self.get_obs()

		done = True
		
		nb_id_frite_to_follow = len(self.id_frite_to_follow)
		
		sum_d = 0
		
		for i in range(nb_id_frite_to_follow):
			current_pos_id_frite = obs[(6+(i*3)):(6+(i*3)+3)]
			goal_pos_id_frite = self.goal[i]
			d =  np.linalg.norm(current_pos_id_frite - goal_pos_id_frite, axis=-1)
			sum_d+=d
			
		
		d = np.float32(sum_d/nb_id_frite_to_follow)

		info = {
			'is_success': self.is_success(d),
			'mean_distance_error' : d,
		}

		reward = -d
		if (d > self.distance_threshold):
			done = False

		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info
		
	def reset(self):
		
		if self.reset_env:
			print('reset env !')
			self.initial_reset()
		
		# sample a new goal
		self.goal = self.sample_goal()
		p.stepSimulation()
		
		# draw goal
		self.draw_goal()
		p.stepSimulation()
		
		return self.get_obs()
		
	def render(self):
		print("render !")
		
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
        

	
