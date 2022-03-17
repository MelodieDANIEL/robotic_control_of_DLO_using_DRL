****** CONFIG DATABASE ***************
nb_x=8, nb_y=22, nb_z=10
d_x=0.20000004768371582, d_y=0.7999999523162842, d_z=0.2999999523162842
step_x=0.02222222752041287, step_y=0.034782606622447136, step_z=0.027272722937844017
range_x=9, range_y=23, range_z=11
delta_x=7, delta_y=12
**************************************


self.id_frite_to_follow = [53, 129, 101, 179]

def set_gym_spaces(self):
		panda_eff_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-4*low_marge
		low_y_up = panda_eff_state[0][1]+4*low_marge
		
		
		z_low_marge = 0.3
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
		
		# action_space = cartesian world velocity (vx, vy, vz) + gripper velocity (theta_dot) = 4 float
		self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
		
		# observation = 32 float -> see function _get_obs
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(32,), dtype=np.float32)
		
		

