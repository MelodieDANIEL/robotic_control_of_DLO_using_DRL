python main.py --mode generate_database --generate_db_dir_name '/db1/'  --db_nb_x 5 --db_nb_y 20 --db_nb_z 5


****** CONFIG DATABASE ***************
nb_x=5, nb_y=20, nb_z=5
d_x=0.20000004768371582, d_y=0.6000000238418579, d_z=0.25
step_x=0.0333333412806193, step_y=0.02857142970675514, step_z=0.041666666666666664
range_x=6, range_y=21, range_z=6
delta_x=5, delta_y=11
**************************************

def set_gym_spaces(self):
		panda_eff_state = p.getLinkState(self.panda_id, self.panda_end_eff_idx)
		
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.5*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-3*low_marge
		low_y_up = panda_eff_state[0][1]+3*low_marge
		
		
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
		
		# action_space = cartesian world velocity (vx, vy, vz) + gripper velocity (theta_dot) = 4 float
		self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
		
		# observation = 32 float -> see function _get_obs
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(32,), dtype=np.float32)


