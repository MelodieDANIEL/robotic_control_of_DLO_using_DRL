python main.py --mode generate_database --generate_db_dir_name '/extra_small/'  --db_nb_x 5 --db_nb_y 20 --db_nb_z 5


****** CONFIG DATABASE ***************
nb_x=5, nb_y=20, nb_z=5
d_x=0.15000009536743164, d_y=0.5, d_z=0.25
step_x=0.02500001589457194, step_y=0.023809523809523808, step_z=0.041666666666666664
range_x=6, range_y=21, range_z=6
delta_x=4, delta_y=11
**************************************


		# EXTRA SMALL
		low_marge = 0.1
		low_x_down = panda_eff_state[0][0]-1.0*low_marge
		low_x_up = panda_eff_state[0][0]+0.5*low_marge
		
		low_y_down = panda_eff_state[0][1]-2.5*low_marge
		low_y_up = panda_eff_state[0][1]+2.5*low_marge
		
		
		z_low_marge = 0.25
		low_z_down = panda_eff_state[0][2]-z_low_marge
		low_z_up = panda_eff_state[0][2]
		
		
self.id_frite_to_follow = [53, 129, 101, 179, 21, 165]
