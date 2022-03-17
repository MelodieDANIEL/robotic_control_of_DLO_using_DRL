<br>The paper "Robotic Control of the Deformation of Soft Linear Objects Using Deep Reinforcement Learning", written by Mélodie Hani Daniel Zakaria<a href="#note1" id="note1ref"><sup>1</sup></a>, Miguel Aranda<a href="#note2" id="note2ref"><sup>2</sup></a>, Laurent Lequièvre<a href="#note1" id="note1ref"><sup>1</sup></a>, Sébastien Lengagne<a href="#note1" id="note1ref"><sup>1</sup></a>, Juan Antonio Corrales Ramón<a href="#note3" id="note3ref"><sup>3</sup></a> and Youcef Mezouar<a href="#note1" id="note1ref"><sup>1</sup></a>, has been submitted to CASE 2022.
<br>
<br>We address the problem of controlling the deformation of a DLO using a robot arm that manipulates it. For simplicity, the robot grasps one end of the object, and the other end is fixed to the ground. The object is represented by a mesh and we describe its deformation by a set of selected mesh nodes. The objective is to control the arm so that the positions of the selected nodes are driven to prescribed values. The difficulty of this indirect control problem lies in the fact that the dynamical model of the system to be controlled is complex and uncertain. We propose a generalizable architecture to solve this problem based on DRL.



## How to install virtualenv on ubuntu 20.04

virtualenv is a tool to create lightweight “virtual environments” with their own site directories isolated from system site directories.
Each "virtual environment" has its own Python binary (which matches the version of the binary that was used to create this environment) 
and can have its own independent set of installed Python packages in its site directories.


sudo apt install python3-pip
<br>pip3 install virtualenv

## How to create a virtualenv named 've_rl' and activate it

virtualenv ve_rl --python=python3
<br>source ve_rl/bin/activate

## Then necessary python3 packages can be installed into that virtualenv

pip install --upgrade pip

<br>pip install gym
<br>pip install torch
<br>pip install matplotlib
<br>pip install mpi4py
<br>pip install pybullet

## Details of 'main.py' parameters

n : number of mpi processors requested.
<br>gui : boolean to show or not the gui.
<br>max_episode : number of episodes.
<br>max_step : number of episode steps.
<br>log_interval : frequency of saving neural weights, every 'log_interval' episodes.
<br>max_memory_size : size of DDPG Agent memory.
<br>batch_size : size of the trained tuples.
<br>distance_threshold : distance necessary to success the goal.
<br>save_dir_name : name of neural weights directory (create it when it doesn't exist).
<br>generate_database_name : name of generated 'goal' database (by default 'database_id_frite.txt').
<br>load_database_name : name of loaded 'goal' database (by default 'database_id_frite.txt').
<br>load_db_dir_name : directory name of 'goal' database loaded.
<br>db_nb_x : how to divide the 'goal space' on x to generate a 'goal' database.
<br>db_nb_y : how to divide the 'goal space' on y to generate a 'goal' database.
<br>db_nb_z : how to divide the 'goal space' on z to generate a 'goal' database.
<br>random_seed : value to initialize the random number generator
<br>generate_db_dir_name : directory name of generated database 
<br>reset_env : boolean to reset the bullet env for each episode

## How to train

cd DDPG_GPU_MPI
<br>mpirun -n 32 python main.py --max_episode 63 --max_step 300 --log_interval 10 --save_dir_name './w32_s_t005/' --load_db_dir_name '/small/' --distance_threshold 0.05

<br>The database used is a file named 'database_id_frite.txt' by default in the directory 'DDPG_GPU_MPI/databases/small'
<br>The neural network weights will be saved in the directory 'DDPG_GPU_MPI/w32_s_t005'

## How to test

cd DDPG_GPU_MPI
<br>python main.py --mode test --gui True --save_dir_name './w32_s_t005/' --load_db_dir_name   '/large/'  --distance_threshold 0.05
<br>python main.py --mode test --gui True --save_dir_name './w32_s_t005/' --load_db_dir_name   '/large/'  --distance_threshold 0.05 --reset_env True


## How to generate a database

cd DDPG_GPU_MPI
<br>python main.py --mode generate_database --generate_db_dir_name '/small/'  --db_nb_x 5 --db_nb_y 20 --db_nb_z 5

<br>Create a file named 'database_id_frite.txt' in the directory 'DDPG_GPU_MPI/databases/small'.
<br>
<br><a id="note1" href="#note1ref"><sup>1</sup></a>CNRS, Clermont Auvergne INP, Institut Pascal,  Université Clermont Auvergne, Clermont-Ferrand, France.
<br><a id="note2" href="#note2ref"><sup>2</sup></a>Instituto de Investigación en Ingeniería de Aragón, Universidad de Zaragoza, Zaragoza, Spain.
<br><a id="note3" href="#note3ref"><sup>3</sup></a>Centro Singular de Investigación en Tecnoloxías Intelixentes (CiTIUS),  Universidade de Santiago de Compostela, Santiago de Compostela, Spain.
