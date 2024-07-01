# cmc-salamandre
Computational Motor Control of Salamandra Robot.

**Date:** Feb-June 2022\
**Course:** CS-432 - Computational Motor Control (Prof. Auke Ijspeert)\
**Collaborators:** Flore Barde, Jeannette Gommendy, Louise Genoud

## General aim of the code

The code is part of a project aiming at developping code to control the walking and swimming of a salamander-like robot. In order to do so, we replicated and studied the Central Pattern Generator (CPG) network, as implemented in previous papers.

## Structure of the code

- **project1/2.py**: A convenient file for running the entire project. Note you can also run the different exercises in parallel by activating parallel=True.
- **exercise_all.py** - Another convenient file for running all or specified exercises depending on
arguments provided. You do not need to modify this file.
- **network.py**: This file contains the different classes and functions for the CPG network and
the Ordinary Differential Equations (ODEs). You can implement the network parameters and
the ODEs here. Note that some parameters can be obtained from robot_parameters.py to help
you control the values.
- **robot_parameters.py**: This file contains the different classes and functions for the parameters
of the robot, including the CPG network parameters. You can implement the network parameters
here. Note that some parameters can be obtained from the SimulationParameters class in
simulation_parameters.py and provided in exercise_#.py to help you control the values (refer
to example.py).
- **simulation_parameters.py**: This file contains the SimulationParameters class and is provided for convenience to send parameters to the setup of the network in network.py::SalamandraNetwork
via the robot parameters in robot_parameters.py::RobotParameters. The SimulationParameters is also intended to be used for experiment-specific parameters for the exercises. All the
values provided in SimulationParameters are logged for each simulation, so you can also reload
these parameters when analyzing the results of an experiment.
- **run_network.py**: By running the script from Python, MuJoCo will be bypassed and you
will run the network without a physics simulation. Make sure to use this file for question 8a
to help you with setting up the CPG network equations and parameters and to analyze its
behavior. This is useful for debugging purposes and rapid controller development since running
the MuJoCo simulation takes more time.
- **exercise_example.py**: Contains the example code structure to help you familiarize with the
other exercises. You do not need to modify this file.
- **exercise_#.py**: To be used to implement and answer the respective exercise questions. Note
that exercise_example.py is provided as an example to show how to run a parameter sweep.
Note that network parameters can be provided here.
- **exercise_all.py**: A convenient file for running different exercises depending on arguments. See
project1.py for an example on how to call it. You do not need to modify this file.
- **plot_results.py**: Use this file to load and plot the results from the simulation. This code runs
with the original example provided and provides examples on how to collect the data.
- **salamandra_simulation** folder: Contains all the remaining scripts for setting up and running
the simulation experiments. 
