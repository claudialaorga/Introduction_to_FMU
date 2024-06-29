from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
import numpy as np
import shutil
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'serif'

# Define the path to each FMU. This is what you have to use when you want to get information in the dump (not the FMU2Slave)
fmu1_path = 'path_to_your_oscillator1.fmu' #CHANGE THIS TO YOUR PATH
fmu2_path = 'path_to_your_oscillator2.fmu'#CHANGE THIS TO YOUR PATH

# Dump the information of these FMUs
dump(fmu1_path)
dump(fmu2_path)

# Set simulation times
start_time = 0.0
stop_time = 10.0
step_size = 0.001 #Macrostep; change this as long as you want to compare between macrosteps.

# Read the model descriptions
model_description_fmu1 = read_model_description(fmu1_path)
model_description_fmu2 = read_model_description(fmu2_path)

# Collect the value references
vrs_fmu1 = {variable.name: variable.valueReference for variable in model_description_fmu1.modelVariables}
vrs_fmu2 = {variable.name: variable.valueReference for variable in model_description_fmu2.modelVariables}

print(vrs_fmu1, 'These are the variables of the first .fmu file')
print(vrs_fmu2, 'These are the variables of the second .fmu file')

# Function to get the value reference (outputs and inputs) of variables
def get_value_reference(model_description, variable_name):
    for variable in model_description.modelVariables:
        if variable.name == variable_name:
            return variable.valueReference
    raise Exception(f"Variable {variable_name} not found in model")

# Save FMU1 variable references as outputs and inputs
vr_fmu1_outputs = {
    'x_1': get_value_reference(model_description_fmu1, 'x_1'),
    'v_1': get_value_reference(model_description_fmu1, 'v_1'),
    'FC': get_value_reference(model_description_fmu1, 'FC')
}
vr_fmu1_inputs = {
    'x_2': get_value_reference(model_description_fmu1, 'x_2'),
}

# Save FMU2 variable references as outputs and inputs
vr_fmu2_outputs = {
    'x_2': get_value_reference(model_description_fmu2, 'x_2'),
    'v_2': get_value_reference(model_description_fmu2, 'v_2')
}
vr_fmu2_inputs = {
    'FC': get_value_reference(model_description_fmu2, 'FC'),
}

# Extract the FMU files
unzipdir_fmu1 = extract(fmu1_path)
unzipdir_fmu2 = extract(fmu2_path)

# Instantiate FMUs
fmu1 = FMU2Slave(guid=model_description_fmu1.guid,
                unzipDirectory=unzipdir_fmu1,
                modelIdentifier=model_description_fmu1.coSimulation.modelIdentifier,
                instanceName='instance1')

fmu2 = FMU2Slave(guid=model_description_fmu2.guid,
                unzipDirectory=unzipdir_fmu2,
                modelIdentifier=model_description_fmu2.coSimulation.modelIdentifier,
                instanceName='instance2')

# Initialize FMUs
fmu1.instantiate()
fmu2.instantiate()

fmu1.setupExperiment(startTime=start_time)
fmu2.setupExperiment(startTime=start_time)

fmu1.enterInitializationMode()
fmu2.enterInitializationMode()

fmu1.exitInitializationMode()
fmu2.exitInitializationMode()

time = start_time

x1_values = []
v1_values = []
FC_values = []
x2_values_fmu1 = []
x2_values_fmu2 = []
v2_values_fmu1 = []
v2_values_fmu2 = []
time_points = []
EM_values = []

# Define constants to compute mechanical energy
m_1 = 1.0  # kg
m_2 = 1.0  # kg
k_1 = 10.0  # N/m
k_c = 100.0  # N/m
k_2 = 1000.0  # N/m

# Matrices M and K
M = np.array([[m_1, 0], [0, m_2]])
K = np.array([[k_1 + k_c, -k_c], [-k_c, k_c + k_2]])

# Simulation loop
while time < stop_time:

    # Perform step in FMU1
    fmu1.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    # Get outputs from FMU1
    x1_value = fmu1.getReal([vr_fmu1_outputs['x_1']])[0]
    v1_value = fmu1.getReal([vr_fmu1_outputs['v_1']])[0]
    FC_value = fmu1.getReal([vr_fmu1_outputs['FC']])[0]

    # Set inputs for FMU2
    fmu2.setReal([vr_fmu2_inputs['FC']], [FC_value])

    # Perform step in FMU2
    fmu2.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    # Get outputs from FMU2
    x2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['x_2']])[0]
    v2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['v_2']])[0]

    # Set inputs for FMU1 based on outputs from FMU2
    fmu1.setReal([vr_fmu1_inputs['x_2']], [x2_value_fmu2])

    # Get inputs for FMU1 (for storing results in the next loop)
    x2_value_fmu1 = fmu1.getReal([vr_fmu1_inputs['x_2']])[0]

    # Calculate mechanical energy
    EM_value = (0.5 * np.dot([v1_value, v2_value_fmu2], np.dot(M, [v1_value, v2_value_fmu2])) +
                0.5 * np.dot([x1_value, x2_value_fmu1], np.dot(K, [x1_value, x2_value_fmu1])))

    # Store results for plotting
    x1_values.append(x1_value)
    v1_values.append(v1_value)
    x2_values_fmu1.append(x2_value_fmu1)
    x2_values_fmu2.append(x2_value_fmu2)
    v2_values_fmu2.append(v2_value_fmu2)
    FC_values.append(FC_value)
    EM_values.append(EM_value)
    time_points.append(time)

    # Increment time for the next loop iteration
    time += step_size

# Terminate and clean up
fmu1.terminate()
fmu2.terminate()

fmu1.freeInstance()
fmu2.freeInstance()

shutil.rmtree(unzipdir_fmu1)
shutil.rmtree(unzipdir_fmu2)

# Plot for x_1 and x_2
plt.figure()
plt.plot(time_points[:len(x1_values)], x1_values, label='Position subsystem 1')
plt.plot(time_points[:len(x2_values_fmu1)], x2_values_fmu1, label='Position subsystem 2')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()

# Plot for v_1 and v_2
plt.figure()
plt.plot(time_points[:len(v1_values)], v1_values, label='Velocity Subsystem 1')
plt.plot(time_points[:len(v2_values_fmu2)], v2_values_fmu2, label='Velocity Subsystem 2', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Plot for EM
plt.figure()
plt.plot(time_points[:len(EM_values)], EM_values, label='Mechanical Energy (co-simulation)', color='purple', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Mechanical Energy (J)')
plt.legend()
plt.show()

# Plot for FC
plt.figure()
plt.plot(time_points[:len(FC_values)], FC_values, label='Transmitted Force', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.show()
