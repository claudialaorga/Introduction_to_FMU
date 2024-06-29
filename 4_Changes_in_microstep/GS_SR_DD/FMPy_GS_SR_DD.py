from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
import numpy as np
import shutil
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'


# Define the name for each FMU. This is what you need to provide when you want to inspect the information in the dump (not FMU2Slave's name)
nombre_fmu1 = 'C:/Users/claud/python/Co_simulation_FMI_Python/GS_SR_DD/GS_SR_DD_FMU/Oscillator1_GS_DD.fmu'
nombre_fmu2 = 'C:/Users/claud/python/Co_simulation_FMI_Python/GS_SR_DD/GS_SR_DD_FMU/Oscillator2_GS_DD.fmu'

# Inspect the information of these FMUs
dump(nombre_fmu1)
dump(nombre_fmu2)

# Set simulation times
start_time = 0.0
stop_time = 10.0
step_size = 0.001

# read the model description
model_description = read_model_description(nombre_fmu1)
model_description2 = read_model_description(nombre_fmu2)


# collect the value references
vrs = {variable.name: variable.valueReference for variable in model_description.modelVariables}  # From the model description, obtain variables and associate them with a port
vrs2 = {variable.name: variable.valueReference for variable in model_description2.modelVariables}  # From the model description, obtain variables and associate them with a port

print(vrs, 'These are the variables of the first .fmu file')
print(vrs2, 'These are the variables of the second .fmu file')


# Get the value references (outputs and inputs) of the variables
def get_value_reference(model_description, variable_name):
    for variable in model_description.modelVariables:
        if variable.name == variable_name:
            return variable.valueReference
    raise Exception(f"Variable {variable_name} not found in model")


# Save the reference variables of FMU1 as outputs and inputs
vr_fmu1_outputs = {
    'x_1': get_value_reference(model_description, 'x_1'),
    'v_1': get_value_reference(model_description, 'v_1')
}
vr_fmu1_inputs = {
    'x_2': get_value_reference(model_description, 'x_2'),
}

# Save the reference variables of FMU2 as outputs and inputs
vr_fmu2_outputs = {
    'x_2': get_value_reference(model_description2, 'x_2'),
    'v_2': get_value_reference(model_description2, 'v_2')
}
vr_fmu2_inputs = {
    'x_1': get_value_reference(model_description2, 'x_1'),
}



# Extract the FMU files
unzipdir = extract(nombre_fmu1)
unzipdir2 = extract(nombre_fmu2)


fmu1 = FMU2Slave(guid=model_description.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=model_description.coSimulation.modelIdentifier,
                instanceName='instance1')

fmu2 = FMU2Slave(guid=model_description2.guid,
                unzipDirectory=unzipdir2,
                modelIdentifier=model_description2.coSimulation.modelIdentifier,
                instanceName='instance2')



# Initialize the FMUs. Very important to provide them with outputs and inputs later
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
x2_values_fmu1 = []
x2_values_fmu2 = []
v2_values_fmu1 = []
v2_values_fmu2 = []
time_points = []
EM_values = []

# Define constants to calculate mechanical energy
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

    # Start the first step in FMU1. This is actually the Jacobi displacement-displacement scheme, which gives us the same
    fmu1.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    # Get the output values from fmu1
    x1_value = fmu1.getReal([vr_fmu1_outputs['x_1']])[0]  # Retrieve x_1 information from FMU
    v1_value = fmu1.getReal([vr_fmu1_outputs['v_1']])[0]  # Retrieve v_1 information from FMU

    # Inputs values for fmu2
    fmu2.setReal([vr_fmu2_inputs['x_1']], [x1_value])  # Set the input to subsystem 2
   
    # Step for subsystem 2 with its new inputs
    fmu2.doStep(currentCommunicationPoint=time, communicationStepSize=step_size) 
    
    # Get the output values from fmu2
    x2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['x_2']])[0]  # Obtain its outputs
    v2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['v_2']])[0]

    

    # With the output values of fmu2, set the input values for fmu1
    fmu1.setReal([vr_fmu1_inputs['x_2']], [x2_value_fmu2])  # Set the inputs to subsystem 1 based on the outputs of subsystem 2

    # Get the input values for fmu1 (to store results) for the next loop
    x2_value_fmu1 = fmu1.getReal([vr_fmu1_inputs['x_2']])[0]
   
    EM_value = (0.5 * np.dot([v1_value, v2_value_fmu2], np.dot(M, [v1_value, v2_value_fmu2])) +
                        0.5 * np.dot([x1_value, x2_value_fmu2], np.dot(K, [x1_value, x2_value_fmu2])))  # Initial mechanical energy value


    
    # Store the results for plotting later
    x1_values.append(x1_value)
    v1_values.append(v1_value)
    x2_values_fmu1.append(x2_value_fmu1)
    x2_values_fmu2.append(x2_value_fmu2)
    v2_values_fmu2.append(v2_value_fmu2)
    EM_values.append(EM_value)
    time_points.append(time)

    # Increment time to continue the loop. Important to do this last because there's an array of time points
    time += step_size
   

 # Terminate and clean up
fmu1.terminate()
fmu2.terminate()

fmu1.freeInstance()
fmu2.freeInstance()

shutil.rmtree(unzipdir)
shutil.rmtree(unzipdir2)

# Figure for x_1 and x_2
plt.figure()
plt.plot(time_points[:len(x1_values)], x1_values, label='Subsystem 1 Position', color='blue')
plt.plot(time_points[:len(x2_values_fmu1)], x2_values_fmu1, label='Subsystem 2 Position', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()

# Figure for v_1 and v_2
plt.figure()
plt.plot(time_points[:len(v1_values)], v1_values, label='Subsystem 1 Velocity', color='blue')
plt.plot(time_points[:len(v2_values_fmu2)], v2_values_fmu2, label='Subsystem 2 Velocity', linestyle='-', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Figure for EM
plt.figure()
plt.plot(time_points[:len(EM_values)], EM_values, label='Mechanical Energy (co-simulation)', color='purple', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Mechanical Energy (J)')
plt.legend()
plt.show()

