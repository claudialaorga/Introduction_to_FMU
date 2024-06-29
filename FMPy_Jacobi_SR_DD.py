from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
import numpy as np
import shutil
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

# Define the name of each FMU.
name_fmu1 = 'C:path_to_yor_oscillator1.fmu' #CHANGE THIS PART TO YOUR OWN OSCILLATOR1.FMU 
name_fmu2 = 'C:path_to_yor_oscillator2.fmu' #CHANGE THIS PART TO YOUR OWN OSCILLATOR2.FMU


# Get information about these FMUs
dump(name_fmu1)
dump(name_fmu2)


# Set simulation times
start_time = 0.0
stop_time = 10.0
step_size = 0.001 #Macrostep, change this as long as you want to compare between macrosteps

# Read the model description
model_description1 = read_model_description(name_fmu1)
model_description2 = read_model_description(name_fmu2)


# Get value references
vrs1 = {variable.name: variable.valueReference for variable in model_description1.modelVariables}
vrs2 = {variable.name: variable.valueReference for variable in model_description2.modelVariables}

print(vrs1, 'Variables from the first .fmu file')
print(vrs2, 'Variables from the second .fmu file')

# Get value references (outputs and inputs) of the variables
def get_value_reference(model_description, variable_name):
    for variable in model_description.modelVariables:
        if variable.name == variable_name:
            return variable.valueReference
    raise Exception(f"Variable {variable_name} not found in model")

# Save value references of the FMUs as outputs and inputs
vr_fmu1_outputs = {
    'x_1': get_value_reference(model_description1, 'x_1'),
    'v_1': get_value_reference(model_description1, 'v_1')
}
vr_fmu1_inputs = {
    'x_2': get_value_reference(model_description1, 'x_2'),
}

vr_fmu2_outputs = {
    'x_2': get_value_reference(model_description2, 'x_2'),
    'v_2': get_value_reference(model_description2, 'v_2')
}
vr_fmu2_inputs = {
    'x_1': get_value_reference(model_description2, 'x_1'),
}

# Extract FMU files
unzipdir1 = extract(name_fmu1)
unzipdir2 = extract(name_fmu2)

fmu1 = FMU2Slave(guid=model_description1.guid,
                 unzipDirectory=unzipdir1,
                 modelIdentifier=model_description1.coSimulation.modelIdentifier,
                 instanceName='instance1')

fmu2 = FMU2Slave(guid=model_description2.guid,
                 unzipDirectory=unzipdir2,
                 modelIdentifier=model_description2.coSimulation.modelIdentifier,
                 instanceName='instance2')



# Initialize the FMUs
for fmu in [fmu1, fmu2]:
    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

time = start_time

x1_values = []
v1_values = []
x2_values_fmu1 = []
x2_values_fmu2 = []
v2_values_fmu1 = []
v2_values_fmu2 = []
EM_values = []
time_points = []


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
    # Step and get output values from fmu1
    fmu1.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    x1_value = fmu1.getReal([vr_fmu1_outputs['x_1']])[0]
    v1_value = fmu1.getReal([vr_fmu1_outputs['v_1']])[0]
    # Set input values for fmu2 and fmu4
    fmu2.setReal([vr_fmu2_inputs['x_1']], [x1_value])

    # Step and get output values from fmu2
    fmu2.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    x2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['x_2']])[0]
    v2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['v_2']])[0]

    # Set input values for fmu1
    fmu1.setReal([vr_fmu1_inputs['x_2']], [x2_value_fmu2])

    # Get input values from fmu1
    x2_value_fmu1 = fmu1.getReal([vr_fmu1_inputs['x_2']])[0]

    # Calculate mechanical energy
    EM_value = (0.5 * np.dot([v1_value, v2_value_fmu2], np.dot(M, [v1_value, v2_value_fmu2])) +
                0.5 * np.dot([x1_value, x2_value_fmu1], np.dot(K, [x1_value, x2_value_fmu1])))

    # Store results
    x1_values.append(x1_value)
    v1_values.append(v1_value)
    x2_values_fmu1.append(x2_value_fmu1)
    x2_values_fmu2.append(x2_value_fmu2)
    v2_values_fmu2.append(v2_value_fmu2)
    EM_values.append(EM_value)

    time_points.append(time)
    # Increment time to continue the loop. Important to do this last because of the time array
    time += step_size


# Free the FMUs
for fmu in [fmu1, fmu2]:
    fmu.terminate()
    fmu.freeInstance()

# Remove temporary directories
for dir in [unzipdir1, unzipdir2]:
    shutil.rmtree(dir, ignore_errors=True)

# Figure for x_1 
plt.figure()
plt.plot(time_points[:len(x1_values)], x1_values, label='Position subsystem 1', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()

# Figure for x_2
plt.figure()
plt.plot(time_points[:len(x2_values_fmu1)], x2_values_fmu1, label='Position subsystem 2', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()

# Figure for v_2
plt.figure()
plt.plot(time_points[:len(v2_values_fmu2)], v2_values_fmu2, label='Velocity subsystem 2', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Figure for v_1 
plt.figure()
plt.plot(time_points[:len(v1_values)], v1_values, label='Velocity subsystem 1', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Figure for EM
plt.figure()
plt.plot(time_points[:len(EM_values)], EM_values, label='Jacobi single-step displacement-displacement (Microstep=0.001s, Macrostep=0.001s)', color='purple', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Mechanical Energy (J)')
plt.legend()
plt.show()
