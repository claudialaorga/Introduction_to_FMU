from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.ticker import ScalarFormatter



plt.rcParams['font.family'] = 'serif'
# Define the name of each FMU.
fmu5_name = 'path_to_your_Oscillator1_GS_DD_micro_step.fmu' #CHANGE THIS TO YOUR PATH 
fmu6_name = 'path_to_your_Oscillator2_GS_DD_micro_step.fmu'#CHANGE THIS TO YOUR PATH 
fmu3_name = 'path_to_your_Oscillator1_GS_DD_microstep.fmu'#CHANGE THIS TO YOUR PATH 
fmu4_name = 'path_to_your_Oscillator2_GS_DD_microstep.fmu'#CHANGE THIS TO YOUR PATH 
fmu1_name = 'path_to_your_Oscillator1_GS_DD.fmu'#CHANGE THIS TO YOUR PATH 
fmu2_name = 'path_to_your_Oscillator2_GS_DD.fmu'#CHANGE THIS TO YOUR PATH 

# Obtain information about these FMUs.
dump(fmu1_name)
dump(fmu2_name)
dump(fmu3_name)
dump(fmu4_name)
dump(fmu5_name)
dump(fmu6_name)

# Set simulation times.
start_time = 0.0
stop_time = 10.0
step_size = 0.001

# Read model description of each FMU.
model_description1 = read_model_description(fmu1_name)
model_description2 = read_model_description(fmu2_name)
model_description3 = read_model_description(fmu3_name)
model_description4 = read_model_description(fmu4_name)
model_description5 = read_model_description(fmu5_name)
model_description6 = read_model_description(fmu6_name)

# Get value references.
vrs1 = {variable.name: variable.valueReference for variable in model_description1.modelVariables}
vrs2 = {variable.name: variable.valueReference for variable in model_description2.modelVariables}
vrs3 = {variable.name: variable.valueReference for variable in model_description3.modelVariables}
vrs4 = {variable.name: variable.valueReference for variable in model_description4.modelVariables}
vrs5 = {variable.name: variable.valueReference for variable in model_description5.modelVariables}
vrs6 = {variable.name: variable.valueReference for variable in model_description6.modelVariables}

print(vrs1, 'Variables from the first .fmu file')
print(vrs2, 'Variables from the second .fmu file')
print(vrs3, 'Variables from the third .fmu file')
print(vrs4, 'Variables from the fourth .fmu file')
print(vrs5, 'Variables from the fifth .fmu file')
print(vrs6, 'Variables from the sixth .fmu file')

# Function to get value reference of a variable.
def get_value_reference(model_description, variable_name):
    for variable in model_description.modelVariables:
        if variable.name == variable_name:
            return variable.valueReference
    raise Exception(f"Variable {variable_name} not found in model")

# Save variable references of FMUs as outputs and inputs.
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

vr_fmu3_outputs = {
    'x_1': get_value_reference(model_description3, 'x_1'),
    'v_1': get_value_reference(model_description3, 'v_1')
}
vr_fmu3_inputs = {
    'x_2': get_value_reference(model_description3, 'x_2'),
}

vr_fmu4_outputs = {
    'x_2': get_value_reference(model_description4, 'x_2'),
    'v_2': get_value_reference(model_description4, 'v_2')
}
vr_fmu4_inputs = {
    'x_1': get_value_reference(model_description4, 'x_1'),
}

vr_fmu5_outputs = {
    'x_1': get_value_reference(model_description5, 'x_1'),
    'v_1': get_value_reference(model_description5, 'v_1')
}
vr_fmu5_inputs = {
    'x_2': get_value_reference(model_description5, 'x_2'),
}

vr_fmu6_outputs = {
    'x_2': get_value_reference(model_description6, 'x_2'),
    'v_2': get_value_reference(model_description6, 'v_2')
}
vr_fmu6_inputs = {
    'x_1': get_value_reference(model_description6, 'x_1'),
}

# Extract FMU files.
unzipdir1 = extract(fmu1_name)
unzipdir2 = extract(fmu2_name)
unzipdir3 = extract(fmu3_name)
unzipdir4 = extract(fmu4_name)
unzipdir5 = extract(fmu5_name)
unzipdir6 = extract(fmu6_name)

# Instantiate FMU2Slave objects.
fmu1 = FMU2Slave(guid=model_description1.guid,
                 unzipDirectory=unzipdir1,
                 modelIdentifier=model_description1.coSimulation.modelIdentifier,
                 instanceName='instance1')

fmu2 = FMU2Slave(guid=model_description2.guid,
                 unzipDirectory=unzipdir2,
                 modelIdentifier=model_description2.coSimulation.modelIdentifier,
                 instanceName='instance2')

fmu3 = FMU2Slave(guid=model_description3.guid,
                 unzipDirectory=unzipdir3,
                 modelIdentifier=model_description3.coSimulation.modelIdentifier,
                 instanceName='instance3')

fmu4 = FMU2Slave(guid=model_description4.guid,
                 unzipDirectory=unzipdir4,
                 modelIdentifier=model_description4.coSimulation.modelIdentifier,
                 instanceName='instance4')

fmu5 = FMU2Slave(guid=model_description5.guid,
                 unzipDirectory=unzipdir5,
                 modelIdentifier=model_description5.coSimulation.modelIdentifier,
                 instanceName='instance5')

fmu6 = FMU2Slave(guid=model_description6.guid,
                 unzipDirectory=unzipdir6,
                 modelIdentifier=model_description6.coSimulation.modelIdentifier,
                 instanceName='instance6')

# Initialize FMUs.
for fmu in [fmu1, fmu2, fmu3, fmu4, fmu5, fmu6]:
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

x1_valuess = []
v1_valuess = []
x2_values_fmu3 = []
x2_values_fmu4 = []
v2_values_fmu3 = []
v2_values_fmu4 = []
EM_valuess = []
time_points = []


x1_values_5 = []
v1_values_5 = []
x2_values_fmu5 = []
x2_values_fmu6 = []
v2_values_fmu5 = []
v2_values_fmu6 = []
EM_values_5 = []

f_c_val = []

# Define constants for calculating mechanical energy.
m_1 = 1.0  # kg
m_2 = 1.0  # kg
k_1 = 10.0  # N/m
k_c = 100.0  # N/m
k_2 = 1000.0  # N/m

# Mass and stiffness matrices.
M = np.array([[m_1, 0], [0, m_2]])
K = np.array([[k_1 + k_c, -k_c], [-k_c, k_c + k_2]])

# Simulation loop.
while time < stop_time:
    # Step and get output values from fmu1, fmu3, and fmu5.
    fmu1.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu3.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu5.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    x1_value = fmu1.getReal([vr_fmu1_outputs['x_1']])[0]
    v1_value = fmu1.getReal([vr_fmu1_outputs['v_1']])[0]
    x1_valuee = fmu3.getReal([vr_fmu3_outputs['x_1']])[0]
    v1_valuee = fmu3.getReal([vr_fmu3_outputs['v_1']])[0]
    x1_value_5 = fmu5.getReal([vr_fmu5_outputs['x_1']])[0]
    v1_value_5 = fmu5.getReal([vr_fmu5_outputs['v_1']])[0]

    # Set input values for fmu2, fmu4, and fmu6.
    fmu2.setReal([vr_fmu2_inputs['x_1']], [x1_value])
    fmu4.setReal([vr_fmu4_inputs['x_1']], [x1_valuee])
    fmu6.setReal([vr_fmu6_inputs['x_1']], [x1_value_5])

    # Step and get output values from fmu2, fmu4, and fmu6.
    fmu2.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu4.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu6.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    x2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['x_2']])[0]
    v2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['v_2']])[0]
    x2_value_fmu4 = fmu4.getReal([vr_fmu4_outputs['x_2']])[0]
    v2_value_fmu4 = fmu4.getReal([vr_fmu4_outputs['v_2']])[0]
    x2_value_fmu6 = fmu6.getReal([vr_fmu6_outputs['x_2']])[0]
    v2_value_fmu6 = fmu6.getReal([vr_fmu6_outputs['v_2']])[0]

    # Set input values for fmu1, fmu3, and fmu5.
    fmu1.setReal([vr_fmu1_inputs['x_2']], [x2_value_fmu2])
    fmu3.setReal([vr_fmu3_inputs['x_2']], [x2_value_fmu4])
    fmu5.setReal([vr_fmu5_inputs['x_2']], [x2_value_fmu6])

    # Get input values from fmu1, fmu3, and fmu5.
    x2_value_fmu1 = fmu1.getReal([vr_fmu1_inputs['x_2']])[0]
    x2_value_fmu3 = fmu3.getReal([vr_fmu3_inputs['x_2']])[0]
    x2_value_fmu5 = fmu5.getReal([vr_fmu5_inputs['x_2']])[0]

    f_c = k_c * (x1_value_5 - x2_value_fmu6)

    # Calculate mechanical energy.
    EM_value = (0.5 * np.dot([v1_value, v2_value_fmu2], np.dot(M, [v1_value, v2_value_fmu2])) +
                0.5 * np.dot([x1_value, x2_value_fmu2], np.dot(K, [x1_value, x2_value_fmu2])))

    EM_valuee = (0.5 * np.dot([v1_valuee, v2_value_fmu4], np.dot(M, [v1_valuee, v2_value_fmu4])) +
                 0.5 * np.dot([x1_valuee, x2_value_fmu4], np.dot(K, [x1_valuee, x2_value_fmu4])))

    EM_valueee = (0.5 * np.dot([v1_value_5, v2_value_fmu6], np.dot(M, [v1_value_5, v2_value_fmu6])) +
                  0.5 * np.dot([x1_value_5, x2_value_fmu6], np.dot(K, [x1_value_5, x2_value_fmu6])))

    # Store results.
    x1_values.append(x1_value)
    v1_values.append(v1_value)
    x2_values_fmu1.append(x2_value_fmu1)
    x2_values_fmu2.append(x2_value_fmu2)
    v2_values_fmu2.append(v2_value_fmu2)
    EM_values.append(EM_value)

    x1_valuess.append(x1_valuee)
    v1_valuess.append(v1_valuee)
    x2_values_fmu3.append(x2_value_fmu3)
    x2_values_fmu4.append(x2_value_fmu4)
    v2_values_fmu3.append(v2_value_fmu4)
    EM_valuess.append(EM_valuee)

    x1_values_5.append(x1_value_5)
    v1_values_5.append(v1_value_5)
    x2_values_fmu5.append(x2_value_fmu5)
    x2_values_fmu6.append(x2_value_fmu6)
    v2_values_fmu5.append(v2_value_fmu6)
    EM_values_5.append(EM_valueee)
    f_c_val.append(f_c)
    time_points.append(time)

    
    time += step_size

print('Returned', len(EM_values), 'points')
print('Returned', len(EM_valuess), 'points')

# Terminate and free FMU instances.
for fmu in [fmu1, fmu2, fmu3, fmu4, fmu5, fmu6]:
    fmu.terminate()
    fmu.freeInstance()

# Remove extracted directories.
shutil.rmtree(unzipdir1)
shutil.rmtree(unzipdir2)
shutil.rmtree(unzipdir3)
shutil.rmtree(unzipdir4)
shutil.rmtree(unzipdir5)
shutil.rmtree(unzipdir6)

######################### Analytical Solution #################

# Initial values
m1 = 1
m2 = 1
k1 = 10
k2 = 1000
kc = 100
c1 = 0.0
c2 = 0.0
cc = 0.00
xx1 = 100
xx2 = -100
x1 = 0
x2 = 0
xxx1 = 0
xxx2 = 0

# Constant matrices
Z0 = np.array([x1, x2, xx1, xx2])
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-(k1 + kc) / m1, kc / m1, -(c1 - cc / m1), cc / m1],
    [kc / m2, -(k2 + kc) / m2, cc / m2, -(c2 + cc) / m2]
])
M = np.array([[m1, 0], [0, m2]])
K = np.array([[k1 + kc, -kc], [-kc, kc + k2]])

# Time parameters
time_step = 0.001
time_end = 10
time = np.arange(0, time_end + time_step, time_step)
num_steps = len(time)

# Initialize arrays to store results
EM_anal = np.zeros(num_steps)
XX_values = np.zeros((2, num_steps))
X_values = np.zeros((2, num_steps))
FC_anal = np.zeros(num_steps)

# Main loop
for i in range(num_steps):
    t = time[i]  # Current time
    exp = expm(A * t)  # Exponential of matrix A multiplied by time
    Z = exp.dot(Z0)

    x_1 = Z[0]
    x_2 = Z[1]
    xx_1 = Z[2]
    xx_2 = Z[3]

    X = np.array([x_1, x_2])  # Position
    XX = np.array([xx_1, xx_2])  # Velocity

    f_c = kc * (x_1 - x_2)
    FC_anal[i] = f_c

    EM = 0.5 * XX.T.dot(M).dot(XX) + 0.5 * X.T.dot(K).dot(X)
    EM_anal[i] = EM

    XX_values[:, i] = XX
    X_values[:, i] = X

# Plot x_1 and x_2
plt.figure()
plt.plot(time, XX_values[1,:], label='Analytical Solution', color='blue')
plt.plot(time_points[:len(v2_values_fmu2)], v2_values_fmu2, label='Subsystem 2 position (Micropass=0.00001s, Macropass=0.1s', color='green',  linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()


# Figura para EM
plt.figure()
plt.plot(time, EM_anal, label='Energía mecánica analítica (Tiempo de paso 0.001))', color='red', linestyle='-')

plt.plot(time_points[:len(EM_values)], EM_values, label='Energía mecánica co-simulada (Micropaso=0.001s, Macropaso=0.1s)', color='blue', linestyle='-')
plt.plot(time_points[:len(EM_valuess)], EM_valuess, label='Energía mecánica co-simulada (Micropaso=0.0001s, Macropaso=0.1s)', linestyle='--')
plt.plot(time_points[:len(EM_values_5)], EM_values_5, label='Energía mecánica co-simulada (Micropaso=0.00001s, Macropaso=0.1s)',linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('Energía mecánica (J)')
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

plt.legend()
plt.show()



# Figura para FC
plt.figure()
plt.plot(time, FC_anal, label='Fuerza transmitida analítica ', color='purple', linestyle='-')


plt.plot(time_points[:len(f_c_val)], f_c_val, label='Fuerza transmitida co-simulada (Micropaso=0.00001s, Macropaso=0.1s)',linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('Fuerza transmitida (N)')
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

plt.legend()
plt.show()



