from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.ticker import ScalarFormatter



plt.rcParams['font.family'] = 'serif'

# Define the name of each FMU.
name_fmu1 = 'path_to_your_Oscillator_1_Jacobi_DD.fmu' #CHANGE TO YOUR PATH 
name_fmu2 = 'path_to_your_/Oscillator2_Jacobi_DD.fmu' #CHANGE TO YOUR PATH 
name_fmu3 = 'path_to_your_Oscillator_1_Jacobi_DD_microstep.fmu' #CHANGE TO YOUR PATH 
name_fmu4 = 'path_to_your_Oscillator2_Jacobi_DD_microstep.fmu' #CHANGE TO YOUR PATH 
name_fmu5 = 'path_to_your_Oscillator_1_Jacobi_DD_micro_step.fmu'#CHANGE TO YOUR PATH 
name_fmu6 ='path_to_your_Oscillator2_Jacobi_DD_micro_step.fmu' #CHANGE TO YOUR PATH 

# Know the information of these FMUs
dump(name_fmu1)
dump(name_fmu2)
dump(name_fmu3)
dump(name_fmu4)
dump(name_fmu5)
dump(name_fmu6)

# Set the simulation times
start_time = 0.0
stop_time = 10.0
step_size = 0.001 #Macrostep, CHANGE THIS AS LONG AS YOU WANT TO COMPARE BETWEEN

# Read the model description
model_description1 = read_model_description(name_fmu1)
model_description2 = read_model_description(name_fmu2)
model_description3 = read_model_description(name_fmu3)
model_description4 = read_model_description(name_fmu4)
model_description5 = read_model_description(name_fmu5)
model_description6 = read_model_description(name_fmu6)

# Get the value references
vrs1 = {variable.name: variable.valueReference for variable in model_description1.modelVariables}
vrs2 = {variable.name: variable.valueReference for variable in model_description2.modelVariables}
vrs3 = {variable.name: variable.valueReference for variable in model_description3.modelVariables}
vrs4 = {variable.name: variable.valueReference for variable in model_description4.modelVariables}
vrs5 = {variable.name: variable.valueReference for variable in model_description5.modelVariables}
vrs6 = {variable.name: variable.valueReference for variable in model_description6.modelVariables}


# Print the value references of the FMUs as outputs and inputs

print(vrs1, 'Variables del primer archivo .fmu')
print(vrs2, 'Variables del segundo archivo .fmu')
print(vrs3, 'Variables del tercer archivo .fmu')
print(vrs4, 'Variables del cuarto archivo .fmu')
print(vrs5, 'Variables del quinto archivo .fmu')
print(vrs6, 'Variables del sexto archivo .fmu')

# Get value reference of the model variables
def get_value_reference(model_description, variable_name):
    for variable in model_description.modelVariables:
        if variable.name == variable_name:
            return variable.valueReference
    raise Exception(f"Variable {variable_name} not found in model")

# Save FMU variables as outputs and inputs
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



# Extract FMU files
unzipdir1 = extract(name_fmu1)
unzipdir2 = extract(name_fmu2)
unzipdir3 = extract(name_fmu3)
unzipdir4 = extract(name_fmu4)
unzipdir5 = extract(name_fmu5)
unzipdir6 = extract(name_fmu6)


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

# Initialize the FMUs
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
time_points = []
EM_valuess = []

x1_values_5 = []
v1_values_5 = []
x2_values_fmu5 = []
x2_values_fmu6 = []
v2_values_fmu6 = []
EM_values_5 = []

m_1 = 1.0  # kg
m_2 = 1.0  # kg
k_1 = 10.0  # N/m
k_c = 100.0  # N/m
k_2 = 1000.0  # N/m

M = np.array([[m_1, 0], [0, m_2]])
K = np.array([[k_1 + k_c, -k_c], [-k_c, k_c + k_2]])

# Loop through the simulation
while time < stop_time:
    # Step of fmu1 and fmu3
    fmu1.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu3.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu5.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    #get the values of outputs of fmu1, fmu3, fmu5

    x1_value = fmu1.getReal([vr_fmu1_outputs['x_1']])[0]
    v1_value = fmu1.getReal([vr_fmu1_outputs['v_1']])[0]
    x1_valuee = fmu3.getReal([vr_fmu3_outputs['x_1']])[0]
    v1_valuee = fmu3.getReal([vr_fmu3_outputs['v_1']])[0]
    x1_value_5 = fmu5.getReal([vr_fmu5_outputs['x_1']])[0]
    v1_value_5 = fmu5.getReal([vr_fmu5_outputs['v_1']])[0]

    # Set the inputs of fmu2, fmu4 and fmu6
    fmu2.setReal([vr_fmu2_inputs['x_1']], [x1_value])
    fmu4.setReal([vr_fmu4_inputs['x_1']], [x1_valuee])
    fmu6.setReal([vr_fmu6_inputs['x_1']], [x1_value_5])


    # Step of fmu2, fmu4 and fmu6
    fmu2.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu4.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu6.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    # Get the outputs of fmu2, fmu4 and fmu6

    x2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['x_2']])[0]
    v2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['v_2']])[0]
    x2_value_fmu4 = fmu4.getReal([vr_fmu4_outputs['x_2']])[0]
    v2_value_fmu4 = fmu4.getReal([vr_fmu4_outputs['v_2']])[0]
    x2_value_fmu6 = fmu6.getReal([vr_fmu6_outputs['x_2']])[0]
    v2_value_fmu6 = fmu6.getReal([vr_fmu6_outputs['v_2']])[0]


    # Set the inputs of fmu1, fmu3 and fmu5
    fmu1.setReal([vr_fmu1_inputs['x_2']], [x2_value_fmu2])
    fmu3.setReal([vr_fmu3_inputs['x_2']], [x2_value_fmu4])
    fmu5.setReal([vr_fmu5_inputs['x_2']], [x2_value_fmu6])


    x2_value_fmu1 = fmu1.getReal([vr_fmu1_inputs['x_2']])[0]
    x2_value_fmu3 = fmu3.getReal([vr_fmu3_inputs['x_2']])[0]
    x2_value_fmu5 = fmu5.getReal([vr_fmu5_inputs['x_2']])[0]


    # Calculate the mechanic energy
    EM_value = (0.5 * np.dot([v1_value, v2_value_fmu2], np.dot(M, [v1_value, v2_value_fmu2])) +
                0.5 * np.dot([x1_value, x2_value_fmu2], np.dot(K, [x1_value, x2_value_fmu2])))
    
    EM_valuee = (0.5 * np.dot([v1_valuee, v2_value_fmu4], np.dot(M, [v1_valuee, v2_value_fmu4])) +
                0.5 * np.dot([x1_valuee, x2_value_fmu4], np.dot(K, [x1_valuee, x2_value_fmu4])))
    
    
    EM_valueee = (0.5 * np.dot([v1_value_5, v2_value_fmu6], np.dot(M, [v1_value_5, v2_value_fmu6])) +
                0.5 * np.dot([x1_value_5, x2_value_fmu6], np.dot(K, [x1_value_5, x2_value_fmu6])))

    # Save the results 
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
    v2_values_fmu4.append(v2_value_fmu4)
    EM_valuess.append(EM_valuee)

    x1_values_5.append(x1_value_5)
    v1_values_5.append(v1_value_5)
    x2_values_fmu5.append(x2_value_fmu5)
    x2_values_fmu6.append(x2_value_fmu6)
    v2_values_fmu6.append(v2_value_fmu6)
    EM_values_5.append(EM_valueee)


    time_points.append(time)
    time += step_size


fmu1.terminate()
fmu2.terminate()
fmu3.terminate()
fmu4.terminate()
fmu5.terminate()
fmu5.terminate()

fmu1.freeInstance()
fmu2.freeInstance()
fmu3.freeInstance()
fmu4.freeInstance()
fmu5.freeInstance()
fmu6.freeInstance()

shutil.rmtree(unzipdir1)
shutil.rmtree(unzipdir2)
shutil.rmtree(unzipdir3)
shutil.rmtree(unzipdir4)
shutil.rmtree(unzipdir5)
shutil.rmtree(unzipdir6)

#########################Analytic solution#################
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

#Constant matrix
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

# Matrix initialization
EM_anal = np.zeros(num_steps)
XX_values = np.zeros((2, num_steps))
X_values = np.zeros((2, num_steps))
FC_anal = np.zeros(num_steps)

# Principal loop
for i in range(num_steps):
    t = time[i]  # Actual time
    exp = expm(A * t) 
    Z = exp.dot(Z0)

    x_1 = Z[0]
    x_2 = Z[1]
    xx_1 = Z[2]
    xx_2 = Z[3]

    X = np.array([x_1, x_2])  # Position
    XX = np.array([xx_1, xx_2])  # Speed

    f_c = kc * (x_1 - x_2)
    FC_anal[i] = f_c

    EM = 0.5 * XX.T.dot(M).dot(XX) + 0.5 * X.T.dot(K).dot(X)
    EM_anal[i] = EM

    XX_values[:, i] = XX
    X_values[:, i] = X
# Figure for x_1
plt.figure()
plt.plot(time, X_values[0, :], label='Analytical (Integration Time 0.001s)', color='purple', linestyle='-')
#plt.plot(time_points[:len(x1_values)], x1_values, label='Subsystem 1 Position (Microstep=0.001s, Macrostep=0.001s)', color='blue')
#plt.plot(time_points[:len(x1_valuess)], x1_valuess, label='Subsystem 1 Position (Microstep=0.0001s, Macrostep=0.001s)', color='green', linestyle='-')
plt.plot(time_points[:len(x1_values_5)], x1_values_5, label='FMPy (Microstep=0.00001s, Macrostep=0.1s)', color='blue', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()

# Figure for x_2
plt.figure()
plt.plot(time, X_values[1, :], label='Analytical (Integration Time 0.001s)', color='purple', linestyle='-')

#plt.plot(time_points[:len(x2_values_fmu1)], x2_values_fmu1, label='Subsystem 2 Position (Microstep=0.001s, Macrostep=0.001s)', color='red')
#plt.plot(time_points[:len(x2_values_fmu3)], x2_values_fmu3, label='Subsystem 2 Position (Microstep=0.0001s, Macrostep=0.001s)', color='blue', linestyle='-')
plt.plot(time_points[:len(x2_values_fmu5)], x2_values_fmu5, label='FMPy (Microstep=0.00001s, Macrostep=0.1s)', color='green', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()

# Figure for v_1
plt.figure()
plt.plot(time, XX_values[0, :], label='Analytical (Integration Time 0.001s)', color='purple', linestyle='-')

#plt.plot(time_points[:len(v1_values)], v1_values, label='Subsystem 1 Velocity (Microstep=0.001s, Macrostep=0.001s)', color='blue')
#plt.plot(time_points[:len(v1_valuess)], v1_valuess, label='Subsystem 1 Velocity (Microstep=0.0001s, Macrostep=0.001s)', color='green', linestyle='-')
plt.plot(time_points[:len(v1_values_5)], v1_values_5, label='FMPy (Microstep=0.00001s, Macrostep=0.1s)', color='orange', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()

# Figure for v_2
plt.figure()
plt.plot(time, XX_values[1, :], label='Analytical (Integration Time 0.001s)', color='purple', linestyle='-')

#plt.plot(time_points[:len(v2_values_fmu2)], v2_values_fmu2, label='Subsystem 2 Velocity (Microstep=0.001s, Macrostep=0.001s)', color='green')
#plt.plot(time_points[:len(v2_values_fmu4)], v2_values_fmu4, label='Subsystem 2 Velocity (Microstep=0.0001s, Macrostep=0.001s)', color='blue', linestyle='-')
plt.plot(time_points[:len(v2_values_fmu6)], v2_values_fmu6, label='FMPy (Microstep=0.00001s, Macrostep=0.1s)', color='red', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Figure for Mechanical Energy (EM)
plt.figure()
plt.plot(time, EM_anal, label='Analytical (Integration Time 0.001s)', color='purple', linestyle='-')

#plt.plot(time_points[:len(EM_values)], EM_values, label='Mechanical Energy (Microstep=0.001s, Macrostep=0.001s)', color='blue', linestyle='-')
#plt.plot(time_points[:len(EM_valuess)], EM_valuess, label='Mechanical Energy (Microstep=0.0001s, Macrostep=0.001s)', color='red', linestyle='--')
plt.plot(time_points[:len(EM_values_5)], EM_values_5, label='FMPy (Microstep=0.00001s, Macrostep=0.1s)', color='blue', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Mechanical Energy (J)')
plt.legend()
plt.ylim(9998.5, 10001.5)  # Adjust limits according to your needs
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

plt.show()
