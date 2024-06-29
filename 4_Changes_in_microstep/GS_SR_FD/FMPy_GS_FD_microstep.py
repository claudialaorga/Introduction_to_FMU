
from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.ticker import ScalarFormatter

plt.rcParams['font.family'] = 'serif'

#Define the path to the fmu
nombre_fmu1 = 'path_to_your_Oscillator1_GS_FD.fmu'
nombre_fmu2 = 'path_to_your_Oscillator2_GS_FD.fmu'
nombre_fmu3 = 'path_to_your_Oscillator1_GS_FD_microstep.fmu'
nombre_fmu4 = 'path_to_your_Oscillator2_GS_FD_microstep.fmu'
nombre_fmu5 = 'path_to_your_/Oscillator1_GS_FD_micro_step.fmu'
nombre_fmu6 = 'path_to_your_/Oscillator2_GS_FD_micro_step.fmu'

#Dump to know the information about the fmu
dump(nombre_fmu1)
dump(nombre_fmu2)
dump(nombre_fmu3)
dump(nombre_fmu4)
dump(nombre_fmu5)
dump(nombre_fmu6)

#Set the simulation times
start_time = 0.0
stop_time = 10.0
step_size = 0.001

# read the model description
model_description = read_model_description(nombre_fmu1)
model_description2 = read_model_description(nombre_fmu2)
model_description3 = read_model_description(nombre_fmu3)
model_description4 = read_model_description(nombre_fmu4)
model_description5 = read_model_description(nombre_fmu5)
model_description6 = read_model_description(nombre_fmu6)

# collect the value references
vrs = {variable.name: variable.valueReference for variable in model_description.modelVariables}
vrs2 = {variable.name: variable.valueReference for variable in model_description2.modelVariables}
vrs3 = {variable.name: variable.valueReference for variable in model_description3.modelVariables}
vrs4 = {variable.name: variable.valueReference for variable in model_description4.modelVariables}
vrs5 = {variable.name: variable.valueReference for variable in model_description5.modelVariables}
vrs6 = {variable.name: variable.valueReference for variable in model_description6.modelVariables}

print(vrs, 'Estas son las variables del primer archivo .fmu')
print(vrs2, 'Estas son las variables del segundo archivo .fmu')
print(vrs3, 'Estas son las variables del tercer archivo .fmu')
print(vrs4, 'Estas son las variables del cuarto archivo .fmu')
print(vrs5, 'Estas son las variables del quinto archivo .fmu')
print(vrs6, 'Estas son las variables del sexto archivo .fmu')


def get_value_reference(model_description, variable_name):
    for variable in model_description.modelVariables:
        if variable.name == variable_name:
            return variable.valueReference
    raise Exception(f"Variable {variable_name} not found in model")

vr_fmu1_outputs = {
    'x_1': get_value_reference(model_description, 'x_1'),
    'v_1': get_value_reference(model_description, 'v_1'),
    'FC': get_value_reference(model_description, 'FC')
}
vr_fmu1_inputs = {
    'x_2': get_value_reference(model_description, 'x_2'),
}
vr_fmu3_outputs = {
    'x_1': get_value_reference(model_description3, 'x_1'),
    'v_1': get_value_reference(model_description3, 'v_1'),
    'FC': get_value_reference(model_description3, 'FC')
}
vr_fmu3_inputs = {
    'x_2': get_value_reference(model_description3, 'x_2'),
}
vr_fmu5_outputs = {
    'x_1': get_value_reference(model_description5, 'x_1'),
    'v_1': get_value_reference(model_description5, 'v_1'),
    'FC': get_value_reference(model_description5, 'FC')
}
vr_fmu5_inputs = {
    'x_2': get_value_reference(model_description5, 'x_2'),
}

vr_fmu2_outputs = {
    'x_2': get_value_reference(model_description2, 'x_2'),
    'v_2': get_value_reference(model_description2, 'v_2')
}
vr_fmu2_inputs = {
    'FC': get_value_reference(model_description2, 'FC'),
}
vr_fmu4_outputs = {
    'x_2': get_value_reference(model_description4, 'x_2'),
    'v_2': get_value_reference(model_description4, 'v_2')
}
vr_fmu4_inputs = {
    'FC': get_value_reference(model_description4, 'FC'),
}
vr_fmu6_outputs = {
    'x_2': get_value_reference(model_description6, 'x_2'),
    'v_2': get_value_reference(model_description6, 'v_2')
}
vr_fmu6_inputs = {
    'FC': get_value_reference(model_description6, 'FC'),
}

unzipdir = extract(nombre_fmu1)
unzipdir2 = extract(nombre_fmu2)
unzipdir3 = extract(nombre_fmu3)
unzipdir4 = extract(nombre_fmu4)
unzipdir5 = extract(nombre_fmu5)
unzipdir6 = extract(nombre_fmu6)

fmu1 = FMU2Slave(guid=model_description.guid,
                 unzipDirectory=unzipdir,
                 modelIdentifier=model_description.coSimulation.modelIdentifier,
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

fmu1.instantiate()
fmu2.instantiate()
fmu3.instantiate()
fmu4.instantiate()
fmu5.instantiate()
fmu6.instantiate()

fmu1.setupExperiment(startTime=start_time)
fmu2.setupExperiment(startTime=start_time)
fmu3.setupExperiment(startTime=start_time)
fmu4.setupExperiment(startTime=start_time)
fmu5.setupExperiment(startTime=start_time)
fmu6.setupExperiment(startTime=start_time)

fmu1.enterInitializationMode()
fmu2.enterInitializationMode()
fmu3.enterInitializationMode()
fmu4.enterInitializationMode()
fmu5.enterInitializationMode()
fmu6.enterInitializationMode()

fmu1.exitInitializationMode()
fmu2.exitInitializationMode()
fmu3.exitInitializationMode()
fmu4.exitInitializationMode()
fmu5.exitInitializationMode()
fmu6.exitInitializationMode()

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

x1_valuess = []
v1_valuess = []
FC_valuess = []
x2_values_fmu11 = []
x2_values_fmu22 = []
v2_values_fmu11 = []
v2_values_fmu22 = []
EM_valuess = []

x1_valuesss = []
v1_valuesss = []
FC_valuesss = []
x2_values_fmu111 = []
x2_values_fmu222 = []
v2_values_fmu111 = []
v2_values_fmu222 = []
EM_valuesss = []

m_1 = 1.0  # kg
m_2 = 1.0  # kg
k_1 = 10.0  # N/m
k_c = 100.0  # N/m
k_2 = 1000.0  # N/m

M = np.array([[m_1, 0], [0, m_2]])
K = np.array([[k_1 + k_c, -k_c], [-k_c, k_c + k_2]])


#Simulation loop
while time < stop_time:

    fmu1.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu3.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    fmu5.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    x_1 = fmu1.getReal([vr_fmu1_outputs['x_1']])[0]
    v_1 = fmu1.getReal([vr_fmu1_outputs['v_1']])[0]
    FC = fmu1.getReal([vr_fmu1_outputs['FC']])[0]

    x_11 = fmu3.getReal([vr_fmu3_outputs['x_1']])[0]
    v_11 = fmu3.getReal([vr_fmu3_outputs['v_1']])[0]
    FC1 = fmu3.getReal([vr_fmu3_outputs['FC']])[0]

    x_111 = fmu5.getReal([vr_fmu5_outputs['x_1']])[0]
    v_111 = fmu5.getReal([vr_fmu5_outputs['v_1']])[0]
    FC2 = fmu5.getReal([vr_fmu5_outputs['FC']])[0]

    fmu2.setReal([vr_fmu2_inputs['FC']], [FC])
    fmu4.setReal([vr_fmu4_inputs['FC']], [FC1]) 
    fmu6.setReal([vr_fmu6_inputs['FC']], [FC2]) 


   
    fmu2.doStep(currentCommunicationPoint=time, communicationStepSize=step_size) 
    fmu4.doStep(currentCommunicationPoint=time, communicationStepSize=step_size) 
    fmu6.doStep(currentCommunicationPoint=time, communicationStepSize=step_size) 


    

    x2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['x_2']])[0] 
    v2_value_fmu2 = fmu2.getReal([vr_fmu2_outputs['v_2']])[0]

    x2_value_fmu22 = fmu4.getReal([vr_fmu4_outputs['x_2']])[0] 
    v2_value_fmu22 = fmu4.getReal([vr_fmu4_outputs['v_2']])[0]

    x2_value_fmu222 = fmu6.getReal([vr_fmu6_outputs['x_2']])[0] 
    v2_value_fmu222 = fmu6.getReal([vr_fmu6_outputs['v_2']])[0]


    fmu1.setReal([vr_fmu1_inputs['x_2']], [x2_value_fmu2]) 
    fmu3.setReal([vr_fmu3_inputs['x_2']], [x2_value_fmu22]) 
    fmu5.setReal([vr_fmu5_inputs['x_2']], [x2_value_fmu222]) 

    x2_value_fmu1 = fmu1.getReal([vr_fmu1_inputs['x_2']])[0]
    x2_value_fmu11 = fmu3.getReal([vr_fmu3_inputs['x_2']])[0]
    x2_value_fmu111 = fmu5.getReal([vr_fmu5_inputs['x_2']])[0]


    EM_value = (0.5 * np.dot([v_1, v2_value_fmu2], np.dot(M, [v_1, v2_value_fmu2])) +
                        0.5 * np.dot([x_1, x2_value_fmu1], np.dot(K, [x_1, x2_value_fmu1]))) 

    EM_valuee = (0.5 * np.dot([v_11, v2_value_fmu22], np.dot(M, [v_11, v2_value_fmu22])) +
                        0.5 * np.dot([x_11, x2_value_fmu11], np.dot(K, [x_11, x2_value_fmu11])))  

    EM_valueee = (0.5 * np.dot([v_111, v2_value_fmu222], np.dot(M, [v_111, v2_value_fmu222])) +
                        0.5 * np.dot([x_111, x2_value_fmu111], np.dot(K, [x_111, x2_value_fmu111])))  





    
    x1_values.append(x_1)
    v1_values.append(v_1)
    x2_values_fmu1.append(x2_value_fmu1)
    x2_values_fmu2.append(x2_value_fmu2)
    v2_values_fmu2.append(v2_value_fmu2)
    FC_values.append(FC)
    EM_values.append(EM_value)
    time_points.append(time)

    x1_valuess.append(x_11)
    v1_valuess.append(v_11)
    x2_values_fmu11.append(x2_value_fmu11)
    x2_values_fmu22.append(x2_value_fmu22)
    v2_values_fmu22.append(v2_value_fmu22)
    FC_valuess.append(FC1)
    EM_valuess.append(EM_valuee)

    x1_valuesss.append(x_111)
    v1_valuesss.append(v_111)
    x2_values_fmu111.append(x2_value_fmu111)
    x2_values_fmu222.append(x2_value_fmu222)
    v2_values_fmu222.append(v2_value_fmu222)
    FC_valuesss.append(FC2)
    EM_valuesss.append(EM_valueee)

    time += step_size
   

fmu1.terminate()
fmu2.terminate()
fmu3.terminate()
fmu4.terminate()
fmu5.terminate()
fmu6.terminate()

fmu1.freeInstance()
fmu2.freeInstance()
fmu3.freeInstance()
fmu4.freeInstance()
fmu5.freeInstance()
fmu6.freeInstance()

shutil.rmtree(unzipdir)
shutil.rmtree(unzipdir2)
shutil.rmtree(unzipdir3)
shutil.rmtree(unzipdir4)
shutil.rmtree(unzipdir5)
shutil.rmtree(unzipdir6)

#########################Analitycal solution#################
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


Z0 = np.array([x1, x2, xx1, xx2])
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-(k1 + kc) / m1, kc / m1, -(c1 - cc / m1), cc / m1],
    [kc / m2, -(k2 + kc) / m2, cc / m2, -(c2 + cc) / m2]
])
M = np.array([[m1, 0], [0, m2]])
K = np.array([[k1 + kc, -kc], [-kc, kc + k2]])


time_step = 0.001
time_end = 10
time = np.arange(0, time_end + time_step, time_step)
num_steps = len(time)

EM_anal = np.zeros(num_steps)
XX_values = np.zeros((2, num_steps))
X_values = np.zeros((2, num_steps))
FC_anal = np.zeros(num_steps)

# Principal loop
for i in range(num_steps):
    t = time[i] 
    exp = expm(A * t)  
    Z = exp.dot(Z0)

    x_1 = Z[0]
    x_2 = Z[1]
    xx_1 = Z[2]
    xx_2 = Z[3]

    X = np.array([x_1, x_2])  
    XX = np.array([xx_1, xx_2])  

    f_c = kc * (x_1 - x_2)
    FC_anal[i] = f_c

    EM = 0.5 * XX.T.dot(M).dot(XX) + 0.5 * X.T.dot(K).dot(X)
    EM_anal[i] = EM

    XX_values[:, i] = XX
    X_values[:, i] = X

# Figure for x_1
plt.figure()
plt.plot(time, X_values[0, :], label='Analítica ', color='purple',  linestyle='-')

plt.plot(time_points[:len(x1_values)], x1_values, label='Position subs 1 (Microstep=0.001s, Macrostep=0.001s)', color='blue', linestyle='--')
plt.plot(time_points[:len(x1_valuess)], x1_valuess, label='Position subs 1 (Microstep=0.0001s, Macrostep=0.001s)', color='green',  linestyle='--')
plt.plot(time_points[:len(x1_valuesss)], x1_valuesss, label='Posición subsistema 1  (Microtep=0.00001s, Macrostep=0.001s)', color='red',  linestyle='--')


plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.show()


#Figure for x_2
plt.figure()
plt.plot(time, X_values[1, :], label='Analytical value', color='purple',  linestyle='-')
plt.plot(time_points[:len(x2_values_fmu1)], x2_values_fmu1, label='Position subs 2 (Microstep=0.001s, Macrostep=0.001s)', color='red', linestyle='--')
plt.plot(time_points[:len(x2_values_fmu11)], x2_values_fmu11, label='Position subs 2 (Microstep=0.0001s, Macrostep=0.001s)', color='blue',  linestyle='--')
plt.plot(time_points[:len(x2_values_fmu222)], x2_values_fmu222, label='Position subs 2 (Microstep=0.00001s, Macrostep=0.001s)', color='green',  linestyle='--')


plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.show()

#Figure for v_1
plt.figure()
plt.plot(time, XX_values[0, :], label='Analytical value', color='purple',  linestyle='-')

plt.plot(time_points[:len(v1_values)], v1_values, label='Speed subs 1 (Microstep=0.001s, Macrostep=0.001s)', color='blue', linestyle='--')
plt.plot(time_points[:len(v1_valuess)], v1_valuess, label='Speed subs 1 (Microstep=0.0001s, Macrostep=0.001s)', color='green',  linestyle='-')
plt.plot(time_points[:len(v1_valuesss)], v1_valuesss, label='Speed subs 1 (Microstep=0.00001s, Macrostep=0.001s)', color='orange',  linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()


# Figura para v_2
plt.figure()
plt.plot(time, XX_values[1, :], label='Analytical value', color='purple',  linestyle='-')
plt.plot(time_points[:len(v2_values_fmu2)], v2_values_fmu2, label='Speed subs 2 (Microstep=0.001s, Macrostep=0.001s)', color='green', linestyle='--')
plt.plot(time_points[:len(v2_values_fmu22)], v2_values_fmu22, label='Speed subs 2 (Microstep=0.0001s, Macrostep=0.001s)', color='blue',  linestyle='-')
plt.plot(time_points[:len(v2_values_fmu222)], v2_values_fmu222, label='Speed subs 2 (Microstep=0.00001s, Macrostep=0.001s)', color='red',  linestyle='--')


plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()



# Figura para EM
plt.figure()
plt.plot(time, EM_anal, label='Analytical value', color='purple',  linestyle='-')
plt.plot(time_points[:len(EM_values)], EM_values, label='Mechanical energy (Microstep=0.001s, Macrostep=0.001s)', color='blue', linestyle='-')
plt.plot(time_points[:len(EM_valuess)], EM_valuess, label='Mechanical energy (Microstep=0.0001s, Macrostep=0.001s)', color='red', linestyle='-')
plt.plot(time_points[:len(EM_valuesss)], EM_valuesss, label='Mechanical energy (Microstep=0.00001s, Macrostep=0.001s)',color='orange',linestyle='--')


plt.xlabel('Time (s)')
plt.ylabel('Mechanical energy (J)')
plt.legend()
#plt.ylim(9997.5, 10002.5)
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))


plt.show()




# Figura FC
plt.figure()
plt.plot(time, FC_anal, label='Analítica', linestyle='-')
plt.plot(time_points[:len(FC_values)], FC_values, label='(Microstep=0.001s, Macrostep=0.001s)', linestyle='--')
plt.plot(time_points[:len(FC_valuess)], FC_valuess, label='(Microstep=0.0001s, Macrostep=0.001s)', linestyle='--')
plt.plot(time_points[:len(FC_valuesss)], FC_valuesss, label='(Microstep=0.00001s, Macrostep=0.001s)', linestyle='--')


plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.show()




