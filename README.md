Hello! 👋🏻, I am Claudia Laorga de Villa, a student of the Chemical Engineering degree at the Polytechnic University of Madrid, and for my Final Degree Project (TFG) I have researched the use of FMI in open-source software, specifically in Python. Throughout this repository, you will find the files with which I export FMUs from pythonfmu and import FMUs in FMPy. With this repository, I aim to help anyone with basic programming knowledge to become familiar with the FMI standard and learn the basic commands to generate and import FMU files in open-source code.

In this work, I will evaluate FMI in the field of co-simulation, and I do so through an example commonly used among those starting in co-simulation: the two-mass linear oscillator. To study it, I have employed various types of compartmentalization (Jacobi and Gauss-Seidel), integration steps, and integrators. I have also compared the results with co-simulations in MATLAB and everything is explained in the .pdf file, but it is in Spanish 😬.

Upon opening the folders, you will find files that start with FMPy___.py and others that are named Oscillators.py. The files starting with FMPy___.py are used to import the co-simulation, while the Oscillators.py files are used to export each subsystem using pythonfmu.

**How to export from .py to FMU in pythonfmu?**

1. Open the command prompt.

2. Enter the following code: python -m pythonfmu build -f "path_to_your_repository.py" -d "New_directory"
   
3. In your new folder named "New_directory," your created .FMU files will appear.

Remember to replace 'path_to_your_repository.py' with your actual .py file and 'New_directory' with the name you want your folder to have.


**How to import a .fmu file in FMPy?**

Just use the FMPy___.py file, but remember to change the path in the code. 

I hope this work proves very useful to you and streamlines your use of FMI. Remember, it is a Final Degree Project, and despite being researched with the utmost rigor, there may be errors in the code at some point. Please do not hesitate to let me know if you detect any, so I can update the information as soon as possible. 

I also recommend referring to the .pdf file included in this repository if you have any questions. 

Have a great day!😊😊😊
