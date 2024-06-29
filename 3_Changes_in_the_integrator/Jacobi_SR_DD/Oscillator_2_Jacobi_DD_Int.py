from pythonfmu import Fmi2Slave, Fmi2Causality, Real
from scipy.integrate import solve_ivp
import numpy as np

class Oscillator_2_Jacobi_DD_Int(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Model parameters
        self.m_1 = 1.0  # kg
        self.m_2 = 1.0  # kg
        self.k_1 = 10.0  # N/m
        self.k_c = 100.0  # N/m
        self.k_2 = 1000.0  # N/m
        self.x_2 = 0.0  # Initial position of mass 2, m
        self.v_2 = -100.0  # Initial velocity of mass 2, m/s
        self.step_size = 0.001  # Time step size, must be initialized as a variable
        self.x_1 = 0.0  # Initial position of mass 1, m
        self.v_1 = 100.0  # Initial velocity of mass 1, m/s

        # Define output variables and input parameters
        self.register_variable(Real("x_1", causality=Fmi2Causality.input))
        self.register_variable(Real("v_1", causality=Fmi2Causality.input))
        self.register_variable(Real("x_2", causality=Fmi2Causality.output))
        self.register_variable(Real("v_2", causality=Fmi2Causality.output))

    def do_step(self, current_time, step_size):
        step_size = self.step_size  # Trick to redefine the step_size
        t_span = (current_time, current_time + step_size)

        # Define the derivative function
        def f(t, y):
            x_2, v_2 = y
            dx2dt = v_2
            dv2dt = (self.k_c * self.x_1 - (self.k_c + self.k_2) * x_2) / self.m_2
            return [dx2dt, dv2dt]

        # Solve using solve_ivp with the LSODA method
        sol = solve_ivp(f, t_span, [self.x_2, self.v_2], method='LSODA')

        self.x_2 = sol.y[0][-1]  # Take the last value of x_2
        self.v_2 = sol.y[1][-1]  # Take the last value of v_2

        return True
