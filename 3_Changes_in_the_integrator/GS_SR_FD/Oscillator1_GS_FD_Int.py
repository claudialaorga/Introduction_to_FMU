from pythonfmu import Fmi2Slave, Fmi2Causality, Real
from scipy.integrate import solve_ivp

class Oscillator1_GS_FD_Int(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Model parameters
        self.m_1 = 1.0  # kg
        self.m_2 = 1.0  # kg
        self.k_1 = 10.0  # N/m
        self.k_c = 100.0  # N/m
        self.k_2 = 1000.0  # N/m
        self.v_1 = 100.0  # Initial velocity of mass 1, m/s
        self.x_1 = 0.0  # Initial position of mass 1, m
        self.x_2 = 0.0  # Initial position of mass 2, m
        self.step_size = 0.001  # Time step size, must be initialized as a variable
        self.v_1_old = 100  # Previous value of velocity of mass 1
        self.x_1_old = 0  # Previous value of position of mass 1
        self.x_2_old = 0  # Previous value of position of mass 2
        self.FC = 0  # Coupling force, initialized to 0

        # Define output variables and input parameters
        self.register_variable(Real("x_1", causality=Fmi2Causality.output))
        self.register_variable(Real("v_1", causality=Fmi2Causality.output))
        self.register_variable(Real("FC", causality=Fmi2Causality.output))
        self.register_variable(Real("x_2", causality=Fmi2Causality.input))

    def do_step(self, current_time, step_size=None):
        step_size = self.step_size  # Trick to redefine the step_size
        t_span = (current_time, current_time + step_size)

        # Define the derivative function
        def derivative(t, y):
            x_1, v_1 = y
            dx1dt = v_1
            dv1dt = (-self.k_1 * x_1 + self.k_c * (self.x_2 - x_1)) / self.m_1
            return [dx1dt, dv1dt]

        # Solve using solve_ivp with the RK23 method
        sol = solve_ivp(derivative, t_span, [self.x_1_old, self.v_1_old], method='RK23')

        # Calculate the coupling force FC
        self.FC = self.k_c * (self.x_1 - self.x_2)

        # Update the state variables
        self.x_1 = sol.y[0][-1]  # Take the last value of x_1
        self.v_1 = sol.y[1][-1]  # Take the last value of v_1
        self.x_1_old = self.x_1  # Update the previous value of x_1
        self.v_1_old = self.v_1  # Update the previous value of v_1
        self.x_2_old = self.x_2  # Update the previous value of x_2

        return True
