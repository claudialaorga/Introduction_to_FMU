from pythonfmu import Fmi2Slave, Fmi2Causality, Real
from scipy.integrate import solve_ivp

class Oscillator2_GS_FD_Int(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Model parameters
        self.m_1 = 1.0  # kg
        self.m_2 = 1.0  # kg
        self.k_1 = 10.0  # N/m
        self.k_c = 100.0  # N/m
        self.k_2 = 1000.0  # N/m
        self.x_1 = 0.0  # Initial position of mass 1, m
        self.v_2 = -100.0  # Initial velocity of mass 2, m/s
        self.x_2 = 0.0  # Initial position of mass 2, m
        self.step_size = 0.001  # Time step size, must be initialized as a variable
        self.v_2_old = -100  # Previous value of velocity of mass 2
        self.x_1_old = 0  # Previous value of position of mass 1
        self.x_2_old = 0  # Previous value of position of mass 2
        self.FC = 0  # Coupling force, initialized to 0
        self.FC_old = 0  # Previous value of coupling force

        # Define output variables and input parameters
        self.register_variable(Real("FC", causality=Fmi2Causality.input))
        self.register_variable(Real("v_2", causality=Fmi2Causality.output))
        self.register_variable(Real("x_2", causality=Fmi2Causality.output))

    def do_step(self, current_time, step_size=None):
        step_size = self.step_size  # Trick to redefine the step_size
        t_span = (current_time, current_time + step_size)

        # Define the derivative function
        def derivative(t, y):
            x_2, v_2 = y
            dx2dt = v_2
            dv2dt = (-self.k_2 * x_2 + self.FC) / self.m_2
            return [dx2dt, dv2dt]

        # Solve using solve_ivp with the RK23 method
        sol = solve_ivp(derivative, t_span, [self.x_2_old, self.v_2_old], method='RK23')

        # Update the state variables
        self.x_2 = sol.y[0][-1]  # Take the last value of x_2
        self.v_2 = sol.y[1][-1]  # Take the last value of v_2
        self.x_2_old = self.x_2  # Update the previous value of x_2
        self.v_2_old = self.v_2  # Update the previous value of v_2

        return True

    def oscillator_2(self, fc, dt, v_2_0, x_2_0):
        v_2 = v_2_0 + dt * (-self.k_2 * x_2_0 + fc) / self.m_2
        x_2 = x_2_0 + dt * v_2
        return v_2, x_2
