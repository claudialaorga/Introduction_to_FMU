from pythonfmu import Fmi2Slave, Fmi2Causality, Real

# SS2 mode first
class Oscillator2_GS_FD(Fmi2Slave):


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
        self.micro = 0.001  # Time step, forced to initialize as a variable
        self.v_2_old = -100  # Previous velocity of mass 2
        self.x_1_old = 0  # Previous position of mass 1
        self.x_2_old = 0  # Previous position of mass 2
        self.FC = 0  # Force calculation
        
        # Define output variables and input parameters
        self.register_variable(Real("FC", causality=Fmi2Causality.input))
        self.register_variable(Real("v_2", causality=Fmi2Causality.output))
        self.register_variable(Real("x_2", causality=Fmi2Causality.output))

    def do_step(self, current_time, step_size):
        t_f = current_time + step_size 
        while current_time < t_f:
            self.v_2, self.x_2 = self.oscillator_2(self.FC, self.micro, self.v_2_old, self.x_2_old)  # Evaluate later

            self.x_2_old = self.x_2
            self.v_2_old = self.v_2
            current_time += self.micro
        return True

    def oscillator_2(self, fc, dt, v_2_0, x_2_0):
        v_2 = v_2_0 + dt * (-self.k_2 * x_2_0 + fc) / self.m_2
        x_2 = x_2_0 + dt * v_2
        return v_2, x_2
