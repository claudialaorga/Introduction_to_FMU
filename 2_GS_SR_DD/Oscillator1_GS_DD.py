from pythonfmu import Fmi2Slave, Fmi2Causality, Real

class Oscillator1_GS_DD(Fmi2Slave):

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
        self.micro = 0.001  # Time step, initialized as variable
        self.v_1_old = 100
        self.x_1_old = 0
        self.x_2_old = 0
        
        # Define output variables and input parameters

        self.register_variable(Real("x_1", causality=Fmi2Causality.output))
        self.register_variable(Real("v_1", causality=Fmi2Causality.output))
        self.register_variable(Real("x_2", causality=Fmi2Causality.input))

    def do_step(self, current_time, step_size):
        
        t_f = current_time + step_size 
        while current_time < t_f:
                    
            self.v_1, self.x_1 = self.oscillator_1(self.x_2, self.micro, self.v_1_old, self.x_1_old)  # Evaluate first
            self.x_1_old = self.x_1
            self.v_1_old = self.v_1
            current_time += self.micro
        return True
    
    def oscillator_1(self, x_2, dt, v_1_0, x_1_0):
        v_1 = v_1_0 + dt * (-self.k_1 * x_1_0 + self.k_c*(x_2 - x_1_0)) / self.m_1
        x_1 = x_1_0 + dt * v_1
        return v_1, x_1
