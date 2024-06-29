from pythonfmu import Fmi2Slave, Fmi2Causality, Integer, Real

class Oscillator2_Jacobi_DD_micro_step(Fmi2Slave):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Model parameters
        self.m_1 = 1.0  # kg
        self.m_2 = 1.0  # kg
        self.k_1 = 10.0  # N/m
        self.k_c = 100.0  # N/m
        self.k_2 = 1000.0  # N/m
        self.x_2 = 0.0  # Initial position of mass 2, m
        self.v_2 = -100  # Initial velocity of mass 2, m/s
        self.micro = 0.00001  #Microstep
        self.x_1 = 0
        self.v_1 = 100
        self.v_1_old = 100
        self.x_1_old = 0
        self.x_2_old = 0
        self.v_2_old = -100

        # Define output variables and input parameters
        self.register_variable(Real("x_1", causality=Fmi2Causality.input))
        self.register_variable(Real("v_1", causality=Fmi2Causality.input))
        self.register_variable(Real("x_2", causality=Fmi2Causality.output))
        self.register_variable(Real("v_2", causality=Fmi2Causality.output))

    def do_step(self, current_time, step_size):
        t_f = current_time + step_size #Macrostep, given when instanciated
        while current_time < t_f:
            self.v_2, self.x_2 = self.oscillator_2(self.x_1_old, self.micro, self.v_2_old, self.x_2_old)
            self.v_1_old = self.v_1
            self.x_1_old = self.x_1
            self.x_2_old = self.x_2
            self.v_2_old = self.v_2

            current_time += self.micro
        return True

    def oscillator_2(self, x_1, dt, v_2_0, x_2_0):
        v_2 = v_2_0 + dt * (self.k_c * x_1 - (self.k_c + self.k_2) * x_2_0) / self.m_2
        x_2 = x_2_0 + dt * v_2
        return v_2, x_2
