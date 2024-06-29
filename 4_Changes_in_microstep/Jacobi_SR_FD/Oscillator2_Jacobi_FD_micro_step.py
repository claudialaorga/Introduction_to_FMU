from pythonfmu import Fmi2Slave, Fmi2Causality, Integer, Real


class Oscillator2_Jacobi_FD_micro_step(Fmi2Slave):



    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Model parameters
        self.m_1 = 1.0  # kg
        self.m_2 = 1.0  # kg
        self.k_1 = 10.0  # N/m
        self.k_c = 100.0  # N/m
        self.k_2 = 1000.0  # N/m
        self.x_2 = 0.0  # Initial position of mass 2, m
        self.v_2 = -100 # Initial velocity of mass 2, m/s
        self.micro = 0.00001  # Time step, forces initialization as a variable

        self.x_2_old = 0
        self.v_2_old = -100
        self.FC_old = 0
       
        self.FC = 0
       
        # Define output variables and input parameters

        self.register_variable(Real("v_2", causality=Fmi2Causality.output))
        self.register_variable(Real("x_2", causality=Fmi2Causality.output))
        self.register_variable(Real("FC", causality=Fmi2Causality.input))


    def do_step(self, current_time, step_size):
        
        t_f = current_time + step_size 
        while current_time < t_f:
            self.v_2, self.x_2 = self.oscillator_2(self.FC_old, self.micro, self.v_2_old, self.x_2_old)
            self.FC_old = self.FC
            self.v_2_old = self.v_2
            self.x_2_old = self.x_2
            current_time+=self.micro
            return True


        
       

            


    def oscillator_2(self, fc, dt, v_2_0, x_2_0):
        v_2 = v_2_0 + dt * (-self.k_2* x_2_0+fc) / self.m_2
        x_2= x_2_0 + dt * v_2
        return v_2, x_2
