%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Analytical Solution%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial values
m1 = 1;
m2 = 1;
k1 = 10;
k2 = 1000;
kc = 100;
c1 = 0.0;
c2 = 0.0;
cc = 0.00;
xx1 = 100;
xx2 = -100;
x1 = 0;
x2 = 0;

% Constant matrices
Z0 = [x1, x2, xx1, xx2];
A = [0, 0, 1, 0; 0, 0, 0, 1; -(k1 + kc) / m1, kc / m1, -(c1 + cc) / m1, cc / m1; kc / m2, -(k2 + kc) / m2, cc / m2, -(c2 + cc) / m2];

M = [m1, 0; 0, m2];
K = [k1 + kc, -kc; -kc, kc + k2];

time_step = 0.0001; % Time step of 0.001 seconds
time_end = 10; % Final time of 10 seconds
time = 0:time_step:time_end; % Time vector

num_steps = length(time); % Number of steps

EM_values = zeros(1, num_steps); % Create matrix to store the values of mechanical energy
XX_values = zeros(2, num_steps); % Create matrix to store the values of velocities
X_values = zeros(2, num_steps); % Create matrix to store the values of positions
FC_values = zeros(1, num_steps);

for i = 1:num_steps
    t = time(i); % Current time
    exp = expm(A * t); % Exponential of matrix A times time
    Z = exp * Z0(:);
    x_1 = Z(1);
    x_2 = Z(2);
    xx_1 = Z(3);
    xx_2 = Z(4);
    X = [x_1; x_2]; % Position
    XX = [xx_1; xx_2]; % Velocity
    f_c = kc * (x_1 - x_2);
    FC_values(:, i) = f_c;

    EM = (1 / 2) * transpose(XX) * M * XX + (1 / 2) * transpose(X) * K * X;
    EM_values(:, i) = EM;
    XX_values(:, i) = XX;
    X_values(:, i) = X;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%Gauss-Seidel d-d%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial conditions and other parameters of interest
m_1 =1 %kg
m_2 =1 %kg
k_1 =10 %N/m
k_c =100 %N/m
k_2 =1000 %N/m
c_1=0 %Nm/s
c_2=0.1 %Nm/s
c_c=0.01 %Nm/s
v_1(1) = 100 %m/s
x_1(1) = 0; %m
v_2(1) = -100;
x_2(1) = 0;
t_0=0

t(1)=t_0;

M=[m_1 0 ; 0 m_2];
K=[k_1 + k_c -k_c ; -k_c k_c + k_2];

t_f=10
dt=0.001
n_p=(t_f-t_0)/dt %number of steps
E_mc(1)=1/2*[v_1(1) v_2(1)]*M*[v_1(1); v_2(1)] + 1/2*[x_1(1) x_2(1)]*K*[x_1(1); x_2(1)];


% Calculation of mechanical energy and inputs and outputs at t+1
for i=1:n_p 

[v_1(i+1),x_1(i+1)]=oscillator_1(x_2(i),dt,v_1(i),x_1(i),v_2(i));
[v_2(i+1),x_2(i+1)]=oscillator_2(x_1(i+1),dt,v_2(i),x_2(i),v_1(i));
t(i+1)=t(i)+dt;
E_mc(i+1)=1/2*[v_1(i+1) v_2(i+1)]*M*[v_1(i+1); v_2(i+1)] + 1/2*[x_1(i+1) x_2(i+1)]*K*[x_1(i+1); x_2(i+1)];
end

EM_gdd=E_mc;
X1_gdd=x_1;
X2_gdd=x_2;
V1_gdd=v_1;
V2_gdd=v_2;

function [v_1,x_1]=oscillator_1(x_2,dt,v_1_0,x_1_0, v_2)

m_1 =1; %kg
k_1 =10; %N/m
k_c =100; %N/m
c_c=0.01; 
c_1=0; 

xxx_1=( c_c * v_2 + k_c*x_2-(c_1+c_c)*v_1_0-(k_1+k_c)*x_1_0)/m_1;
v_1 = v_1_0 + dt*xxx_1;
x_1 = x_1_0 + dt*v_1;

end


function [v_2,x_2]=oscillator_2(x_1,dt,v_2_0,x_2_0, v_1)

m_2 =1; %kg
k_2 =1000; %N/m
k_c =100; %N/m
c_c=0.01; 
c_2=0.1;
xxx_2=(c_c*(v_1-v_2_0)+k_c*(x_1-x_2_0)-c_2*v_2_0-k_2*x_2_0)/m_2;
v_2 = v_2_0 + dt*xxx_2;
x_2 = x_2_0 + dt*v_2;



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%Gauss-Seidel f-d%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define constants
%Initial values
M_1 =1 %kg
M_2 =1 %kg
K_1 =10 %N/m
K_c =100 %N/m
K_2 =1000 %N/m
C_1=0 %Nm/s
C_2=0.0 %Nm/s
C_c=0.00 %Nm/s
V_1(1) = 100 %m/s
X_1(1) = 0; %m
V_2(1) = -100;
X_2(1) = 0;
T_0=0;
F_c(1)=0

T(1)=T_0;

M=[M_1 0 ; 0 M_2];
K=[k_1 + k_c -k_c ; -k_c k_c + K_2];

T_f=10
DT=0.001
n_p=(T_f-T_0)/DT %number of steps
E_mc1(1)=1/2*[V_1(1) V_2(1)]*M*[V_1(1); V_2(1)] + 1/2*[X_1(1) X_2(1)]*K*[X_1(1); X_2(1)];



% Calculation of mechanical energy and inputs and outputs at t+1
for i=1:n_p 

[V_1(i+1),X_1(i+1),F_c(i+1)]=oscillator_11(X_2(i),DT,V_1(i),X_1(i), V_2(i));
[V_2(i+1),X_2(i+1)]=oscillator_22(F_c(i+1),DT,V_2(i),X_2(i), V_1(i));
T(i+1)=T(i)+DT;
E_mc1(i+1)=1/2*[V_1(i+1) V_2(i+1)]*M*[V_1(i+1); V_2(i+1)] + 1/2*[X_1(i+1) X_2(i+1)]*K*[X_1(i+1); X_2(i+1)];
end
EM_gfd=E_mc1;
X1_gfd=X_1;
X2_gfd=X_2;
V1_gfd=V_1;
V2_gfd=V_2;
function [v_1,x_1,f_c]=oscillator_11(x_2,dt,v_1_0,x_1_0, v_2)

m_1 =1; %kg
k_1 =10; %N/m
k_c =100; %N/m
c_c = 0
c_1 =0
xxx_1=( c_c * v_2 + k_c*x_2-(c_1+c_c)*v_1_0-(k_1+k_c)*x_1_0)/m_1;
v_1 = v_1_0 + dt*xxx_1;
x_1 = x_1_0 + dt*v_1;
f_c=k_c*(x_1-x_2);
end


function [v_2,x_2]=oscillator_22(f_c,dt,v_2_0,x_2_0, v_1)

m_2 =1; %kg
k_2 =1000; %N/m
c_c=0.00; 
c_2=0.0;
xxx_2=(f_c-c_2*v_2_0-k_2*x_2_0)/m_2;
v_2 = v_2_0 + dt*xxx_2;
x_2 = x_2_0 + dt*v_2;

end


%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%Gauss-Seidel f-f%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

%Initial values
M_1 =1 %kg
M_2 =1 %kg
K_1 =10 %N/m
K_c =100 %N/m
K_2 =1000 %N/m
C_1=0 %Nm/s
C_2=0.0 %Nm/s
C_c=0.00 %Nm/s
V_1(1) = 100 %m/s
X_1(1) = 0; %m
V_2(1) = -100;
X_2(1) = 0;
T_0=0;
F_c(1)=0

T(1)=T_0;

M=[M_1 0 ; 0 M_2];
K=[k_1 + k_c -k_c ; -k_c k_c + K_2];

T_f=10
DT=0.001
n_p=(T_f-T_0)/DT %number of steps
E_mc2(1)=1/2*[V_1(1) V_2(1)]*M*[V_1(1); V_2(1)] + 1/2*[X_1(1) X_2(1)]*K*[X_1(1); X_2(1)];



% Calculation of mechanical energy and inputs and outputs at t+1
for i=1:n_p 

[V_1(i+1),X_1(i+1),F_c(i+1)]=oscillator_111(X_2(i),DT,V_1(i),X_1(i), V_2(i));
[V_2(i+1),X_2(i+1)]=oscillator_222(F_c(i+1),DT,V_2(i),X_2(i), V_1(i));
T(i+1)=T(i)+DT;
E_mc2(i+1)=1/2*[V_1(i+1) V_2(i+1)]*M*[V_1(i+1); V_2(i+1)] + 1/2*[X_1(i+1) X_2(i+1)]*K*[X_1(i+1); X_2(i+1)];
end
EM_gff=E_mc2;
X1_gff=X_1;
X2_gff=X_2;
V1_gff=V_1;
V2_gff=V_2;
function [v_1,x_1,f_c]=oscillator_111(x_2,dt,v_1_0,x_1_0, v_2)

m_1 =1; %kg
k_1 =10; %N/m
k_c =100; %N/m
c_c = 0
c_1 =0
xxx_1=( c_c * v_2 + k_c*x_2-(c_1+c_c)*v_1_0-(k_1+k_c)*x_1_0)/m_1;
v_1 = v_1_0 + dt*xxx_1;
x_1 = x_1_0 + dt*v_1;
f_c=k_c*(x_1-x_2);
end


function [v_2,x_2]=oscillator_222(f_c,dt,v_2_0,x_2_0, v_1)

m_2 =1; %kg
k_2 =1000; %N/m
c_c=0.00; 
c_2=0.0;
xxx_2=(f_c-c_2*v_2_0-k_2*x_2_0)/m_2;
v_2 = v_2_0 + dt*xxx_2;
x_2 = x_2_0 + dt*v_2;

end


%%%%%%%%%%%%% plotting
figure
plot(time, EM_values, 'k-', t, EM_gdd, 'r', T, EM_gfd, 'b', T, EM_gff, 'm');
title('Mechanical Energy');
legend('Analytical', 'd-d', 'f-d', 'f-f');

figure
plot(time, XX_values(1,:), 'k', t, V1_gdd, 'r', T, V1_gfd, 'b', T, V1_gff, 'm');
title('Velocity Oscillator 1');
legend('Analytical', 'd-d', 'f-d', 'f-f');

figure
plot(time, XX_values(2,:), 'k', t, V2_gdd, 'r', T, V2_gfd, 'b', T, V2_gff, 'm');
title('Velocity Oscillator 2');
legend('Analytical', 'd-d', 'f-d', 'f-f');

figure
plot(time, X_values(1,:), 'k', t, X1_gdd, 'r', T, X1_gfd, 'b', T, X1_gff, 'm');
title('Position Oscillator 1');
legend('Analytical', 'd-d', 'f-d', 'f-f');

figure
plot(time, X_values(2,:), 'k', t, X2_gdd, 'r', T, X2_gfd, 'b', T, X2_gff, 'm');
title('Position Oscillator 2');
legend('Analytical', 'd-d', 'f-d', 'f-f');

figure
plot(time, FC_values(1,:), 'k', T, F_c, 'm');
title('Contact Force');
legend('Analytical', 'f-f');

