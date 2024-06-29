%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%Monolithic%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define constants
% Initial values
m1=1;
m2=1;
k1=10;
k2=1000;
kc=100;
c1=0.0;
c2=0.1;
cc=0.01;
xx1=100;
xx2=-100;
x1=0;
x2=0;
M=[m1,0;0, m2]
K=[k1+kc, -kc; -kc, kc+k2]

% Temporal values
dt = 1/1000;
total_time = 10; % Total simulation time
num_steps = total_time / dt; % Total number of steps
aux=[cc, kc];

% Initialize matrices to store the variables of each subsystem at each time step
X1_values = zeros(1, num_steps); % Position 1
X2_values = zeros(1, num_steps); % Position 2
XX1_values = zeros(1, num_steps); % Velocity 1
XX2_values = zeros(1, num_steps); % Velocity 2
XXX1_values = zeros(1, num_steps); % Acceleration 1
XXX2_values = zeros(1, num_steps); % Acceleration 2
% The initial values are known
X1_values(1)=x1;
X2_values(1)=x2;
XX1_values(1)=xx1;
XX2_values(1)=xx2;

EM_values1 = zeros(1, 10); % Create matrix to store mechanical energy values
X = [x1; x2];
XX = [xx1; xx2];
EM = (1/2) * transpose(XX) * M * XX + (1/2) * transpose(X) * K * X;
EM_values1(1)=EM;
F_C(1)= kc*(X1_values(1) -X2_values(1));
times(1)=0;

% Simulation loop
for i = 1:num_steps

    % Calculate the acceleration values at time step i
    XXX1_values(i)=(X2_values(i)*kc-(k1+kc)*X1_values(i))/m1;
    XXX2_values(i)=(kc*X1_values(i)-(kc+k2)*X2_values(i))/m2;

    % Update the state variables of subsystem 1
    XX1_values(i+1)=XX1_values(i)+dt*XXX1_values(i);
    X1_values(i + 1) = X1_values(i)+dt*XX1_values(i+1); 
   
    % Update the state variables of subsystem 2
    XX2_values(i+1)=XX2_values(i)+dt*XXX2_values(i);
    X2_values(i + 1) = X2_values(i)+dt*XX2_values(i+1);
    
    % Calculate the mechanical energy
    X = [X1_values(i+1); X2_values(i+1)];
    XX = [XX1_values(i+1); XX2_values(i+1)];
    EM = (1/2) * transpose(XX) * M * XX + (1/2) * transpose(X) * K * X;
    EM_values1(:, i+1) = EM;

    % Calculate FC
    F_C(i+1)= kc*(X1_values(i) -X2_values(i));
    times(i+1)=times(i)+dt;
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%Complete Analytical%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial values
m1 = 1;
m2 = 1;
k1 = 10;
k2 = 1000;
kc = 100;
c1 = 0.0;
c2 = 0.0;
cc = 0.0;
xx1 = 100;
xx2 = -100;
x1 = 0;
x2 = 0;
xxx1 = 0;
xxx2 = 0; % Corrected, I suppose it should be xxx2

% Constant matrices
Z0 = [x1, x2, xx1, xx2];
A = [0, 0, 1, 0; 0, 0, 0, 1; -(k1 + kc) / m1, kc / m1, -(c1 - cc / m1), cc / m1; kc / m2, -(k2 + kc) / m2, cc / m2, -(c2 + cc) / m2];

M = [m1, 0; 0, m2];
K = [k1 + kc, -kc; -kc, kc + k2];
expm(A * 3) * Z0(:);

time_step = 0.001; % Time step of 0.001 seconds
time_end = 10; % End time of 10 seconds
time = 0:time_step:time_end; % Time vector

num_steps = length(time); % Number of steps

EM_values = zeros(1, num_steps); % Create matrix to store mechanical energy values
XX_values = zeros(2, num_steps); % Create matrix to store velocity values
X_values = zeros(2, num_steps); % Create matrix to store position values
FC_values = zeros(1, num_steps);

for j = 1:num_steps
    t = time(j); % Current time
    exp = expm(A * t); % Compute the matrix exponential of A times the time
    Z = exp * Z0(:);
    x_1 = Z(1);
    x_2 = Z(2);
    xx_1 = Z(3);
    xx_2 = Z(4);
    X = [x_1; x_2]; % Position
    XX = [xx_1; xx_2]; % Velocity
    f_c= kc*(x_1 -x_2);
    FC_values(:, j)=f_c;

    EM = (1 / 2) * transpose(XX) * M * XX + (1 / 2) * transpose(X) * K * X;
    EM_values(:, j) = EM;
    XX_values(:, j) = XX;
    X_values(:, j) = X;

end

% Plot mechanical energy over time
figure;
plot(times, EM_values1, 'm-', 'LineWidth', 2); % Monolithic solution in purple
hold on;
plot(time, EM_values, 'b--', 'LineWidth', 2); % Analytical solution in blue
xlabel('Time[s]');
ylabel('Mechanical Energy[J]');
legend('Monolithic Solution', 'Analytical Solution');
grid on;

% Plot the position of x_1
figure;
plot(times, X1_values, 'm-', 'LineWidth', 2); % Monolithic solution in purple
hold on;
plot(time, X_values(1, :), 'g--', 'LineWidth', 2); % Analytical solution in green
xlabel('Time[s]');
ylabel('Position[m]');
lgd = legend('Monolithic Solution', 'Analytical Solution');
lgd.FontSize = 14; % Adjust the legend font size to 14 (you can change the number to adjust the size)
grid on;

% Plot the position of x_2
figure;
plot(times, X2_values, 'b-', 'LineWidth', 2); % Monolithic solution in purple
hold on;
plot(time, X_values(2, :), 'r--', 'LineWidth', 2); % Analytical solution in green
xlabel('Time[s]');
ylabel('Position[m]');
lgd = legend('Monolithic Solution', 'Analytical Solution');
lgd.FontSize = 14; % Adjust the legend font size to 14 (you can change the number to adjust the size)
grid on;

% Plot the position of v_1
figure;
plot(times, XX1_values, 'b-', 'LineWidth', 2); % Monolithic solution in purple
hold on;
plot(time, XX_values(1, :), 'g--', 'LineWidth', 2); % Analytical solution in green
xlabel('Time[s]');
ylabel('Velocity [m/s]');
lgd = legend('Monolithic Solution', 'Analytical Solution');
lgd.FontSize = 14; % Adjust the legend font size to 14 (you can change the number to adjust the size)
grid on;

% Plot the position of v_2
figure;
plot(times, XX2_values, 'r-', 'LineWidth', 2); % Monolithic solution in purple
hold on;
plot(time, XX_values(2, :), 'g--', 'LineWidth', 2); % Analytical solution in green
xlabel('Time[s]');
ylabel('Velocity [m/s]');
lgd = legend('Monolithic Solution', 'Analytical Solution');
lgd.FontSize = 14; % Adjust the legend font size to 14 (you can change the number to adjust the size)
grid on;

% Plot the transmitted force
figure;
plot(times, FC_values, 'r-', 'LineWidth', 2); % Monolithic solution in purple
hold on;
plot(time, F_C, 'g--', 'LineWidth', 2); % Analytical solution in green
xlabel('Time[s]');
ylabel('Transmitted force [N]');
lgd = legend('Monolithic Solution', 'Analytical Solution');
lgd.FontSize = 14; % Adjust the legend font size to 14 (you can change the number to adjust the size)
grid on;
