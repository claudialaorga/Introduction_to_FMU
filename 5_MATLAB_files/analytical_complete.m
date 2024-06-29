% Initial values
m1 = 1;
m2 = 1;
k1 = 10;
k2 = 1000;
kc = 100;
c1 = 0.0;
c2 = 0.1;
cc = 0.01;
xx1 = 100;
xx2 = -100;
x1 = 0;
x2 = 0;
xxx1 = 0;
xxx2 = 0; 

% Constant matrices
Z0 = [x1, x2, xx1, xx2];
A = [0, 0, 1, 0; 0, 0, 0, 1; -(k1 + kc) / m1, kc / m1, -(c1 - cc / m1), cc / m1; kc / m2, -(k2 + kc) / m2, cc / m2, -(c2 + cc) / m2];

M = [m1, 0; 0, m2];
K = [k1 + kc, -kc; -kc, kc + k2];
expm(A * 3) * Z0(:);

time_step = 0.00001; % Time step of 0.001 seconds
time_end = 10; % End time of 10 seconds
time = 0:time_step:time_end; % Time vector

num_steps = length(time); % Number of steps

EM_values = zeros(1, num_steps); % Create matrix to store mechanical energy values
XX_values = zeros(2, num_steps); % Create matrix to store velocity values
X_values = zeros(2, num_steps); % Create matrix to store position values
FC_values = zeros(1, num_steps);

for i = 1:num_steps
    t = time(i); % Current time
    exp = expm(A * t); % Compute the matrix exponential of A times the time
    Z = exp * Z0(:);
    x_1 = Z(1);
    x_2 = Z(2);
    xx_1 = Z(3);
    xx_2 = Z(4);
    X = [x_1; x_2]; % Position
    XX = [xx_1; xx_2]; % Velocity
    f_c= kc*(x_1 -x_2);
    FC_values(:, i)=f_c;

    EM = (1 / 2) * transpose(XX) * M * XX + (1 / 2) * transpose(X) * K * X;
    EM_values(:, i) = EM;
    XX_values(:, i) = XX;
    X_values(:, i) = X;

end


% Plot EM_values against time
figure;
plot(time, EM_values, 'b-', 'LineWidth', 2); % Plot mechanical energy as a function of time
hold on
xlabel('Time [s]');
ylabel('Mechanical energy [J]');
title('Mechanical energy as a function of time');
grid on;

% Plot only the first column of XX_values against time
figure;
plot(time, XX_values(1, :), 'r-', 'LineWidth', 2); % Plot the first column of XX_values as a function of time

xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity of the first oscillator');

grid on;

figure;
plot(time, XX_values(2, :), 'b-', 'LineWidth', 2); % Plot the first column of XX_values as a function of time

xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity of the second oscillator');
grid on;

% Plot only the first column of X_values against time
figure;
plot(time, X_values(1, :), 'g-', 'LineWidth', 2); % Plot the first column of X_values as a function of time

xlabel('Time [s]');
ylabel('Position [m]');
title('Position of the first oscillator');

grid on;

figure;
plot(time, X_values(2, :), 'b-', 'LineWidth', 2); % Plot the first column of X_values as a function of time

xlabel('Time [s]');
ylabel('Position [m]');
title('Position of the second oscillator');
grid on;


% Plot FC_values against time
figure;
plot(time, FC_values, 'b-', 'LineWidth', 2); % Plot the transmitted force as a function of time
xlabel('Time [s]');
ylabel('Transmitted force [N], microstep=0.0001 s');
grid on;
