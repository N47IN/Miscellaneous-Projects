% Setup and Initialization
% R Navin Sriram , ED21B044, Modern Control Theory, fall '23
% to be used as an mlx file 
clc; clear; close all;

% initializing the pf structure which will carry all our parameters and
% solutions

global pf;
global g;

% parameters
pf = struct();
pf.N = 500;
pf.n = 4;
pf.A1 = 28; %(cm^2)
pf.A2 = 32;
pf.A3 = 28;
pf.q=[];
pf.A4 = 32;
pf.A = [pf.A1, pf.A2, pf.A3, pf.A4];
pf.a1 = 0.071; pf.a3 = 0.071; %(cm^2)
pf.a2 = 0.057; pf.a4 = 0.057;
pf.a = [pf.a1, pf.a2, pf.a3, pf.a4];
pf.kc = 0.5; % (V/cm) 
g = 981; %(cm/s^2)
pf.gamma1 = 0.7; pf.gamma2 = 0.6;  
pf.k1 = 3.33; pf.k2 = 3.35; %[cm^3/Vs]
pf.kc = 1; % [V/cm]
pf.kc = 1; % [V/cm]
pf.v1 = 3; pf.v2 = 3; % (V)
pf.U = [pf.v1; pf.v2];
pf.h0 = [12.4; 12.7; 1.8; 1.4];
pf.P_pr = 10*eye(4); pf.Q = 10*eye(4); pf.R = 20*eye(2);


pf.innov=[];
T = [];
pf.x_pr = [];



% Finding the value for the term T for all the Elements
for j = 1:4
    T(j) =  (pf.A(j)/pf.a(j))*sqrt(2*pf.h0(j)/g) ;
end

% Initializing the Control Input Matrix, State Matrix and Output Matrix
pf.Ac = [ -1/T(1), 0, pf.A3/(pf.A1*T(3)), 0 ; 0, -1/T(2), 0, pf.A4/(pf.A2*T(4)); 0, 0, -1/T(3), 0; 0, 0, 0, -1/T(4)];
pf.Bc = [pf.gamma1*pf.k1/pf.A1 0 ; 0 pf.gamma2*pf.k2/ pf.A2; 0 (1 - pf.gamma2)*pf.k2/pf.A3; (1-pf.gamma1)*pf.k1/pf.A4 0];
pf.Dc = 0;
pf.Hc = [pf.kc 0 0 0; 0 pf.kc 0 0];


% Initializing the array containing posterior values of the States, with
% first val being h0
pf.x_po(:,1) = pf.h0;

% finding the solution to the ODEs using ODE45 for a give time and 500
% samples
ts = 0.1;
t_span = linspace(0, pf.N*ts, pf.N);
[t, y] = ode45(@myODEs, t_span, pf.h0);

% initialising the array containing all the measurements

pf.Z = [y(:,1:2)] * pf.kc;

[rows,cols] = size(t);

% Discretizing the matrics based on step of 0.1
state_space = ss(pf.Ac, pf.Bc, pf.Hc, pf.Dc);
state_space_discrete = c2d(state_space, ts);
pf.Ad = state_space_discrete.A;
pf.Bd = state_space_discrete.B;
pf.Hd = state_space_discrete.C;
pf.Dd = state_space_discrete.D;

% Initial Sample generation and roughening based on initial covariance
pf.L = chol(pf.P_pr);
pf.x_po = (pf.x_po * ones(1,pf.N))' + randn(pf.N,pf.n) * pf.L;
pf.x_po = pf.x_po';

% Required results
pf.state_pr = [];
pf.state_po = [];
pf.resid = [];
pf.Z_po = [];
pf.Z_pr = [];


%Prediction step and weight generation (Likelihood step)

for count = 1:500
    
 pf.v = [];
 pf.x_pr = [];
 pf.Zest = [];
 pf.q = [];

 pf.state_po(:,end+1) = mean(pf.x_po,2);
 pf.Z_po(:, end +1) = pf.Hd * pf.state_po(:,count);

 % additional roughening based on process noise
 pf.w = chol(pf.Q) * randn(pf.n,pf.N);
 pf.x_po = pf.x_po + pf.w;
 pf.w1 = pf.w(1,:);
 pf.w2 = pf.w(2,:);
 pf.w3 = pf.w(3,:);
 pf.w4 = pf.w(4,:);

% prediction step / propogration of model dynamics
for i = 1:pf.N

        pf.Z1 = pf.Z(count,1)* ones(1,pf.N);
        pf.Z2 = pf.Z(count,2)* ones(1,pf.N);
        pf.Z_ext = [pf.Z1; pf.Z2];
        
        pf.x_pr(1,i) = ts*(-pf.a1/pf.A1 * sqrt(2*g*pf.x_po(1,i)) + pf.a3/pf.A1 * sqrt(2*g*pf.x_po(3,i)) + pf.gamma1*pf.k1*pf.v1/pf.A1) + pf.w1(i);
        pf.x_pr(2,i) = ts*(-pf.a2/pf.A2 * sqrt(2*g*pf.x_po(2,i)) + pf.a4/pf.A2 * sqrt(2*g*pf.x_po(4,i)) + pf.gamma2*pf.k2*pf.v2/pf.A2) + pf.w2(i); % Derivative for state 2
        pf.x_pr(3,i) = ts*(-pf.a3/pf.A3 * sqrt(2*g*pf.x_po(3,i)) + (1-pf.gamma2)*pf.k2*pf.v2/pf.A3) + pf.w3(i);
        pf.x_pr(4,i) = ts*(-pf.a4/pf.A4 * sqrt(2*g*pf.x_po(4,i)) + (1-pf.gamma1)*pf.k1*pf.v2/pf.A4) + pf.w4(i);    
        pf.x_pr(:,i) = pf.x_pr(:,i) + pf.x_po(:,i);
end

pf.x_pr = abs(pf.x_pr);

% Weight generation based on likelihood density function and innovations
for i = 1:pf.N
        pf.Zest(:,i) = pf.Hd * pf.x_pr(:,i);
        pf.v(:,end+1) = pf.Z_ext(:,i) - pf.Zest(:,i);
        pf.q(end+1) = exp(-0.5 * (pf.v(:,i)' * inv(pf.R) * pf.v(:,i)));  
end

% Generating the Residuals
pf.state_pr(:,end+1) = mean(pf.x_pr,2);
pf.Z_pr(:,end+1) = pf.Hd * pf.state_pr(:,count);
pf.innov(:,end+1) = pf.Z_pr(:,count) - pf.Z(count,:)'  ;
pf.resid(:,end +1) = pf.Z_po(:,count) - pf.Z(count,:)';

for i = 1:pf.N 
        pf.wt(i) =pf.q(i)/sum(pf.q);
end

%Update and Resampling

% Resampling function which favours propogation of samples with higher
% weights associated with them
M = length(pf.wt);
Q = cumsum(pf.wt);
selected = zeros(1, pf.N);
T = linspace(0,1-1/pf.N,pf.N) + rand/pf.N;
i = 1; j = 1;

while(i<=pf.N && j<= M )
    
    while Q(j) < T(i)
        j = j + 1;
    end 
     selected(i) = j;
     pf.x_po(1,i) = pf.x_pr(1,j);
     pf.x_po(2,i) = pf.x_pr(2,j);
     pf.x_po(3,i) = pf.x_pr(3,j);
     pf.x_po(4,i) = pf.x_pr(4,j);
    i = i + 1;
end
count = count + 1;
end
%Plots and Results

  figure(1)
  plot(pf.resid(1,:), 'LineWidth', 1);
  hold on
  plot(pf.innov(1,:), 'LineWidth', 1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("Residue h1","Innovations",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Height (cm)",'FontWeight','bold','FontSize',10)
  title("Remnants", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off  

    figure(2)
  plot(pf.resid(2,:), 'LineWidth', 1);
  hold on
  plot(pf.innov(2,:), 'LineWidth', 1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("Residue h2","Innovations",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Height (cm)",'FontWeight','bold','FontSize',10)
  title("Remnants", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off  

figure(3)
  plot(pf.state_po(1,:), 'LineWidth', 1);
  hold on
  plot(y(:,1), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("estimated "," measured ",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H1 (cm)",'FontWeight','bold','FontSize',10)
  title(" Heights ", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

figure(4)
  plot(pf.state_po(2,:), 'LineWidth', 1);
  hold on
  plot(y(:,2), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("estimated "," measured ",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H2 (cm)",'FontWeight','bold','FontSize',10)
  title(" Heights ", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

 figure(5)
  plot(pf.state_po(3,:), 'LineWidth', 1);
  hold on
  plot(y(:,3), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("estimated "," measured ",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H3 (cm)",'FontWeight','bold','FontSize',10)
  title(" Heights ", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

  figure(6)
  plot(pf.state_po(4,:), 'LineWidth', 1);
  hold on
  plot(y(:,4), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("estimated "," measured ",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H4 (cm)",'FontWeight','bold','FontSize',10)
  title(" Heights ", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

%Measurement sample generation

function dydt = myODEs(t, y)

    global pf;
    global g;
    
    dydt = zeros(4, 1); 
    dydt(1) = -pf.a1/pf.A1 * sqrt(2*g*y(1)) + pf.a3/pf.A1 * sqrt(2*g*y(3)) + pf.gamma1*pf.k1*pf.v1/pf.A1; % Derivative for state 1
    dydt(2) = -pf.a2/pf.A2 * sqrt(2*g*y(2)) + pf.a4/pf.A2 * sqrt(2*g*y(4)) + pf.gamma2*pf.k2*pf.v2/pf.A2; % Derivative for state 2
    dydt(3) = -pf.a3/pf.A3 * sqrt(2*g*y(3)) + (1-pf.gamma2)*pf.k2*pf.v2/pf.A3;
    dydt(4) = -pf.a4/pf.A4 * sqrt(2*g*y(4)) + (1-pf.gamma1)*pf.k1*pf.v2/pf.A4;
end



