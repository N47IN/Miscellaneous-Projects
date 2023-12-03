% R Navin Sriram , ED21B044, Modern Control Theory, fall '23
% to be used as an mlx file 
clc; clear; close all;

% initializing the kf structure which will carry all our parameters and
% solutions

global kf;
global g;

% parameters
kf = struct();
kf.A1 = 28; %(cm^2)
kf.A2 = 32;
kf.A3 = 28;
kf.A4 = 32;
kf.A = [kf.A1, kf.A2, kf.A3, kf.A4];
kf.a1 = 0.071; kf.a3 = 0.071; %(cm^2)
kf.a2 = 0.057; kf.a4 = 0.057;
kf.a = [kf.a1, kf.a2, kf.a3, kf.a4];
kf.kc = 0.5; % (V/cm) 
g = 981; %(cm/s^2)
kf.gamma1 = 0.7; kf.gamma2 = 0.6;  
kf.k1 = 3.33; kf.k2 = 3.35; %[cm^3/Vs]
kf.kc = 1; % [V/cm]
kf.kc = 1; % [V/cm]
kf.v1 = 3; kf.v2 = 3; % (V)
kf.U = [kf.v1; kf.v2];
kf.h0 = [12.4; 12.7; 1.8; 1.4];
kf.P_pr = 1000*eye(4); kf.Q = 100*eye(4); kf.R = 10*eye(2);

% desired outputs for plotting

kf.innov=[];
kf.z_res =[];
kf.Y=[];
kf.p_pr_trace = [];
kf.p_po_trace = [];
T = [];
kf.x_pr = [];
kf.Zest = [];
kf.K_norm = [];

% Finding the value for the term T for all the Elements

for j = 1:4
    T(j) =  (kf.A(j)/kf.a(j))*sqrt(2*kf.h0(j)/g) ;
end

% Initializing the Control Input Matrix, State Matrix and Output Matrix

kf.Ac = [ -1/T(1), 0, kf.A3/(kf.A1*T(3)), 0 ; 0, -1/T(2), 0, kf.A4/(kf.A2*T(4)); 0, 0, -1/T(3), 0; 0, 0, 0, -1/T(4)];
kf.Bc = [kf.gamma1*kf.k1/kf.A1 0 ; 0 kf.gamma2*kf.k2/ kf.A2; 0 (1 - kf.gamma2)*kf.k2/kf.A3; (1-kf.gamma1)*kf.k1/kf.A4 0];
kf.Dc = 0;
kf.Hc = [kf.kc 0 0 0; 0 kf.kc 0 0];

% Initializing the array containing posterior values of the States, with
% first val being h0

kf.x_po(:,1) = kf.h0;

% finding the solution to the ODEs using ODE45 for a give time and 10,000
% samples
t_span = linspace(0, 1000, 10000);
[t, y] = ode45(@myODEs, t_span, kf.h0);

% initialising the array containing all the measurements

kf.Z = [y(:,1:2)] * kf.kc;

[rows,cols] = size(t);

% changing the Matrices to discrete domain based on step of 0.1
state_space = ss(kf.Ac, kf.Bc, kf.Hc, kf.Dc);
state_space_discrete = c2d(state_space, 0.1);
kf.Ad = state_space_discrete.A;
kf.Bd = state_space_discrete.B;
kf.Hd = state_space_discrete.C;
kf.Dd = state_space_discrete.D;

% Kalman Filter loop, count is the iterable representing the number of
% prediction + update steps performed
% We divide the filtering into 3 steps, 1) priori evaluation, 2) Kappa
% Evaluation, 3) Posterior evaluation for which we have 3 fns
for  count = 1:rows

    % initialzing the Array consisting of Traces of Prior Process Covariance
    kf.p_pr_trace(end+1) = trace(kf.P_pr);
    kf = priori(kf,count);
    kf = Kappa(kf);
    kf = posterior(kf,count);

    % initializing the residual array
    kf.z_res(:,end+1) = kf.Z(count,:)' - kf.x_po(1:2,count+1);
    % initialzing the Array consisting of Traces of Posterior Process Covariance
    kf.p_po_trace(end+1) = trace(kf.P_pr);

end

% Initialzing the Innovation array
kf.innov = kf.Z(1:length(kf.Zest),:)' - kf.Zest;

% All plots

  figure(1)
  plot(kf.innov(1,:), 'LineWidth', 1);
  hold on
  plot(kf.z_res(1,:), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("Innovation","Residuals",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Height (cm)",'FontWeight','bold','FontSize',10)
  title("Remnants", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

  figure(2)
  plot(kf.x_pr(1,:), 'LineWidth', 1);
  hold on
  plot(kf.x_po(1,:), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("x_k^-","x_k^+",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H1 (cm)",'FontWeight','bold','FontSize',10)
  title("Prior and Posterior", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

   figure(3)
  plot(kf.x_pr(2,:), 'LineWidth', 1);
  hold on
  plot(kf.x_po(2,:), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("x_k^-","x_k^+",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H2 (cm)",'FontWeight','bold','FontSize',10)
  title("Prior and Posterior", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

   figure(4)
  plot(kf.x_pr(3,:), 'LineWidth', 1);
  hold on
  plot(kf.x_po(3,:), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("x_k^-","x_k^+",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H3 (cm)",'FontWeight','bold','FontSize',10)
  title("Prior and Posterior", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

   figure(5)
  plot(kf.x_pr(4,:), 'LineWidth', 1);
  hold on
  plot(kf.x_po(4,:), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("x_k^-","x_k^+",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("H4 (cm)",'FontWeight','bold','FontSize',10)
  title("Prior and Posterior", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

  figure(6)
  plot(kf.p_pr_trace, 'LineWidth', 1);
  hold on
  plot(kf.p_po_trace, 'LineWidth', 1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("tr(P_k^-)","tr(P_k^+)",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Trace",'FontWeight','bold','FontSize',10)
  title("Process Covariance trace", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

  figure(7)
  plot(kf.x_po(1,:), 'LineWidth', 1);
  hold on
  plot(kf.x_po(2,:), 'LineWidth', 1);
  plot(kf.x_po(3,:), 'LineWidth', 1);
  plot(kf.x_po(4,:), 'LineWidth', 1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("h1","h2","h3","h4",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Height in cm",'FontWeight','bold','FontSize',10)
  title("Estimated heights", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  ylim([0 15])
  hold off

  figure(8)
  plot(kf.K_norm, 'LineWidth', 1);
  hold on
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("Norm1",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Norm of Kalman Gain",'FontWeight','bold','FontSize',10)
  title("Kalman Gain", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off

  figure(9)
  plot(kf.Y(1,:), 'LineWidth', 1);
  hold on
  plot(kf.Y(2,:), 'LineWidth', 1);

  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("V1","V2",'fontsize',6);
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Control Inputs",'FontWeight','bold','FontSize',10)
  title("Control Reference Inputs", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off





function kf = priori(kf,count)
Xpost = kf.x_po(:,count);
kf.Y(:,end+1) =  awgn(kf.U , 45 );
Xpr= kf.Ad*Xpost + kf.Bd*kf.Y(:,count) ;
kf.P_pr = kf.Ad * kf.P_pr * kf.Ad' + kf.Q  ;
kf.x_pr(:,end+1) = Xpr;

end

% evaluating Kappa
function kf = Kappa(kf)

dr = kf.Hd * kf.P_pr * kf.Hd' +  kf.R;
nr = kf.P_pr * kf.Hd' ;
kf.kappa = nr / dr;
kf.K_norm(end+1)=norm(kf.kappa);

end

% evaluating Posterior density
function kf = posterior(kf, count)

kf.x_po(:,end+1) = kf.x_pr(:,count) + kf.kappa * ( kf.Z(count,:)' - kf.Hd * kf.x_pr(:,count) ) ;
kf.P_pr = (eye(4) - kf.kappa*kf.Hd)*kf.P_pr;
kf.Zest (:,end+1) = kf.Hd * kf.x_pr(:,count);

end

% function representing all ODEs taken as an arg in the ODE45 solver!
function dydt = myODEs(t, y)

    global kf;
    global g;
    
    dydt = zeros(4, 1); 
    dydt(1) = -kf.a1/kf.A1 * sqrt(2*g*y(1)) + kf.a3/kf.A1 * sqrt(2*g*y(3)) + kf.gamma1*kf.k1*kf.v1/kf.A1; % Derivative for state 1
    dydt(2) = -kf.a2/kf.A2 * sqrt(2*g*y(2)) + kf.a4/kf.A2 * sqrt(2*g*y(4)) + kf.gamma2*kf.k2*kf.v2/kf.A2; % Derivative for state 2
    dydt(3) = -kf.a3/kf.A3 * sqrt(2*g*y(3)) + (1-kf.gamma2)*kf.k2*kf.v2/kf.A3;
    dydt(4) = -kf.a4/kf.A4 * sqrt(2*g*y(4)) + (1-kf.gamma1)*kf.k1*kf.v2/kf.A4;
end




