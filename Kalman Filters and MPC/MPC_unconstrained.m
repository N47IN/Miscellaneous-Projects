% R Navin Sriram , ED21B044, Modern Control Theory, fall '23
% to be used as an mlx file 
clc; clear; close all;

% initializing the kf structure which will carry all our parameters and
% solutions

global kf;
global g;
global mpc;

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
T = [];
kf.x_pr = [];
kf.err=[];

% desired outputs for plotting
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
kf.x_po(:,2) = kf.h0;
kf.Z = [];

% changing the Matrices to discrete domain based on step of 0.1
state_space = ss(kf.Ac, kf.Bc, kf.Hc, kf.Dc);
state_space_discrete = c2d(state_space, 0.1);
kf.Ad = state_space_discrete.A;
kf.Bd = state_space_discrete.B;
kf.Hd = state_space_discrete.C;
kf.Dd = state_space_discrete.D;





mpc = struct();
mpc.Np = 50;
mpc.Nc = 30 ;
mpc.A = [kf.Ad, zeros(4,2);kf.Hd*kf.Ad, eye(2)];
mpc.B = [kf.Bd; kf.Hd*kf.Bd];
mpc.C = [zeros(2,4), eye(2)];
mpc.F = mpc.C*mpc.A;
mpc.reference = kf.Hd*[13.4; 13.7; 0; 0];
mpc.R = 0.5*eye(2*mpc.Nc);
mpc.U = [kf.v1; kf.v2];
mpc.Y = zeros(2,1);
mpc.F = mpc.C*mpc.A;
dims1 = size(kf.Hd);
mpc.q = dims1(1);
dims2 = size(kf.Bd);
mpc.m = dims2(2)

mpc.b = []

for n = 2:mpc.Np
    mpc.F = vertcat(mpc.F, mpc.C*mpc.A^n);
    mpc.reference = vertcat(mpc.reference, [13.4; 13.7]);
end

dim = size(mpc.C*mpc.B);
mpc.phi = horzcat(mpc.C*mpc.B, zeros(dim(1), dim(2)*(mpc.Nc-1)));
for p = 1:mpc.Np-1
    filler = mpc.C*(mpc.A^p)*mpc.B;
    for c = 1: p
        if c+1<=mpc.Nc
            filler = horzcat(filler, mpc.C*(mpc.A^(p-c))*mpc.B);
        end
    end
    if (p+1<=mpc.Nc)
        for c = 2:mpc.Nc-p
            filler = horzcat(filler, zeros(dim));
        end
    end
    mpc.phi = vertcat(mpc.phi, filler);
    
end
% size(mpc.phi)
% mpc.a = [-eye(mpc.q*mpc.Np); 
%     eye(mpc.q*mpc.Np);-mpc.phi;
%     mpc.phi; tril(eye(mpc.q*mpc.Np)); 
%     -tril(eye(mpc.q*mpc.Np))]


iter = 1000;
for i = 1:iter
  mpc = uncstr(mpc,kf);
  mpc = plant(mpc,kf);
  kf = Kalman(kf,mpc);
end



  figure(1)
  plot(kf.x_po(1,:), 'LineWidth', 1);
  hold on
  plot(kf.x_po(2,:), 'LineWidth', 1);
  plot(kf.x_po(3,:), 'LineWidth', 1);
  plot(kf.x_po(4,:), 'LineWidth', 1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("H1","H2","H3","H4",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Height (cm)",'FontWeight','bold','FontSize',10)
  title("Heights", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  ylim([0 15])
  hold off

  figure(2)
  plot(kf.err(1,:), 'LineWidth', 1);
  hold on
  plot(kf.err(2,:), 'LineWidth',1);
  grid on
  ax = gca;
  ax.FontSize = 10;
  legend("delH1","delH2",'FontSize',6)
  xlabel("Iteration",'FontWeight','bold','FontSize',10)
  ylabel("Error (cm)",'FontWeight','bold','FontSize',10)
  title("Error from Setpoint", 'FontWeight', 'bold','FontSize',10)
  axis tight   
  hold off



function kf = priori(kf, mpc)
    Xpost = kf.x_po(:,end);
    Xpr= kf.Ad*Xpost + kf.Bd * mpc.U(:,end);
    kf.P_pr = kf.Ad * kf.P_pr * kf.Ad' + kf.Q  ;
    kf.x_pr(:,end+1) = Xpr;
end

function kf = Kappa(kf)
    dr = kf.Hd * kf.P_pr * kf.Hd' +  kf.R;
    nr = kf.P_pr * kf.Hd' ;
    kf.kappa = nr / dr;
end

function kf = posterior(kf,mpc)
    kf.x_po(:,end+1) = kf.x_pr(:,end) + kf.kappa * ( mpc.Y(:,end) - kf.Hd * kf.x_pr(:,end) ) ;
    kf.P_pr = (eye(4) - kf.kappa*kf.Hd)*kf.P_pr;
    kf.err(:,end+1) = kf.x_po(1:2,end) - [13.4;13.7];
end

function kf = Kalman(kf,mpc)
    kf = priori(kf,mpc);
    kf = Kappa(kf);
    kf = posterior(kf,mpc);
end

function mpc = uncstr(mpc,kf)
    x_cur = kf.x_po(:, end);
    x_prev = kf.x_po(:, end-1);
    del_x = x_cur - x_prev;
    mpc.x = [del_x; mpc.Y(:,end)];
    delta = (mpc.phi' * mpc.phi + mpc.R)\ (mpc.phi' * (mpc.reference - mpc.F*mpc.x));
    mpc.U(:, end+1) = delta(1:2) + mpc.U(:,end);
end

function mpc = cstr(mpc,kf)
    x_cur = kf.x_po(:, end);
    x_prev = kf.x_po(:, end-1);
    del_x = x_cur - x_prev;
    mpc.x = [del_x; mpc.Y(:,end)];
    H = mpc.phi' * mpc.phi + mpc. round ;
    F = mpc.phi' * (mpc.reference -mpc.F * mpc.x ) ;
    mpc.U(:, end+1) = quadprog(H,F)
end

function mpc = plant(mpc,kf)
   mpc.Y(:,end+1) = kf.Hd * ( kf.Ad * kf.x_po(:,end) + kf.Bd * mpc.U(:,end));
   mpc.Y(:,end) = awgn(mpc.Y(:,end),50);
end


