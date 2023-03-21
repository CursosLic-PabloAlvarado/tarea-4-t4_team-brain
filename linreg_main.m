% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
%
% Linear regression as testbench for optimizer

clear all; close all;

% Generate some artificial data:
x = 1.5*rand(30,1)-0.5; ## random data between -0.5 and 1
X = bsxfun(@power,x,0:2);
y = x.^2 + 0.05*rand(30,1); ## add noise

% Show the data
figure(1,"name","optimizer testbench");

% show the original data
plot(X(:,2),y,'x');
xlabel("x");
ylabel("y");
axis([-0.5,1,-0.1,1.1]);
grid on;
hold on;


figure(2,"name","Loss evolution");
hold on;

function h=linreg_hyp(theta,X)
  % theta must be a column vector
  h=X*theta(:);
endfunction

function err=linreg_loss(theta,X,y)
  ## residuals
  r=y-linreg_hyp(theta,X);
  err=0.5*(r'*r); # OLS
endfunction

function grad=linreg_gradloss(theta,X,y)
  ## residuals
  h=linreg_hyp(theta,X);
  grad=sum((h-y).*X);
endfunction

## Initial configuration for the optimizer
opt=optimizer("method","sgd",
              "minibatch",8,
              "maxiter",500,
              "alpha",0.002);
              
theta0=rand(columns(X),1)-0.5; ## Common starting point (column vector)          

px=bsxfun(@power,linspace(-0.5,1,100)',0:2);

# test all optimization methods
methods={"batch","sgd","momentum","rmsprop","adam"};
for m=1:numel(methods)
  method=methods{m};
  printf("Probando método '%s'.\n",method);
  msg=sprintf(";%s;",method); ## use method in legends

  try
    opt.configure("method",method); ## Just change the method
    [ts,errs]=opt.minimize(@linreg_loss,@linreg_gradloss,theta0,X,y);
    theta=ts{end};
    
    figure(1);
    py=linreg_hyp(theta,px);
    plot(px(:,2),py,msg,"linewidth",2);
    
    figure(2);
    plot(errs,msg,"linewidth",2);  
  catch
    printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
           method,lasterror.message);
  end_try_catch
endfor

figure(2);
xlabel("Iteration");
ylabel("Loss");
grid on;  
