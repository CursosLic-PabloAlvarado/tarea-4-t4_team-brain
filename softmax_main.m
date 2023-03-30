% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 4
% (C) 2023 <Team brAIn>

% Logistic regression testbench

clear all; close all;
[Xtr,Ytr,Xte,Yte,names] = loadpenguindata("species");
Xtr = [ones(length(Xtr),1) Xtr];
Xte = [ones(length(Xte),1) Xte];
Y = Ytr;
%Yte = Yte;
NX=normalizer("normal");
NXtr=NX.fit_transform(Xtr);
NXte=NX.transform(Xte);

## Initial configuration for the optimizer
opt=optimizer("method","sgd",
              "minibatch",11,
              "maxiter",600,
              "alpha",0.05);
###

theta0=rand(columns(NXtr),2)-0.5; ## Common starting point (column vector)

# test all optimization methods
methods={"sgd","momentum","rmsprop","adam","batch"};
##methods={"batch"};

%tts1=zeros(numel(methods),5); %para los thetas
%es=zeros(1,67);%para la grafica 2d

for m=1:numel(methods)
  method=methods{m};
  printf("Probando método '%s'.\n",method);
  msg=sprintf(";%s;",method); ## use method in legends

  %try
    opt.configure("method",method); ## Just change the method
    [ts,errs]=opt.minimize(@softmax_loss,@softmax_gradloss,theta0,NXtr,Y);
    theta=ts{end}

    py=softmax_hyp(theta,NXte);
    err=sum((py>0.5)!=Yte);
    tot=100*(err/rows(Yte));

    printf("errores de prueba: %d de %d (%.2f%%)\n", err, length(Yte), tot);

    py=softmax_hyp(theta,NXtr);
    err=sum((py>0.5)!=Y);
    tot=100*(err/rows(Y));
    printf("errores de entreneamineto: %d de %d (%.2f%%)\n", err, length(Y), tot);


    figure(1);
    plot(errs,msg,"linewidth",2);
    hold on;
  %catch
   % printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
    %       method,lasterror.message);
  %end_try_catch
endfor
xlabel("Iteration");
ylabel("Loss");
grid on;

##################################################################################
comp=100;
columna1=0;
columna2=0;

for i=1:4
  for j=i+1:5

    feats=[i,j];
    x2=Xtr(:,feats);
    N2=normalizer("normal");
    nx2=N2.fit_transform(x2);

    opt.configure("method","batch"); ## Just change the method
    [ts,errs]=opt.minimize(@softmax_loss,@softmax_gradloss,theta0(feats,:),nx2,Y);
    theta2=ts{end};

    py2=softmax_hyp(theta2,nx2);
    err2=sum((py2>0.5)!=Y);
    tot2=100*(err2/rows(Y));

    if tot2<=comp
      comp=tot2;
      columna1=i;
      columna2=j;
    endif

    mins=min(x2);
    maxs=max(x2);

    e1=linspace(mins(1),maxs(1),50);
    e2=linspace(mins(2),maxs(2),50);
  endfor

endfor

printf("el menor error obtenido es: %d al evaluar las columnas %d y %d\n", comp, columna1, columna2);


[ee1,ee2]=meshgrid(e1,e2);
x2test=N2.transform([ee1(:) ee2(:)]);

ytest=softmax_hyp(theta2,x2test);


################################
figure(2,"name","Probabilidad para Adelie")

colormap(hot);
surf(ee1,ee2,reshape(ytest(:,1),size(ee1)));
xlabel("culmen length [mm]");
ylabel("bodymass [g]");
zlabel("p(Adelie|x");
hold on;
contour3(ee1,ee2,reshape(ytest(:,1),size(ee1)),[0.25,0.5,0.75],"linewidth",3,"linecolor","black");
################################
figure(3,"name","Probabilidad para Chinstrap")

colormap(rainbow);
surf(ee1,ee2,reshape(ytest(:,2),size(ee1)));
xlabel("culmen length [mm]");
ylabel("bodymass [g]");
zlabel("p(Chinstrap|x");
hold on;
contour3(ee1,ee2,reshape(ytest(:,2),size(ee1)),[0.25,0.5,0.75],"linewidth",3,"linecolor","black");
################################
figure(4,"name","Probabilidad para Gentoo")

colormap(winter);
surf(ee1,ee2,reshape(ytest(:,3),size(ee1)));
xlabel("culmen length [mm]");
ylabel("bodymass [g]");
zlabel("p(Gentoo|x");
hold on;
contour3(ee1,ee2,reshape(ytest(:,3),size(ee1)),[0.25,0.5,0.75],"linewidth",3,"linecolor","black");
################################

ygrap=zeros(size(ytest(:,1)));
for i=1:length(ygrap)
  c1=ytest(i,1);
  c2=ytest(i,2);
  c3=ytest(i,3);
  if c1>c2 && c1>c3
    ygrap(i)=1;
  endif
  if c2>c1 && c2>c3
    ygrap(i)=2;
  endif
  if c3>c2 && c3>c1
    ygrap(i)=3;
  endif
endfor

cmap = [1,0,0; 0,1,0; 0,0,1];
img = reshape(ygrap,size(ee1));
rgb_img = ind2rgb(img, cmap);
figure(5,"name","Regiones de las clases ganadoras para el espacio de entrada bidimensional");
image(rgb_img);
xlabel("culmen length [mm]");
ylabel("bodymass [g]");
axis equal;
hold on;

y_prob = (ytest) ./ sum((ytest), 2);
color_weight = y_prob * cmap'; % Calcula los pesos para cada color
##mixed_color = reshape(color_weight,size(ee1)); % Mezcla los colores
mixed_color= reshape(color_weight, [50 50 3]);

figure(6,"name","Ponderación de colores asignados a las clases, de acuerdo a la probabilidad de pertenecer a esa clase");
image(mixed_color);
xlabel("culmen length [mm]");
ylabel("bodymass [g]");
axis equal;





