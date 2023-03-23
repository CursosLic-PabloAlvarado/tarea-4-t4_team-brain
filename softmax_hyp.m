% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Team brAIn>



% Hypothesis function used in softmax
% Theta: matrix, its columns are each related to one
%        particular class.
% returns the hypothesis, which has only k-1 values for each sample
%         as the last one is computed as 1 minus the sum of all the rest.
function h=softmax_hyp(Theta,X)
  ## Dummy implementation just returns a random value
 % h=rand(rows(X),columns(Theta));
  h=exp( Theta'*X' );
  nor=sum(h) + ones(1,columns(h)); ## the ones 'cause exp(0) for k
  h = h ./ nor;
endfunction
