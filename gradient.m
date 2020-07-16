function [theta, J_history] = gradient(X,y,theta,alpha,num_iter,lambda)
J_history = zeros(num_iter, 1);
m = length(X); % number of training examples
temp = zeros(length(theta),1);
for i = 1:num_iter
    h = X*theta; 
    grad = (1/m)*(X'*(h-y));
    grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);
    
    temp0 = theta(1) - alpha*grad(1)/m;
    temp1 = theta(2) - alpha*grad(2)/m;
%     temp2 = theta(3) - alpha*grad(3)/m;
%     temp3 = theta(4) - alpha*grad(4)/m;
% 
    theta(1) = temp0;
    theta(2) = temp1;
%     theta(3) = temp2;
%     theta(4) = temp3;

%     temp = theta - (alpha/m)*grad;
%     theta = temp;
    J_hist = costfunction(X,y,theta,lambda);
    J_history(i,1) = J_hist;
%     hold on
%     plot(num_iter,J_history(num_iter),"rx",'MarkerSize',5)
end
end