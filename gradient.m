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

    theta(1) = temp0;
    theta(2) = temp1;

    J_hist = costfunction(X,y,theta,lambda);
    J_history(i,1) = J_hist;

end
end
