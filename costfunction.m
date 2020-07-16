function J = costfunction(X,y,theta,lambda)
    m = length(X); % number of training examples
    prediction = X*theta;
    sqErrors = (prediction-y).^2;
    J = 1/(2*m)*sum(sqErrors) + lambda*sum((lambda(2:end)).^2)/(2*m);
end

