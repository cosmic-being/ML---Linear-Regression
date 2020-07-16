% Salary Problem
clear all
clc

% Split the data
data = csvread('Salary_Data.csv',1,0);

cv = cvpartition(size(data,1),'HoldOut',0.4);
idx = cv.test;
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

% cv = cvpartition(size(irrev,1),'HoldOut',0.5);
% idx_T = cv.test;
% dataCV = data(~idx_T,:);
% dataTest  = data(idx_T,:);

X = dataTrain(:,1);
y = dataTrain(:,2);
% Xval = dataCV(:,1);
% yval = dataCV(:,2);
Xtest = dataTest(:,1);
ytest = dataTest(:,2);
figure(1)
plot(X,y,'r.','MarkerSize',30);
xlabel('Years of Experience')
ylabel('Salary')
figure(3)
plot(Xtest,ytest,'r.','MarkerSize',30);
xlabel('Years of Experience')
ylabel('Salary')

m = length(X); % number of training examples
X = [ones(m, 1), dataTrain(:,1)]; % Add a column of ones to x
m1 = length(Xtest); % number of training examples
Xtest = [ones(m1, 1), dataTest(:,1)]; % Add a column of ones to x

theta = zeros(2, 1); % initialize fitting parameters
iteration = 1000;
alpha = 0.1;
lambda = 1;
%%
[theta,J] = gradient(X,y,theta,alpha,iteration,lambda);

Jtest = costfunction(Xtest,ytest,theta,lambda);
figure(1)
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))
hold on; % keep previous plot visible
plot(X(:,2), X*theta,'b-', 'LineWidth', 1.5);
legend('Training data', 'Linear regression')
title('Salary vs Experience(Training Set)')
hold off

figure(2)
% Plot the convergence graph
plot(1:iteration, J, 'b.', 'LineWidth',1.5);
xlabel('Number of iterations');
ylabel('Cost J');
title('Convergence Test')

figure(3)
hold on; % keep previous plot visible
plot(X(:,2), X*theta,'b-', 'LineWidth', 1.5);
legend('Test data', 'Linear regression')
title('Salary vs Experience(Test Set)')
hold off

%%
