function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ ones(m, 1) X]; % (5000 x 401) which is 5000 exmaples with 
                     %  401 features(including bias units)
a2 = sigmoid(Theta1*X'); % (25(hidden_layer_size) x 5000) taking the X' to aling 
                         % the outputs with the neural network architecture
a2 = [ ones(m, 1) a2' ]; % adding bias unit for second layer ( 5000 * 26)

h = sigmoid(Theta2*a2'); % (10 x 5000) prediction for all 5000 examples) 


y_vf = zeros(num_labels,m); % ( 10 x 5000) y in vector form


% changing y into vector form
for i = 1:m
  y_vf(y(i), i)= 1;
endfor

J = (1/m)*sum(sum( (-y_vf) .* log(h)-(1-y_vf) .* log(1-h) ));

t1 = Theta1( :, 2:size(Theta1, 2));
t2 = Theta2( :, 2:size(Theta2, 2));

sum_sqr = sum( sum( t1.^2)) + sum( sum( t2.^2));
disp(sum_sqr);
J = J + (lambda/(2*m))*sum_sqr;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for i = 1:m
  a1 = X(i, :); %(1 x 401) an example which is a row in X 
  a1 = a1'; % (401 x 1)
  z2 = Theta1* a1; % (25 x 1) as Theta1 is (25 x 401)  
  a2 = sigmoid(z2); 
  
  a2 = [ 1; a2]; % (26 x 1) adding bias unit 
  z3 = Theta2 * a2; % (10 x 1) as Theta2 is (10 x 26)
  a3 = sigmoid(z3); % 
  
  d3 = a3 - y_vf(:, i); % (10 x 1) each columm is the label for an example
  
  z2 = [1; z2]; % ( 26 x 1) adding the bias unit
  
  d2 = (Theta2'*d3).* sigmoidGradient(z2); % ( ) which comes from ((26x10)*(10x1)).*(26x1)
  
  d2 = d2(2: end); % (25 x1) skipping d2(1) as it is bias term
  
  Theta2_grad = Theta2_grad + d3 * a2'; % (10 x 26) which comes from (10x1)*(1x26)
  Theta1_grad = Theta1_grad + d2 * a1'; % (25 x 401) which comes from (25x1)*(1x401)

  
endfor

Theta1_grad = (1/m) * Theta1_grad; 
Theta2_grad = (1/m) * Theta2_grad;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 


% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
