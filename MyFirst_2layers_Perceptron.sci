// GENERATING THE TRAINING SET
function [X, y] = generate_dataset (NbEch)
X = rand (2, NbEch);
y = (X(1, :) + X(2, :)) >= 1;
endfunction

NbEch = 1000;
[X_train , y_train] = generate_dataset (NbEch);

// VARIABLES INITIALIZATION
NbIt = 100;
alpha = 0.1; // define a learning rate (set to 0.1 here) ;
n0 = size (X_train, "r"); // number of desired variables in ‘X’ ;
n2 = size (y_train, "r");
n1 = 2; // define the number of neurons in layer 1 ;

// CREATING THE INITIALIZATION FUNCTION
function [W1, b1, W2, b2] = initialize (n0, n1, n2)
// 1st-layer parameters:
W1 = rand (n1, n0);
b1 = rand (n1, 1) ;
// 2nd-layer parameters:
W2 = rand (n2, n1);
b2 = rand (n2, 1);
endfunction

// CREATING THE MODEL FUNCTION (CALLED FORWARD-PROPAGATION)
function [A1, A2] = forward_propagation (X, W1, b1, W2, b2)
Z1 = W1 * X + repmat (b1, 1, size (X, 1+i));
A1 = 1 ./ (1 + exp (-Z1));
Z2 = W2 * A1 + b2* ones (1, size (A1, 1+i));
A2 = 1 ./ (1 + exp (-Z2));
endfunction

// CREATING THE GRADIENT FUNCTION (CALLED BACK-PROPAGATION)
y_size = size (y_train);
m = y_size (2);
function [dW1, db1, dW2, db2] = backpropagation (X, y, A1, A2, W1)

dZ2 = A2 - y;
dW2 = (1 / m) * dZ2 * (A1');
db2 = (1 / m) * sum(dZ2, "c");
db2 = db2 * ones (1, size (dZ2, 2)); // to avoid a broadcasting effect and preserve a 2D matrix
dZ1 = (W2 '* dZ2).*(A1 .*(1 - A1));
dW1 = (1 / m) * dZ1 * (X');
db1 = (1 / m) * sum(dZ1, "c");
db1 = db1 * ones (1, size (dZ1, 2)); // to avoid a broadcasting effect and preserve a 2D matrix
endfunction

// CREATING THE UPDATE FUNCTION
function [W1, b1, W2, b2] = update (dW1, dW2, db1, db2, W1, b1, W2, b2, alpha)
W1 = W1 - alpha * dW1 ;
b1 = repmat (b1, 1, size (db1, 1+i)) - alpha * db1;
W2 = W2 - alpha * dW2 ;
b2 = b2 - alpha * db2 ;
endfunction

// CREATING THE COST FUNCTION
function Loss = logloss (A, y)
m = size (A ,2);
Loss = (-1 / m) * sum ((1 - y) .* log (A) + y .* log (1 - A));
endfunction

// CREATING THE PREDICTION FUNCTION
function result = predict (X, W1, b1, W2, b2) // boolean, i.e., 1 ( TRUE ) or 0 ( FALSE )
[A1, A2] = forward_propagation (X, W1, b1, W2, b2);
result = A2 >= 0.5;
endfunction

// CREATING THE ITERATIVE ALGORITHM
function [W1, W2, b1, b2] = neural_ntw (X_train, y_train, alpha, NbIt, n1)
// parameters initialization
[W1, b1, W2, b2] = initialize (n0, n1, n2);

train_loss =[];
train_acc =[];
for i = 1: NbIt
disp (i)
[A1, A2] = forward_propagation (X_train, W1, b1, W2, b2);
[dW1, db1, dW2, db2] = backpropagation (X_train, y_train, A1, A2, W1);
[W1, b1, W2, b2] = update (dW1, dW2, db1, db2, W1, b1, W2, b2, alpha);
Loss = logloss (A2, y_train);
if ( modulo (i ,10) == 0)
train_loss = [train_loss, Loss];
y_pred = predict (X_train, W1, b1, W2, b2);
end
end

plot (train_loss, 'b', 'LineWidth', 1);
xlabel ('Iterations');
ylabel ('Train Loss');
title ('Train Loss');
legend ('Learning curves of my 2-layers perceptron');

endfunction

[W1, W2, b1, b2] =  neural_ntw (X_train, y_train, alpha, NbIt, n1)
