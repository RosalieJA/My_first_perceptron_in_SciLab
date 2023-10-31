// GENERATE THE DATASET (X, Y)
NbLignes = 100;
NbVar = 2;
rand ("seed",0);

// GENERATE BIMODAL DISTRIBUTED VALUES
X1 = rand (NbLignes /2, NbVar) + 0.4; // distrib #1 (add +0.4 to each value) ;
X2 = rand (NbLignes /2, NbVar) - 0.4; // distrib #2 (remove -0.4 to each value) ;
X = [X1; X2]; // Concatenate the 2 distributions

// ATTRIBUTE BINAY LABELS
y1 = ones (NbLignes/2, 1); // ‘class 1’ label for X1 ;
y2 = zeros (NbLignes/2, 1); // ‘class 2’ label for X2 ;
y = [y1; y2]; // concatenate labels ;

// GRAPHICAL REPRESENTATION OF THE TRAINING SET
clf
plot (X1 (: ,1), X1 (: ,2), 'bo');
plot (X2 (: ,1), X2 (: ,2), 'go');
xlabel ('Variable 1');
ylabel ('Variable 2');
title ('Dataset visualization');

// CREATING THE INITIALIZATION FUNCTION
function [W, b] = initialize (X)
W = rand (NbVar, 1);
b = rand ();
endfunction

// CREATING THE MODEL FUNCTION
function A = model (X, W, b)
Z = X * W + b;
A = 1 ./ (1 + exp(-Z));
endfunction

// CREATING THE COST FUNCTION
m = NbLignes ;
function Loss = logloss (A, y)
Loss = (-1 / m) * sum ((1 - y) .* log (A) + y .* log (1 - A));
endfunction

// CREATING THE GRADIENT FUNCTION
function [dW , db] = gradients (A, X, y)
dW = (1 / m) * X' * (A - y);
db = (1 / m) * sum (A - y);
endfunction

// CREATING THE UPDATE FUNCTION
function [W, b] = update (dW, db, W, b, alpha)
W = W - alpha * dW;
b = b - alpha * db;
endfunction

// CREATING THE PREDICTION FUNCTION
function result = predict (X, W, b) // Boolean, i.e.: 1 (TRUE) or 0 ( FALSE ) ;
A = model (X, W, b);
result = A >= 0.5;
endfunction

// BUILD THE ITERATIVE ALGORITHM TO TRAIN THE NEURON WEIGHTS
alpha = 0.1; // define a learning rate (set at 0.1 here) ;
NbIt = 100; // define the number of iterations (set at 100 here) ;

function [W, b, ErrorList] = artificial_neuron (X, y, alpha, NbIt)
[W, b] = initialize (X); // Initialize W and b ;

Iteration = 1:NbIt ;
ErrorList = zeros (NbIt, 1);

for i = 1:NbIt
A = model (X, W, b);
Loss = logloss (A, y);
[dW , db] = gradients (A, X, y);
[W, b] = update (dW, db, W, b, alpha);
ErrorList (i) = Loss ; // accuracy = (1- Loss) ;
end

endfunction

[W, b, ErrorList] = artificial_neuron (X, y, alpha, NbIt);
// clf ();
// plot (1:NbIt , ErrorList , 'r');
// xlabel ('Learning cycles');
// ylabel ('Error');

// FUTURE PREDICTIONS
newElement = rand (4 ,2);
plot (newElement (: ,1) , newElement (: ,2) ,'rx','MarkerSize' ,5);

y_predbis = predict (newElement , W, b);
disp (y_predbis);

// PLOT THE DECISION BOUNDARY
x0 = linspace (max (X(:, 1))+0.5 , min (X(:, 1)) -0.5 , 100) ;
x1 = (-W(1) * x0 - b) / W(2) ;
plot (x0 , x1 , 'r');
