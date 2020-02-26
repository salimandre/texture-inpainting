function [B]=lowerTriu(v)
A = upperTriu(fliplr(v))';
B = A - diag(diag(A));
end