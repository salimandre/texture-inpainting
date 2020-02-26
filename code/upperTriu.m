function [A]=upperTriu(v)
M=length(v);
%A = zeros(M,M);
vA = zeros(M*M,1);
for i=1:M
    vA(i:(M+1):(M-i+1)*M,1)=v(i);
end
A = reshape(vA,M,M);
%A=A+A'-diag(diag(A));
A=A';
end