function [v]=chgtcaff(u1,u2) %chngt de constrate affine 

pmax = max(u2(:)); % quant source
pmin = min(u2(:));

qmax = max(u1(:)); % quant cible
qmin = min(u1(:));

v = ((qmax-qmin)/(pmax-pmin)) .* (u2-pmin) + qmin;

end