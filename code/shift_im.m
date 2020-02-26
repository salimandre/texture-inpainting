function [v] = shift_im(u,t) %input: u image
[M,N,nc]= size(u);  
v=u;
for chan=1:nc
    if t(1)>0
    v(:,:,chan) = v([t(1)+1:M,1:t(1)],:,chan);
    end
    if t(2)>0
    v(:,:,chan) = v(:,[t(2)+1:N,1:t(2)],chan);
    end
end
end