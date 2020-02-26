function [sptn]=dsptn(h,n) % return a sequence of randomly shifted spot 
[M,N,~]=size(h);
sptn=zeros(M,N);
for i=1:n
    t_0=[floor(rand()*M)+1;floor(rand()*N)+1];
    [v]=shift_im(h,t_0);
    %disp(min(v(:)));
    sptn=sptn+v;
end
end 