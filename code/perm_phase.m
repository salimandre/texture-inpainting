function [uuu,vvv]=perm_phase(u,v)  %% permute phase des tdf des 2 images en input
    tfu = fft2(u);
    tfv = fft2(v);
    tfuu = abs(tfu) .* tfv ./ abs(tfv);
    tfvv = abs(tfv) .* tfu ./ abs(tfu);
    uuu = real(ifft2(tfuu));
    vvv = real(ifft2(tfvv));
end     