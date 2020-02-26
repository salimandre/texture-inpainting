function [w] = conv_im(u,v) % convolution de 2 images via fft
    tfu = fft2(u);
    tfv = fft2(v);
    tfw = tfu.*tfv;
    w = real(ifft2(tfw));
end