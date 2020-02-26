close all;
clear
clc
fprintf('Hi\ninput: micro-texture with masked region\noutput: inpainted micro-texture\n\n');
%% input image 

im_name='wood.jpg';
imrgb = imread(im_name);  % load an image
fprintf('loading... '); disp(im_name); fprintf('\n');
[M,N,nchan] = size(imrgb);  % image size
figure;

u = double(imrgb); 
u_R = u(:,:,1);  %input image in red
u_G = u(:,:,2);  %input image in green
u_B = u(:,:,3);  %input image in blue

u_mean_R = sum(u_R(:))/(M*N); u_mean_G = sum(u_G(:))/(M*N); u_mean_B = sum(u_B(:))/(M*N); 
u_mean = sum(u(:))/(nchan*M*N); 
imshow(imrgb,[0, 255]), title('input image');

%% mask & masked image

xmin = 65; xmax = 104;
ymin = 65; ymax = 104;

mask = {xmin:xmax,ymin:ymax};
mask_M=xmax-xmin+1; mask_N=ymax-ymin+1; card_mask=mask_M*mask_N;
fprintf('mask size: %d * %d / wrt image %3.4f%% \n\n', mask_M,mask_N, 100*mask_M*mask_N/(M*N));

u_R_masked = u_R; u_G_masked = u_G; u_B_masked = u_B; u_masked=zeros(M,N,3);
u_R_masked(mask{1},mask{2}) = zeros(length(mask{1}),length(mask{2})); u_masked(:,:,1)=u_R_masked;
u_G_masked(mask{1},mask{2}) = zeros(length(mask{1}),length(mask{2})); u_masked(:,:,2)=u_G_masked;
u_B_masked(mask{1},mask{2}) = zeros(length(mask{1}),length(mask{2})); u_masked(:,:,3)=u_B_masked;

% figure;
% imshow(u_R_masked,[0, 255]), title('masked red channel from image input');
% figure;
% imshow(u_G_masked,[0, 255]), title('masked green channel from image input');
% figure;
% imshow(u_B_masked,[0, 255]), title('masked blue channel from image input');
figure
imshow(uint8(u_masked),[0, 255]), title('masked image rgb from image input');

%% spot of width 2*dP

P=[250-randi(100),280]; %Spot defini autour de ce point ! Attention aux dim!
dP= 90;
omega_spot = {(P(1)-dP):(P(1)+dP-1),(P(2)-dP):(P(2)+dP-1)};
wx = omega_spot{1}; wy = omega_spot{2};
fprintf('spot size: %d * %d / wrt image %3.1f%% \n\n', 2*dP,2*dP, 100*(2*dP)^2/(M*N));

spot_R_ = zeros(M,N); spot_G_ = zeros(M,N); spot_B_ = zeros(M,N);
spot_R = u_R(wx,wy); spot_G = u_G(wx,wy); spot_B = u_B(wx,wy);
spot_R_(wx,wy) = spot_R; spot_G_(wx,wy) = spot_G; spot_B_(wx,wy) = spot_B;

spot_show=zeros(M,N,3); spot_show(:,:,1)=spot_R_; spot_show(:,:,2)=spot_G_; spot_show(:,:,3)=spot_B_;

figure;
imshow(uint8(spot_show),[0 255]), title('Spot Noise all channels');
% figure;
% imshow(spot_R_,[0, 255]), title('red spot');
% figure;
% imshow(spot_G_,[0, 255]), title('blue spot');
% figure;
% imshow(spot_B_,[0, 255]), title('green spot');

%% conditionnal sample set of width delta

delta = 20; % largeur bande autour masque
K=3; % K subsets of conditionnal set
fprintf('width of conditionnal set: %d\n', delta);

cs_1 = {(xmin-delta):(xmin-1),(ymin-delta):(ymax+delta)};  %conditionnal set part 1
cs_2 = {xmin:xmax,[(ymin-delta):(ymin-1),(ymax+1):(ymax+delta)]};
cs_3 = {(xmax+1:xmax+delta),(ymin-delta:ymax+delta)};

cs = {cs_1,cs_2,cs_3}; %conditionnal set

u_cond_R_ = zeros(M,N); u_cond_G_ = zeros(M,N); u_cond_B_ = zeros(M,N);
u_cond_bin_R = zeros(M,N); u_cond_bin_G = zeros(M,N); u_cond_bin_B = zeros(M,N); % 1 if in cond set 0 if not 
for i =1:K
    %red channel
u_cond_R_(cs{i}{1},cs{i}{2}) = u_R(cs{i}{1},cs{i}{2});
u_cond_bin_R(cs{i}{1},cs{i}{2}) = 1;
    %green channel
u_cond_G_(cs{i}{1},cs{i}{2}) = u_G(cs{i}{1},cs{i}{2});
u_cond_bin_G(cs{i}{1},cs{i}{2}) = 1;
    %blue channel
u_cond_B_(cs{i}{1},cs{i}{2}) = u_B(cs{i}{1},cs{i}{2});
u_cond_bin_B(cs{i}{1},cs{i}{2}) = 1;
end
% 
% figure;
% imshow(u_cond_R_,[0, 255]), title('red conditionnal set ');
% figure;
% imshow(u_cond_G_,[0, 255]), title('green conditionnal set ');
% figure;
% imshow(u_cond_B_,[0, 255]), title('blue conditionnal set ');
u_cond_full=zeros(M,N,3);u_cond_full(:,:,1)=u_cond_R_; u_cond_full(:,:,2)=u_cond_G_;u_cond_full(:,:,3)=u_cond_B_;
figure;
imshow(uint8(u_cond_full),[]), title('Conditionnal set all channels');


%% spot & ADSN

spot_mean_R = sum(spot_R(:))./(2*dP)^2; 
spot_mean_G = sum(spot_G(:))./(2*dP)^2; 
spot_mean_B = sum(spot_B(:))./(2*dP)^2; 

norm_spot_R = (spot_R - spot_mean_R)./(2*dP);
norm_spot_G = (spot_G - spot_mean_G)./(2*dP);
norm_spot_B = (spot_B - spot_mean_B)./(2*dP);

t_v_R = zeros(M,N); t_v_G = zeros(M,N); t_v_B = zeros(M,N);
t_v_R(wx,wy)=norm_spot_R; t_v_G(wx,wy)=norm_spot_G; t_v_B(wx,wy)=norm_spot_B;

% figure;
% imshow(t_v_R,[0 255]), title('t_v Red ');
G_noise = normrnd(0,1,[M,N]); % bruit blanc gaussien
ADSN_R = conv_im(t_v_R,G_noise);   % asymptotic discret spot noise ADSN for red channel
ADSN_G = conv_im(t_v_G,G_noise);   % asymptotic discret spot noise ADSN for green channel
ADSN_B = conv_im(t_v_B,G_noise);   % asymptotic discret spot noise ADSN for blue channel

% figure;
% imshow(ADSN_R,[]), title('ADSN Red ');
% figure;
% imshow(ADSN_G,[]), title('ADSN green ');
% figure;
% imshow(ADSN_B,[]), title('ADSN Blue ');
ADSN_show=zeros(M,N,3); ADSN_show(:,:,1)=ADSN_R+spot_mean_R; ADSN_show(:,:,2)=ADSN_G+spot_mean_G; ADSN_show(:,:,3)=ADSN_B+spot_mean_B;
figure;
imshow(uint8(ADSN_show),[]), title('ADSN sample'); 

ADSN_R_rest_=zeros(M,N); ADSN_G_rest_=zeros(M,N); ADSN_B_rest_=zeros(M,N);
ADSN_R_rest=ADSN_R(xmin:xmax,ymin:ymax)+spot_mean_R; ADSN_G_rest=ADSN_G(xmin:xmax,ymin:ymax)+spot_mean_G; ADSN_B_rest=ADSN_B(xmin:xmax,ymin:ymax)+spot_mean_B;
ADSN_R_rest_(xmin:xmax,ymin:ymax)=ADSN_R_rest; ADSN_G_rest_(xmin:xmax,ymin:ymax)=ADSN_G_rest; ADSN_B_rest_(xmin:xmax,ymin:ymax)=ADSN_B_rest;
u_R_filled=u_R_masked+ADSN_R_rest_;
u_G_filled=u_G_masked+ADSN_G_rest_;
u_B_filled=u_B_masked+ADSN_B_rest_;

% figure;
% imshow(u_R_filled,[0 255]), title('red channel filled by ADSN'); 
% % figure;
% % imshow(u_G_filled,[0 255]), title('green channel filled by ADSN'); 
% % figure;
% % imshow(u_B_filled,[0 255]), title('blue channel filled by ADSN'); 

%% covariance matrix for observed variables

c_v_R = real(ifft2(abs(fft2(t_v_R)).^2));
c_v_G = real(ifft2(abs(fft2(t_v_G)).^2));
c_v_B = real(ifft2(abs(fft2(t_v_B)).^2));

M_cond=xmax+delta-(xmin-delta)+1;
N_cond=ymax+delta-(ymin-delta)+1;

cardl=[N_cond*ones(delta,1);2*delta*ones(xmax-xmin+1,1);N_cond*ones(delta,1)]; %number of variables per layer/row of the domain
cardObs = sum(cardl); %number of variables yielding the information (Observable variables)
fprintf('card of conditionnal set: %d / wrt image: %3.1f%% \n\n', cardObs, 100*cardObs/(M*N));

gam_cond_R = zeros(cardObs,cardObs); % space : cardObs^2 !!!
v_1_R = zeros(1,cardObs); v_2_R = zeros(1,cardObs);
v_1_G = zeros(1,cardObs); v_2_G = zeros(1,cardObs);
v_1_B = zeros(1,cardObs); v_2_B = zeros(1,cardObs);

z=1; % 
for k=1:K
    for i=cs{k}{1}
        for j=cs{1}{2}
            v_1_R(z) = c_v_R(i-cs{1}{1}(1)+1,j-cs{1}{2}(1)+1);
            v_1_G(z) = c_v_G(i-cs{1}{1}(1)+1,j-cs{1}{2}(1)+1);
            v_1_B(z) = c_v_B(i-cs{1}{1}(1)+1,j-cs{1}{2}(1)+1);
            if j-cs{1}{2}(end)==0
                v_2_R(z) = c_v_R(i-cs{1}{1}(1)+1,j-cs{1}{2}(end)+1);
                v_2_G(z) = c_v_G(i-cs{1}{1}(1)+1,j-cs{1}{2}(end)+1);
                v_2_B(z) = c_v_B(i-cs{1}{1}(1)+1,j-cs{1}{2}(end)+1);
            else 
                v_2_R(z) = c_v_R(i-cs{1}{1}(1)+1,j-cs{1}{2}(end)+1+N);
                v_2_G(z) = c_v_G(i-cs{1}{1}(1)+1,j-cs{1}{2}(end)+1+N);
                v_2_B(z) = c_v_B(i-cs{1}{1}(1)+1,j-cs{1}{2}(end)+1+N);
            end
            z=z+1;
        end
    end
end 

for i=1:M_cond                 % time : M_cond^2
    x_i=sum(cardl(1:i-1));
    for j=1:i-1
        y_j=sum(cardl(1:j-1));
        gam_cond_R(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = zeros(cardl(i),cardl(j));
        gam_cond_G(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = zeros(cardl(i),cardl(j));
        gam_cond_B(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = zeros(cardl(i),cardl(j));
    end
    for j=i:M_cond
        y_j=sum(cardl(1:j-1));
        bin1_R = u_cond_bin_R(cs{1}{1}(1)+i-1,cs{1}{2}(1):cs{K}{2}(end) );
        bin2_R = u_cond_bin_R(cs{1}{1}(1)+j-1,cs{1}{2}(1):cs{K}{2}(end) );
        
        bin1_G = u_cond_bin_G(cs{1}{1}(1)+i-1,cs{1}{2}(1):cs{K}{2}(end) );
        bin2_G = u_cond_bin_G(cs{1}{1}(1)+j-1,cs{1}{2}(1):cs{K}{2}(end) );
        
        bin1_B = u_cond_bin_B(cs{1}{1}(1)+i-1,cs{1}{2}(1):cs{K}{2}(end) );
        bin2_B = u_cond_bin_B(cs{1}{1}(1)+j-1,cs{1}{2}(1):cs{K}{2}(end) );
        
        z_ij=(j-i)*N_cond;
        
        M_temp1_R = upperTriu(v_1_R(z_ij+1:z_ij+N_cond));
        M_temp1_G = upperTriu(v_1_G(z_ij+1:z_ij+N_cond));
        M_temp1_B = upperTriu(v_1_B(z_ij+1:z_ij+N_cond));
        
        gam_cond_R(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = M_temp1_R(logical(bin1_R),logical(bin2_R));
        gam_cond_G(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = M_temp1_G(logical(bin1_G),logical(bin2_G));
        gam_cond_B(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = M_temp1_B(logical(bin1_B),logical(bin2_B));
        if j>i
            M_temp2_R = lowerTriu(v_2_R(z_ij+1:z_ij+N_cond));
            M_temp2_G = lowerTriu(v_2_G(z_ij+1:z_ij+N_cond));    
            M_temp2_B = lowerTriu(v_2_B(z_ij+1:z_ij+N_cond));   
            
            gam_cond_R(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = ...
            gam_cond_R(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) + M_temp2_R(logical(bin1_R),logical(bin2_R));
        
            gam_cond_G(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = ...
            gam_cond_G(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) + M_temp2_G(logical(bin1_G),logical(bin2_G));
        
            gam_cond_B(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) = ...
            gam_cond_B(x_i+1:x_i+cardl(i),y_j+1:y_j+cardl(j)) + M_temp2_B(logical(bin1_B),logical(bin2_B));
        end
    end
end

gam_cond_R = gam_cond_R' + gam_cond_R - diag(diag(gam_cond_R));

gam_cond_G = gam_cond_G' + gam_cond_G - diag(diag(gam_cond_G));

gam_cond_B = gam_cond_B' + gam_cond_B - diag(diag(gam_cond_B));


%% kriging component

u_cond_R = zeros(cardObs,1);
u_cond_G = zeros(cardObs,1);
u_cond_B = zeros(cardObs,1);

ADSN_cond_R = zeros(cardObs,1);
ADSN_cond_G = zeros(cardObs,1);
ADSN_cond_B = zeros(cardObs,1);
z=1;
for k =1:K
      for i = cs{k}{1}
          for j = cs{k}{2} 
              u_cond_R(z)=u_R(i,j);
              ADSN_cond_R(z) = ADSN_R(i,j);
              
              u_cond_G(z)=u_G(i,j);
              ADSN_cond_G(z) = ADSN_G(i,j);
              
              u_cond_B(z)=u_B(i,j);
              ADSN_cond_B(z) = ADSN_B(i,j);
              
              z=z+1;
          end 
      end       
end

phi_1_R = gam_cond_R\(u_cond_R-spot_mean_R); %% mldivide algo is supposed to use Cholesky solver if the matrix is hermitian
phi_1_G = gam_cond_G\(u_cond_G-spot_mean_G); %% solver if the matrix is hermitian
phi_1_B = gam_cond_B\(u_cond_B-spot_mean_B); %% but it appears to much faster than using 'Chol'


phi_2_R = gam_cond_R\ADSN_cond_R;
phi_2_G = gam_cond_G\ADSN_cond_G;
phi_2_B = gam_cond_B\ADSN_cond_B;


phi_1_pad_R = zeros(M,N);
phi_2_pad_R = zeros(M,N);

phi_1_pad_G = zeros(M,N);
phi_2_pad_G = zeros(M,N);

phi_1_pad_B = zeros(M,N);
phi_2_pad_B = zeros(M,N);

z=1;
for k =1:K
      for i = cs{k}{1}
          for j = cs{k}{2} 
              phi_1_pad_R(i,j) = phi_1_R(z);
              phi_2_pad_R(i,j) = phi_2_R(z);
              
              phi_1_pad_G(i,j) = phi_1_G(z);
              phi_2_pad_G(i,j) = phi_2_G(z);
              
              phi_1_pad_B(i,j) = phi_1_B(z);
              phi_2_pad_B(i,j) = phi_2_B(z);
              
              z=z+1;
          end 
      end       
end


krig_1_R_ = conv_im(c_v_R,phi_1_pad_R);
krig_1_G_ = conv_im(c_v_G,phi_1_pad_G);
krig_1_B_ = conv_im(c_v_B,phi_1_pad_B);

krig_1_R=krig_1_R_(xmin:xmax,ymin:ymax);
krig_1_G=krig_1_G_(xmin:xmax,ymin:ymax);
krig_1_B=krig_1_B_(xmin:xmax,ymin:ymax);

krig_2_R = conv_im(c_v_R,phi_2_pad_R);
krig_2_G = conv_im(c_v_G,phi_2_pad_G);
krig_2_B = conv_im(c_v_B,phi_2_pad_B);

inno_compo_R_ = ADSN_R - krig_2_R;
inno_compo_R=inno_compo_R_(xmin:xmax,ymin:ymax);

inno_compo_G_ = ADSN_G - krig_2_G;
inno_compo_G=inno_compo_G_(xmin:xmax,ymin:ymax);

inno_compo_B_ = ADSN_B - krig_2_B;
inno_compo_B=inno_compo_B_(xmin:xmax,ymin:ymax);

krig_show=zeros(mask_M,mask_N,3); 
krig_show(:,:,1)=krig_1_R+spot_mean_R; krig_show(:,:,2)=krig_1_G+spot_mean_G; krig_show(:,:,3)=krig_1_B+spot_mean_B;
inno_show=zeros(mask_M,mask_N,3); 
inno_show(:,:,1)=inno_compo_R+spot_mean_R; inno_show(:,:,2)=inno_compo_G+spot_mean_G; inno_show(:,:,3)=inno_compo_B+spot_mean_B;

figure;
imshow(uint8(krig_show),[]), title('kriging component all channels');
% figure;
% imshow(krig_1_R,[]), title('kriging component for red channel');
% figure;
% imshow(inno_compo_R,[]), title('innovation component  for red channel');
figure;
imshow(uint8(inno_show),[]), title('innovation component for channels');

% figure;
% imshow(krig_1_G,[]), title('kriging component  for green channel');
% figure;
% imshow(inno_compo_G,[]), title('innovation component  for green channel');
% 
% figure;
% imshow(krig_1_B,[]), title('kriging component  for blue channel');
% figure;
% imshow(inno_compo_B,[]), title('innovation component  for blue channel');
% 

output_1 = zeros(M,N,3); output_2 = zeros(M,N,3);
output_1(:,:,1) = u_cond_R_;  output_2(:,:,1) = u_R_masked;
output_1(:,:,2) = u_cond_G_;  output_2(:,:,2) = u_G_masked;
output_1(:,:,3) = u_cond_B_;  output_2(:,:,3) = u_B_masked;

m_temp_R=krig_1_R ;%+ inno_compo_R;
m_temp_G=krig_1_G ;%+ inno_compo_G;
m_temp_B=krig_1_B ;%+ inno_compo_B;

output_1(xmin:xmax,ymin:ymax,1) =  m_temp_R + spot_mean_R; 
output_2(xmin:xmax,ymin:ymax,1) =  m_temp_R + spot_mean_R; 

fprintf('  ----- error ------ \n');
true_R=u_R(xmin:xmax,ymin:ymax); mean_true_R=sum(true_R(:))/card_mask;
err_R = sum(sum(abs(uint8(output_1(xmin:xmax,ymin:ymax,1)) - uint8(true_R)))) / card_mask;
fprintf('mean error l1 for red channel on mask : %f\n', err_R);
fprintf('relative mean error l1 for red channel on mask : %3.1f%% \n\n', 100*err_R/mean_true_R);

output_1(xmin:xmax,ymin:ymax,2) =  m_temp_G + spot_mean_G; 
output_2(xmin:xmax,ymin:ymax,2) =  m_temp_G + spot_mean_G; 

true_G=u_G(xmin:xmax,ymin:ymax); mean_true_G=sum(true_G(:))/card_mask;
err_G = sum(sum(abs(uint8(output_1(xmin:xmax,ymin:ymax,2)) - uint8(true_G)))) / card_mask;
fprintf('mean error l1 for green channel on mask : %f\n', err_G);
fprintf('relative mean error l1 for green channel on mask : %3.1f%% \n\n', 100*err_G/mean_true_G);

output_1(xmin:xmax,ymin:ymax,3) =  m_temp_B + spot_mean_B; 
output_2(xmin:xmax,ymin:ymax,3) =  m_temp_B + spot_mean_B; 

true_B=u_B(xmin:xmax,ymin:ymax); mean_true_B=sum(true_B(:))/card_mask;
err_B = sum(sum(abs(uint8(output_1(xmin:xmax,ymin:ymax,3)) - uint8(true_B)))) / card_mask;
fprintf('mean error l1 for blue channel on mask : %f\n', err_B);
fprintf('relative mean error l1 for blue channel on mask : %3.1f%% \n\n', 100*err_B/mean_true_B);

figure;
imshow(uint8(output_1),[]), title('output: mask filled + conditionnal set');

figure;
imshow(uint8(output_2),[]), title('output: inpainted image');

%% end



