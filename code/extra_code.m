close all;
clc
clear
%% permute mod and phasis on structured/texture image

im_name_1='venice.jpg';
im_name_2='water.bmp';
u_1 = double(sum(imread(im_name_1),3)./3); 
u_2 = double(sum(imread(im_name_2),3)./3);
[M_1,N_1,NC_1]=size(u_1) 
[M_2,N_2,NC_2]=size(u_2)
u_2=u_2(1:M_1,:,:);
u_2=u_2(:,1:N_1,:);
figure;
imshow(u_1,[]), title('venice');
figure;
imshow(u_2,[]), title('water');



[uu_1,uu_2]=perm_phase(u_1,u_2);
figure;
imshow(uu_1,[]), title('phase: water modulus: venice');
figure;
imshow(uu_2,[]), title('phase: venice modulus: water');

diff_1 = sum(sum((u_1-uu_1).^2))
diff_2 = sum(sum((u_2-uu_2).^2))

%% Discret spot noise

close all;
clc
clear

im_name_1='paper.jpg';
u_1=chgtcaff([0 255],sum(double(imread(im_name_1)),3)./3);

[M,N,nc]=size(u_1)
figure;
imshow(uint8(u_1),[]), title(im_name_1);

m=sum(u_1(:))/(M*N);

u_1_=u_1-m;

wx=100:200; wy=200:300;
spot=zeros(M,N,nc);
spot(wx,wy,:)=u_1_(wx,wy,:);

figure;
imshow(spot,[]), title('paper spot');

n=200;
sptn=dsptn(spot,n);
figure;
imshow(chgtcaff([0 255],sptn),[]), title('paper adsn');
%% statistics of order 1 and 2: can we discriminate?

%1st order
close all;
clc
clear

im_name_1='paper.jpg';
im = imread(im_name_1);
im_R= im(:,:,1); im_G= im(:,:,2);  im_B= im(:,:,3); 
nf = 34; feature_1_R = imhist(im_R, nf); %nf quantization parameter
nf = 33; feature_1_G = imhist(im_G, nf); %nf quantization parameter
nf = 33; feature_1_B = imhist(im_B, nf); %nf quantization parameter
figure; subplot(2,2,1); imshow(im,[]), title('rounded floor 1');
subplot(2,2,2), bar(feature_1_R), axis square, title('histogram for red'); 
subplot(2,2,3), bar(feature_1_G), axis square, title('histogram for green');
subplot(2,2,4), bar(feature_1_B), axis square, title('histo for blue');

im_name_1='floor_2.jpg';
im = imread(im_name_1);
im_R= im(:,:,1); im_G= im(:,:,2);  im_B= im(:,:,3); 
nf = 34; feature_2_R = imhist(im_R, nf); %nf quantization parameter
nf = 33; feature_2_G = imhist(im_G, nf); %nf quantization parameter
nf = 33; feature_2_B = imhist(im_B, nf); %nf quantization parameter
figure; subplot(2,2,1); imshow(im,[]), title('rounded floor 2');
subplot(2,2,2), bar(feature_2_R), axis square, title('histogram for red'); 
subplot(2,2,3), bar(feature_2_G), axis square, title('histogram for green');
subplot(2,2,4), bar(feature_2_B), axis square, title('histo for blue');

dist = sqrt( sum((feature_1_R-feature_2_R).^2) ...
    + sum((feature_1_G-feature_2_G).^2) ...
    + sum((feature_1_B-feature_2_B).^2) ) / 100;

fprintf('dist 1st order = %3.3f \n\n', dist);

%% second order
close all;
clc
clear

im_name_1='floor_1.jpg';
im = imread(im_name_1);
im_R= im(:,:,1); im_G= im(:,:,2);  im_B= im(:,:,3);

  glcm = graycomatrix(im(:,:,1));
  temp = graycoprops(glcm);
  featureVec(1) = temp.Contrast;
  featureVec(2) = temp.Correlation;
  featureVec(3) = temp.Energy;
  featureVec(4) = temp.Homogeneity;

   offsets = [0 1; -3 3;30 -10;0 5];
   [glcms, SI] = graycomatrix(im_R,'NumLevels',50,'Offset',offsets);
   %imagesc(glcms(:,:,1))
figure; subplot(2,2,1); imagesc(glcms(:,:,1)), title('coo matrix 1');
subplot(2,2,2), imagesc(glcms(:,:,2)), axis square, title('coo matrix 2'); 
subplot(2,2,3), imagesc(glcms(:,:,3)), axis square, title('coo matrix 3');
subplot(2,2,4),imagesc(glcms(:,:,4)), axis square, title('coo matrix 4');

im_name_1='floor_2.jpg';
im = imread(im_name_1);
im_R= im(:,:,1); im_G= im(:,:,2);  im_B= im(:,:,3);

  glcm = graycomatrix(im(:,:,1));
  temp = graycoprops(glcm);
  featureVec(1) = temp.Contrast;
  featureVec(2) = temp.Correlation;
  featureVec(3) = temp.Energy;
  featureVec(4) = temp.Homogeneity;

   offsets = [0 1; -3 3;30 -10;0 5];
   [glcms, SI] = graycomatrix(im_R,'NumLevels',50,'Offset',offsets);
   %imagesc(glcms(:,:,1))
figure; subplot(2,2,1); imagesc(glcms(:,:,1)), title('coo matrix 1');
subplot(2,2,2), imagesc(glcms(:,:,2)), axis square, title('coo matrix 2'); 
subplot(2,2,3), imagesc(glcms(:,:,3)), axis square, title('coo matrix 3');
subplot(2,2,4),imagesc(glcms(:,:,4)), axis square, title('coo matrix 4');

%%






