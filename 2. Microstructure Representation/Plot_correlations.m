x = 0:1:62;
y = 0:1:62;
%% Class 1
cd('F:/Studies/DDP/micros/20_20_5_7')
fileID = fopen('1.in','r');
formatSpec = '%d';
sizeA = [4 Inf];
A = fscanf(fileID,formatSpec,sizeA);
A = A';
B=reshape(A(:,4),64,64,64);
B = permute(B,[3, 1, 2]);
cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
P = TwoPoint('auto',32,'periodic',B);
Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));

% X - axis
z = Q(:,32,:);
z = reshape(z,63,63);
subplot(2,2,1);
%subaxes(4,6,1,subplot1, 0);
%subaxes(2, 2, 1, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
surf(x,y,z,'EdgeColor','none')
daspect([1 1 1])
colorbar
xt = get(gca, 'XTick');
set(gca, 'FontSize', 12)
axis tight
view(2)
title('Class1: Cross-correlations along X-axis')

% Y - axis
z = Q(:,:,32);
z = reshape(z,63,63);
subplot(2,2,2)
subaxes(2, 2, 2, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
surf(x,y,z,'EdgeColor','none')
daspect([1 1 1])
colorbar
xt = get(gca, 'XTick');
set(gca, 'FontSize', 12)
axis tight
view(2)
title('Class1: Cross-correlations along Y-axis')

% Z - axis
z = Q(32,:,:);
z = reshape(z,63,63);
subplot(2,2,3)
subaxes(2, 2, 3, 'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0);
surf(x,y,z,'EdgeColor','none')
daspect([1 1 1])
colorbar
xt = get(gca, 'XTick');
set(gca, 'FontSize', 12)
axis tight
view(2)
title('Class1: Cross-correlations along Z-axis')

%% Class 1
cd('F:/Studies/DDP/micros/10_10_10_1')

%% Class 2
cd('F:/Studies/DDP/micros/15_15_15_2')

%% Class 3
cd('F:/Studies/DDP/micros/20_10_10_3')

%% Class 4
cd('F:/Studies/DDP/micros/20_20_10_4')

%% Class 5
cd('F:/Studies/DDP/micros/50_5_5_5')

%% Class 6
cd('F:/Studies/DDP/micros/20_5_2.5_6')

%% Class 7
cd('F:/Studies/DDP/micros/20_20_5_7')

%% Class 8
cd('F:/Studies/DDP/micros/30_10_5_8')
