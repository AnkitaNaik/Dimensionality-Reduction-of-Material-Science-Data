X_con = zeros(400,(2*63*63*63));
i = 0
%% Class 1
cd('F:/Studies/DDP/micros/10_10_10_1')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/10_10_10_1')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  size(P)
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end

%% Class 2
cd('F:/Studies/DDP/micros/15_15_15_2')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/15_15_15_2')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end

%% Class 3
cd('F:/Studies/DDP/micros/20_10_10_3')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/20_10_10_3')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end

%% Class 4
cd('F:/Studies/DDP/micros/20_20_10_4')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/20_20_10_4')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end

%% Class 5
cd('F:/Studies/DDP/micros/50_5_5_5')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/50_5_5_5')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end

%% Class 6
cd('F:/Studies/DDP/micros/20_5_2.5_6')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/20_5_2.5_6')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end



%% Class 7
cd('F:/Studies/DDP/micros/20_20_5_7')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/20_20_5_7')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end

%% Class 8
cd('F:/Studies/DDP/micros/30_10_5_8')
fnames = dir('*.in');
numfids = length(fnames);
vals = cell(1,numfids);

for K = 1:numfids
  i = i+1;
  i
  cd('F:/Studies/DDP/micros/30_10_5_8')
  fileID = fopen(fnames(K).name,'r');
  formatSpec = '%d';
  sizeA = [4 Inf];
  A = fscanf(fileID,formatSpec,sizeA);
  fclose(fileID);
  A = A';
  size(A)
  B=reshape(A(:,4),64,64,64);
  B = permute(B,[3, 1, 2]);
  cd('F:/Studies/DDP/Final Codes/MATLAB-Spatial-Correlation-Toolbox-3.1.0/ahmetcecen-MATLAB-Spatial-Correlation-Toolbox-e593286')
  P = TwoPoint('auto',32,'periodic',B);
  Q = TwoPoint('cross',32,'periodic',B,(floor((B-0.5)/2)*-1));
  X_con(i,1:(63*63*63)) = reshape(P,1,(63*63*63));
  X_con(i,((63*63*63)+1):(2*63*63*63)) = reshape(Q,1,(63*63*63));
end

%%
X_con = X_con';
filename = 'correlations_final.csv';
cd('F:/Studies/DDP/Final Codes/Final Datasets/')
csvwrite(filename,X_con);