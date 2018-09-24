cd('F:/Studies/DDP/Final Codes/Final Datasets/')
%% Class 1
X = csvread('class1.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', 'blue')
hold on

%% Class 2
X = csvread('class2.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', 'red')
hold on

%% Class 3
X = csvread('class3.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', 'cyan')
hold on

%% Class 4
X = csvread('class4.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', 'green')
hold on

%% Class 5
X = csvread('class5.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', 'magenta')
hold on

%% Class 6
X = csvread('class6.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', 'yellow')
hold on

%% Class 7
X = csvread('class7.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', 'black')
hold on

%% Class 8
X = csvread('class8.csv');
if(size(X,1) == 3)
    X = X';
end
k = convhulln(X);
trisurf(k,X(:,1),X(:,2),X(:,3), 'Edgecolor', 'none', 'facecolor', [1 0.4 0.6])
xlabel('\alpha_1','FontSize',20,'FontWeight','bold')
ylabel('\alpha_2','FontSize',20,'FontWeight','bold')
zlabel('\alpha_3','FontSize',20,'FontWeight','bold')
xt = get(gca, 'XTick');
set(gca, 'FontSize', 14,'fontweight','bold','Fontname','Cambria')
hold off
h_legend = legend('Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8');
set(h_legend,'FontSize',18);