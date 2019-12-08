%
clear all ;
close all;
addpath(genpath(cd));
% [vertex , normal, texture , faces] = objread('cat-low.obj') ; %
% [vertex , normal, texture , faces] = objread('cat-low1.obj') ; %
[vertex , normal, texture , faces] = objread('Desktop.obj') ; %
% [vertex , normal, texture , faces] = objread('Desktop_Smpl.obj') ; %
for i = 1:length(normal)
    if(norm(normal(i,:))>1e-5)
        normal(i,:) = normal(i,:) / norm(normal(i,:));
   end
end
%%
VertexLen = length(vertex);
% line1 = [ faces(:,1:2);faces(:,2:3) ];
% line2 =  sort(line1,2);
% line2 =  unique(line2,'rows');
KIN = 8; 
line2 = zeros(KIN * VertexLen, 2);
for i = 0:VertexLen-1
    line2(KIN*i+1:KIN*i+KIN, 1) = i + 1;
    dist = sum( (repmat( vertex(i+1,:) , VertexLen, 1) - vertex).^2 , 2);
    [~, idx] = sort(dist); 
    line2 (KIN*i+1:KIN*i+KIN, 2) = idx(1+1:KIN+1); 
end
dist = sum((vertex(line2(:,1),:) - vertex(line2(:,2),:)).^2, 2);
Twist = zeros(VertexLen * 6, 1) ;
canonical_xyz = vertex;
canonical_normal = normal;
%%
% KnnIndex = randperm(VertexLen);
% KnnIndex = KnnIndex(1:floor(VertexLen/5));
% KnnIndexLen = length(KnnIndex);
figure;
%DepthInput = canonical_xyz(KnnIndex,:) + randn(KnnIndexLen,3);
%DepthInput = vertex(KnnIndex,:) + ones(KnnIndexLen,3);
% KnnIndex  = randperm(VertexLen);
% KnnIndex    = 1:VertexLen ; %% KnnIndex(1:floor(VertexLen/5)); 
% KnnIndex  = randperm(VertexLen);
KnnIndex  = 1:2:VertexLen; % 1:30;% ;
% KnnIndex  = [ 1         322         609          79         401         234         336         235         354          17         659         669           2          90         611         832         568         273 ...
%              477          69         947          45         887         228         608         883         936         417          92         593         162         407         328         400         574 ...
%              658         812          99         765         420         531         500         683         173         721         781         141         334         181         855         244         552 ...
%             739         100         146         184         410         988           5         415         814         225         898         166         935         904         991          25         983 ]; 
KnnIndex    = [ones(1,50), KnnIndex]; 
% KnnIndex    = KnnIndex(1:500);
KnnIndexLen = length(KnnIndex);
REG_SCALE  = 0.1;
Offset = 0.000001;
xyz = [-0.17999, 0.236278, -0.067966];
xrange = xyz(1);

for i = 1:1
    Rotation   = rodriguesVectorToMatrix([0.0, 0.0, 1]' * i);% + randn(3,1)*0.001);
    DepthInput = canonical_xyz (KnnIndex,:); %% * Rotation; % randn(KnnIndexLen,3) * 8 + 1 
    NormalInput = canonical_normal(KnnIndex,:); %% 
    len = sum(DepthInput(:,1)< xrange);
    tailIdx = DepthInput(:,1)< xrange;
    DepthInput(tailIdx, :) = (DepthInput(tailIdx, :) - repmat([xrange, xyz(2), xyz(3)], len, 1))  *  Rotation + repmat([xrange,  xyz(2), xyz(3)], len, 1);
    NormalInput(tailIdx, :)  = NormalInput(tailIdx, :) *  Rotation;
    
    DepthInputAll = canonical_xyz;%%  * Rotation; 
    len = sum(DepthInputAll(:,1)< xrange);
    DepthInputAll(DepthInputAll(:,1)< xrange, :) = (DepthInputAll(DepthInputAll(:,1)< xrange, :) - repmat([xrange, xyz(2), xyz(3)], len, 1))  *  Rotation + repmat([xrange, xyz(2), xyz(3)], len, 1);
    DepthInput(1:50, 1) =  DepthInput(1:50, 1) + Offset ; %* randn(1,1);%% 
    DepthInput(1:50, 2) =  DepthInput(1:50, 2) + Offset ; %* randn(1,1);%% 
    DepthInput(1:50, 3) =  DepthInput(1:50, 3) + Offset ; %* randn(1,1);%% 
    %% vertex(KnnIndex,:) = DepthInput ;
    for j=1:10
        [ JacobianData , LossData ] =  JacobianDataWithNormalFunc (canonical_xyz , canonical_normal, KnnIndex , DepthInput , NormalInput, Twist);
        %[ JacobianData , LossData ] =  JacobianDataFunc (canonical_xyz , KnnIndex , DepthInput , Twist);

        [ JacobianReg  , LossReg  ] =  JacobianRegFunc(canonical_xyz , line2    , Twist);
        
        JacobianReg =  JacobianReg * REG_SCALE;
        LossReg     =  LossReg * REG_SCALE;
        
        HessianReg  = JacobianReg' * JacobianReg;
        HessianData = JacobianData' * JacobianData;
        Hessian     = HessianReg + HessianData;
        
        Residual    = JacobianData' *  LossData + JacobianReg' * LossReg; %
        % delta_X     = - Hessian \ Residual; %
        %         if(j<8)
        %             preConditionFlag = 3 ;
        %             delta_X2 = - pcgFunc( Hessian , Residual,preConditionFlag) ;
        %             delta_X = delta_X2;
        %         else
        %             preConditionFlag = 2 ;
        %             delta_X2 = - pcgFunc( Hessian , Residual,preConditionFlag) ;
        %             delta_X = delta_X2;
        %         end

        preConditionFlag = 3;
        delta_X2 = - pcgFunc( Hessian , Residual, preConditionFlag);
        delta_X = delta_X2;
        
        %disp(['max(delta_X)  = ', num2str(max(abs(delta_X)))]) ; %
        Twist  = updateTwist( delta_X, Twist );
        vertex = updateVertex(canonical_xyz, Twist);
        %disp(['max(Twist)  = ', num2str(max(abs(Twist)))]) ; %
        %disp(['rank(Hessian) = ', num2str(rank(Hessian))]) ; % 
        disp(['LossData + LossReg = ', num2str(norm(LossData) + norm(LossReg))]) ; %

        clf;
        plot3( vertex(:,1), vertex(:,2) , vertex(:,3) , 'b*'); axis equal ; axis vis3d ; hold on; %
        %plot3( vertex(1,1), vertex(1,2) , vertex(1,3) , 'bs'); axis equal ; axis vis3d ; hold on; %
        plot3( DepthInput(:,1), DepthInput(:,2) , DepthInput(:,3) , 'r*'); axis equal ; axis vis3d ; hold on; % 
        plot3( DepthInputAll(:,1), DepthInputAll(:,2) , DepthInputAll(:,3) , 'r*'); axis equal ; axis vis3d ; hold on; % 
        view([0,0,1]);
        camup([0 1 0])
        % plot3( DepthInput(:,1), DepthInput(:,2) , DepthInput(:,3) , 'rs'); axis equal ; axis vis3d ; hold on; %
        drawnow ;
        view([0,0,1]); 
        camup([0 1 0])
    end
end

OBJwrite('data/OutSmpl.obj', vertex, faces)
%%
disp(['sum(sum(abs(Hessian-Hessian))) = ', num2str( sum (sum(abs(Hessian-Hessian'))) )]); %
disp(['cond(Hessian) = ', num2str(cond(Hessian))]) ; %
disp(['max(delta_X)  = ', num2str(max(abs(delta_X)))]) ; %
disp(['rank(Hessian) = ', num2str(rank(Hessian))]) ; %
%%
figure; plot3( vertex(:,1), vertex(:,2), vertex(:,3),'*'); axis equal ; axis vis3d ; hold on; %
plot3( DepthInput(:,1),DepthInput(:,2),DepthInput(:,3),'rs'); %
reshape( delta_X , 6 , length(delta_X)/6); %
reshape( Twist   , 6 , length(Twist)/6); %  