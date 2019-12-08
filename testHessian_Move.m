%
clear all ;
close all;
addpath(genpath(cd));
[vertex , normal, texture , faces] = objread('cat-low.obj') ; %
%%
VertexLen = length(vertex);
line1 = [ faces(:,1:2);faces(:,2:3) ];
line2 =  sort(line1,2);
line2 =  unique(line2,'rows');
Twist = zeros(VertexLen * 6, 1) ;
canonical_xyz = vertex;
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
KnnIndex  = [ 1 174    55   139   226     1    75    61   245   158   124    68    43   160     8   238    82     2    29    50    91   188   156   133 ...
              62   136   103    39    88   220   118   214     3   228   206    52    47    46   212   195   107    76   185   100   244   224    78....
               173   241   196   192]; 
KnnIndex    = [ones(1,50), KnnIndex]; 
KnnIndex    = KnnIndex(1:100);
KnnIndexLen = length(KnnIndex);
REG_SCALE  = 0.1;
Offset = 200;
for i = 1:1
    Rotation   = rodriguesVectorToMatrix([0.01,0.4,0.01]' * i );% + randn(3,1)*0.001);
    DepthInput = canonical_xyz (KnnIndex,:) * Rotation; % randn(KnnIndexLen,3) * 8 + 1 
    DepthInputAll = canonical_xyz * Rotation;
    DepthInput(1:50, 1) =  DepthInput(1:50, 1) + Offset ; %* randn(1,1);%% 
    DepthInput(1:50, 2) =  DepthInput(1:50, 2) + Offset ; %* randn(1,1);%% 
    DepthInput(1:50, 3) =  DepthInput(1:50, 3) + Offset ; %* randn(1,1);%% 
    %% vertex(KnnIndex,:) = DepthInput ;
    for j=1:10
        [ JacobianData , LossData ] =  JacobianDataFunc (canonical_xyz , KnnIndex , DepthInput , Twist);
        [ JacobianReg  , LossReg  ] =  JacobianRegFunc  (canonical_xyz , line2    , Twist);
        
        JacobianReg =  JacobianReg * REG_SCALE;
        LossReg     =  LossReg * REG_SCALE;
        
        HessianReg  = JacobianReg' * JacobianReg;
        HessianData = JacobianData' * JacobianData;
        Hessian     = HessianReg + HessianData;
        
        Residual    = JacobianData' *  LossData + JacobianReg' * LossReg; %
        delta_X     = - Hessian \ Residual; %
%         if(j<8)
%             preConditionFlag = 1 ;
%             delta_X2 = - pcgFunc( Hessian , Residual,preConditionFlag) ;
%             delta_X = delta_X2;
%         else             
%             preConditionFlag = 3 ;
%             delta_X2 = - pcgFunc( Hessian , Residual,preConditionFlag) ;
%             delta_X = delta_X2;
%         end
        preConditionFlag = 1;
        delta_X2 = - pcgFunc( Hessian , Residual,preConditionFlag) ;
        delta_X = delta_X2;
        
        
        %disp(['max(delta_X)  = ', num2str(max(abs(delta_X)))]) ; %
        Twist  = updateTwist( delta_X, Twist );
        vertex = updateVertex(canonical_xyz, Twist);
        %disp(['max(Twist)  = ', num2str(max(abs(Twist)))]) ; %
        %disp(['rank(Hessian) = ', num2str(rank(Hessian))]) ; % 
        disp(['LossData + LossReg = ', num2str(norm(LossData) + norm(LossReg))]) ; %

        clf
        plot3( vertex(:,1), vertex(:,2) , vertex(:,3) , 'b*'); axis equal ; axis vis3d ; hold on; %
        plot3( vertex(1,1), vertex(1,2) , vertex(1,3) , 'bs'); axis equal ; axis vis3d ; hold on; %
        plot3( DepthInput(:,1), DepthInput(:,2) , DepthInput(:,3) , 'r*'); axis equal ; axis vis3d ; hold on; % 
        plot3( DepthInputAll(:,1), DepthInputAll(:,2) , DepthInputAll(:,3) , 'r*'); axis equal ; axis vis3d ; hold on; % 
        view([0,0,1]);
        camup([0 1 0])
        %         plot3( DepthInput(:,1), DepthInput(:,2) , DepthInput(:,3) , 'rs'); axis equal ; axis vis3d ; hold on; %
        drawnow ;
        view([0,0,1]); 
        camup([0 1 0])
    end
end
%%
disp(['sum(sum(abs(Hessian-Hessian))) = ', num2str( sum (sum(abs(Hessian-Hessian'))) )]); %
disp(['cond(Hessian) = ', num2str(cond(Hessian))]) ; %
disp(['max(delta_X)  = ', num2str(max(abs(delta_X)))]) ; %
disp(['rank(Hessian) = ', num2str(rank(Hessian))]) ; %
%%
figure; plot3( vertex(:,1), vertex(:,2),vertex(:,3),'*'); axis equal ; axis vis3d ; hold on; %
plot3( DepthInput(:,1),DepthInput(:,2),DepthInput(:,3),'rs'); %
reshape( delta_X , 6 , length(delta_X)/6); %
reshape( Twist   , 6 , length(Twist)/6); %  
