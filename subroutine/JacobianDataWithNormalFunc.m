function [ Jacobian , Loss ] =  JacobianDataWithNormalFunc(canonical_xyz, canonical_Normal, KnnIndex, DepthInput, NormalInput, Twist )
DataLen   = length( KnnIndex ) ; 
ParamLen  = length( Twist ) ; 

Jacobian  = zeros(DataLen, ParamLen); 
Loss      = zeros(DataLen, 1); 

for i = 1 : length( KnnIndex ) 
    pairInd = KnnIndex(i) - 1; 
    Trans = twist2Transform(Twist(pairInd*6+1:pairInd*6+6)) ; 
    WarpedVertex =  Trans * [canonical_xyz(pairInd+1,:),1]'; 
    WarpedNormal =  (Trans(1:3, 1:3) * canonical_Normal(pairInd+1,:)')';
    InputVertex  =  DepthInput(i,:); % 
    %% Normal =  NormalInput(i,:); % 
    Nabla_Normal2so3 = skew_matrix(WarpedNormal);
    Nabla_Normal2se3 = [Nabla_Normal2so3, zeros(3)];
    Jacobian(i, pairInd*6+1 : pairInd*6+6) = WarpedNormal * skew_matrix_ex(WarpedVertex) + (WarpedVertex - InputVertex')' * Nabla_Normal2se3; %
    Loss(i)  = WarpedNormal * (WarpedVertex - InputVertex'); %
end