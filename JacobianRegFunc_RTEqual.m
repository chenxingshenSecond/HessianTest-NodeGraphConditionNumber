function [Jacobian,Loss] = JacobianRegFunc_RTEqual(canonical_xyz,line2,Twist)

Jacobian = zeros( length(line2) * 2 * 2 * 3 , length(canonical_xyz) * 6);
Loss= zeros( length(line2) * 2 * 2 * 3 , 1);

for i = 1:length(line2)
    indexi = line2(i,1) - 1 ;
    indexj = line2(i,2) - 1 ;
    
    T1 = twist2Transform(Twist(indexi*6+1:indexi*6+6));
    T2 = twist2Transform(Twist(indexj*6+1:indexj*6+6));
    
    Matrixi =    eye(6) ;
    Matrixj =  - eye(6) ;
    
    Jacobian( i*12-11:i*12-6 , indexi*6+1:indexi*6+6 ) = Matrixi ;
    Jacobian( i*12-11:i*12-6 , indexj*6+1:indexj*6+6 ) = Matrixj ;
    
    Matrixi =   - eye(6) ;
    Matrixj =     eye(6) ;
    
    Jacobian( i*12-5:i*12-0 , indexi*6+1:indexi*6+6 ) = Matrixi ;
    Jacobian( i*12-5:i*12-0 , indexj*6+1:indexj*6+6 ) = Matrixj ;
    
    Loss(i*12-11 : i*12-6) = Twist(6 * indexi + 1 : 6* indexi+6) - Twist(6 * indexj + 1: 6* indexj+6);
    Loss(i*12-5  : i*12-0) = - (Twist(6 * indexi+ 1 : 6* indexi+6) - Twist(6 * indexj+ 1 : 6* indexj+6));
end
end