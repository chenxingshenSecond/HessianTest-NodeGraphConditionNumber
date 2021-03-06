function x = pcgFunc(A,b,preConditionFlag)
x = zeros(size(b)); %
if preConditionFlag==0  % conjugated gradients
    r = b - A * x;
    p = r ;
    rsold = r' * r ;
    for i = 1 : 5 
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-6
            break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
else
    if preConditionFlag==1
        Precoditioner = diag( 1./ diag(A) ) ; %
        r = b - A * x;
        z = Precoditioner * r ; 
        p = z ; 
        rsold = z' * r ;
        for i = 1:10
            Ap = A * p ;
            alpha = (r' * z) / (p' * Ap);
            x = x + alpha * p;
            r = r - alpha * Ap;
            z = Precoditioner * r ; 
            rsnew = z' * r;
            if sqrt(rsnew) < 1e-6
                break;
            end
            p = z + (rsnew / rsold) * p;
            rsold = rsnew;
        end
    else
        if preConditionFlag == 2
            % Precoditioner = diag( diag(A) ) ; %        Precoditioner = diag( diag(A) ) ; %
            Precoditioner = zeros(size(A));
            for i =1:size(A,1)/6
                Precoditioner(i*6-5:i*6,i*6-5:i*6) = inv(A(i*6-5:i*6,i*6-5:i*6));
            end
            r = b - A * x;
            z = Precoditioner * r ;
            p = z ;
            rsold = z' * r ;
            for i = 1:10
                Ap = A * p ;
                alpha = (r' * z) / (p' * Ap);
                x = x + alpha * p;
                r = r - alpha * Ap;
                z = Precoditioner * r ;
                rsnew = z' * r;
                if sqrt(rsnew) < 1e-2
                    break;
                end
                p = z + (rsnew / rsold) * p;
                rsold = rsnew;
                
            end 
        else
            % Precoditioner = diag( diag(A) ) ; %        Precoditioner = diag( diag(A) ) ; %
            block = 3;
            Precoditioner = zeros(size(A));
            for i = 0 : size(A,1)/block - 1
                Precoditioner(i*block+1:i*block+block,i*block+1:i*block+block) = inv(A(i*block+1:i*block+block,i*block+1:i*block+block));
            end
            r = b - A * x;
            z = Precoditioner * r ;
            p = z ;
            rsold = z' * r ;
            for i = 1:10
                Ap = A * p ;
                alpha = (r' * z) / (p' * Ap);
                x = x + alpha * p;
                r = r - alpha * Ap;
                z = Precoditioner * r ;
                rsnew = z' * r;
                if sqrt(rsnew) < 1e-6
                    break;
                end
                p = z + (rsnew / rsold) * p;
                rsold = rsnew;
                
            end 
        end
            
    end
end

end