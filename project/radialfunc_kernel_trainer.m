%-------------------------------- @psyam--------------------------------------------------------%
%----------------- creating the gaussian kernels by choosing centroids by unsupervised learning---------------------------%




function [K,Centroids]= radialfunc_kernel_trainer(X_Train)
CC=10; 
            sigma_square=25;
            Centroids=k_meansclustering(1500,X_Train);
            XXX = sum(X_Train.^2, 2);
            XCg = X_Train * Centroids';
            CCg = sum(Centroids.^2, 2)';
            eud = sqrt(bsxfun(@plus, CCg, bsxfun(@minus, XXX, 2*XCg)));
            K = exp(-eud/(2 * sigma_square));
            t=size(K)

           


function Centroids=k_meansclustering(noofcentres,X_Train)
       %initializing centers
            n=size(X_Train,1);
            index = randperm(n);
            C=(X_Train(index(1:noofcentres),:));%random centers
            flag=true;
            x=0;
            while(flag)
                %cluster assignmnet
                XX = sum(X_Train.^2, 2);
                XC = X_Train * C';
                CCC = sum(C.^2, 2)';
                dists = sqrt(bsxfun(@plus, CCC, bsxfun(@minus, XX, 2*XC)));
                [B,I] = sort(dists,2);
                Classifypoints= I(:,1);
                %center mean
                for centers=1:size(C,1)
                    meanmat=X_Train((Classifypoints == centers),:);%contains data cluster points of this class
                    if  isempty(meanmat)  
                       Centroids(centers,:)=C(centers,:); 
                    else 
                       Centroids(centers,:)=mean(meanmat);
                    end
                end
                %terminating cond.
                if Centroids == C 
                    flag=false;
                else
                    C=Centroids;
                end  
                x=x+1
                
            end    
            XX = sum(X_Train.^2, 2);
            XC = X_Train * Centroids';
            CCC = sum(Centroids.^2, 2)';
            dists = sqrt(bsxfun(@plus, CCC, bsxfun(@minus, XX, 2*XC)));
            [B,I] = sort(dists,2);
            Classifypoints= I(:,1);

             for centers=1:size(C,1)
               var=X_Train((Classifypoints == centers),1);
                  if isempty(var)
                    n=size(X_Train,1);
                    index = randperm(n);
                    Centroids(centers,:)=mean(X_Train(index(1:500),:));%re-initialize random centers
                  end         
             end    
end 
end
