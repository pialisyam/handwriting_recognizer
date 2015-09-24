%---------------------------------@psyam----------------------------------------%
%-------------------------- improved model of Radial Basis function network with Gaussian kernels is implemented-------------%



function [weights,Centroids]=trainingdata(handwritingData)
addpath('C:\Users\munai\Desktop\UCR class files\Fall 14 courses\machine learning\assignments');
%handwritingData=load ('handwriting.data');
%classified=0;
classifiedtrain=0;
crossvalitrain=0;
crossvalitest=0;
avgtest=0;
kk=0;
kkk=0;
fileID = fopen('exp.txt','wt');
fprintf(fileID,'%6s %12s\n\n\n\t\t\t\t\t','ActualData','Classlabels');
    for k=1:1
        m=size(handwritingData,1); % input data set
        
        M_part = floor(m/11);
        for n= 1:11                % cross-validation
            test_ind = 1 + M_part*(n-1) : M_part*n;
            
            test_data_ind =  index(test_ind);
            train_data_ind= setdiff(index,test_data_ind);
            TestData=(handwritingData(test_data_ind,:));
            TrainData=(handwritingData(train_data_ind,:));
            TrainDatanew = [TrainData ones( size (TrainData, 1),1)];%adding bias at the end
            x_traindata= TrainDatanew(:,2:end);
            TestDatanew = [TestData ones( size (TestData, 1),1)];
            num=size(TestData,1);
            xTestDatanew=TestDatanew(:,2:end);
            CC=10;
            num11=size(TrainData,1);
            [K,Centroids]= radialfunc_kernel_trainer(x_traindata); % creating gaussian kernels with train data
            %training data of this fold 
            %model = cell(26,1);
            for mulclass=0:25         % creating 26 classifiers (a-z letters)
                for q = 1:size(TrainData,1)
                    if TrainData(q,1)== mulclass
                        y_val(q,1)=1;
                    else
                        y_val(q,1)=0;
                    end    
                end
                %training the data for this class
%                 w(1:130,mulclass+1) = zeros(size(x_traindata,2),1);
%                 m=0;
%                 while(true)
%                     oldw = w(1:130,mulclass+1);
%                     p = exp (x_traindata*w(1:130,mulclass+1)) ;
%                     p = p./( 1+p ) ;
%                     q =(p.*(1-p )) ;
%                     for g=1:size(x_traindata,1)
%                         for h=1:size(x_traindata,2)
%                             U(g,h) = q(g,1).*(x_traindata(g,h));
%                         end
%                     end
%                     %z = x_traindata*w(1:130,mulclass+1) + U\( y_val-p ) ;
%                     %w(1:130,mulclass+1) = (x_traindata' * U * x_traindata)\(x_traindata' * U * z ) ;
%                     w(1:130,mulclass+1) = w(1:130,mulclass+1)+(x_traindata' * U)\(x_traindata' * ( y_val-p )) ;
%                     if (sum((oldw-w(1:130,mulclass+1)).*(oldw-w(1:130,mulclass+1)))<1e-2) | m>30
%                         break ;
%                     end 
%                     m=m+1;
%                 end


%----------------------------------------radial basis function network method----------------------%
                weights(:,mulclass+1)=  pinv(K)*y_val; 
            end

            
            %tesing with whole training part of fold
            
%             for lll=1:size(x_traindata,1)
% %                 for ll=1:size(w,2)
% %                     pp=  (xTrainDatanew(lll,:)*w(:,ll)) ;
% %                     fxx(1,ll)= 1/(1+ exp(-pp));
% %                 end 
% 
%                 for ll=1:size(weights,2)
%                     %fxx(1,ll)=K' *alpha(:,ll) + b(1,ll);
%                     fxx(1,ll)= (K(lll,:) * weights(:,ll));
%                 end 
% 
%                [B,I] = sort(fxx,'descend');
%                   clslabell = I(1,1);
%                   clslabell= clslabell - 1; %as 1 has been added to the class labels during matrix creation so now getting actual label by subtracting that 1
%                    if clslabell == TrainData(lll,1)
%                      classifiedtrain=classifiedtrain+1;
% %                    else
% %                      kk=kk+1;  
% %                      %predictlabtrain(1:130,kk)= w(:,I(1,1));
% %                      actuallabtrain(1,kk)= TrainData(lll,1);
% %                      aaa(1,kk)= clslabell;
%                    end
%             end
%             A = [TrainData(:,1); C];
%             fprintf(fileID,'%6s\n\n\n\t\t\t\t\t','--------------------------Training part misclassifications--------------------------------------------');
%             fprintf(fileID,'%6.2f %12.8f\n\n\n\n\t\t',A);
            
            
            %-------------------------------------tesing with testing---- part of fold
            
%             t=size(weights)
            sigma_square=1;
            XXX = sum(xTestDatanew.^2, 2);
            XCg = xTestDatanew * Centroids';
            CCg = sum(Centroids.^2, 2)';
            eud = sqrt(bsxfun(@plus, CCg, bsxfun(@minus, XXX, 2*XCg)));
            Ktest = exp(-eud/(2 * sigma_square));
            r=size(Ktest,1)
            classified=0;
            for l=1:size(xTestDatanew,1)
%                 for ll=1:size(w,2)
%                     pp=  (xTestDatanew(l,:)*w(:,ll)) ;
%                     fx(1,ll)= 1/(1+ exp(-pp));
%                 end 

                for ll=1:size(weights,2)
                    fx(1,ll)= (Ktest(l,:) * weights(:,ll));
                end 

                [B,I] = sort(fx,'descend');
                  clslabel = I(1,1);
                  clslabel= clslabel - 1; %as 1 has been added to the class labels during matrix creation so now getting actual label by subtracting that 1
                   if clslabel == TestData(l,1)
                     classified=classified+1;
%                    else
%                      kkk=kkk+1;  
%                      %predictlabtest(1:130,kkk)= w(:,I(1,1));
%                      actuallabtest(1,kkk)= TestData(l,1);
%                      aa(1,kkk)= clslabel;
                   end     
            end
            avgtest=avgtest+(classified/r);
%             AA = [actuallabtest; aa];
%             fprintf(fileID,'%6s\n\n\n\t\t\t\t\t','--------------------------Testing part misclassifications--------------------------------------------');
%             fprintf(fileID,'%6.2f %12.8f\n\n\n\n\t\t',AA);
        end
    end 
    fclose(fileID);
    traincrossvalaccuracy=(classifiedtrain/(1*11*num11))*100
    testcrossvalaccuracy=(avgtest/(11))*100
    fileIDD = fopen('weights_project.txt','w');
    %fprintf(fileIDD,'%6s %12s\n\n\n\t\t\t\t\t','TrainCrossValAccuracy','TestCrossValAccuracy');
    fprintf(fileIDD,'%6.2f',weights);
   % fprintf(fileIDD,'%6.2f   %12.8f\n\n\n',traincrossvalaccuracy,testcrossvalaccuracy);
    fclose(fileIDD);
    save('RBFWeights.txt','weights','-ascii');
    save('Centroids.txt','Centroids','-ascii');
   
   
   % for displaying the pattern of each weights for each classifier 
%     for v=1:26%size(predictlabtest,2)/1000
%         figure;
%         meanval= sum(w(:,v));
%         ww(1:130,1)= w(:,v)/meanval;
%         imagesc(reshape(ww(2:129,1),[8 16])');
%         colormap(1.0 - gray);
%         axis equal;
%     end
     %disp(weights);
end
