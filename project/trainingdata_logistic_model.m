%---------------------------------@psyam-----------------------------------%
%--------------------------- simple model of binary logistic regression converted to multi-class using 1-vs-rest strategy------------------------%





function trainingdata_logistic_model()
handwritingData=load ('handwriting.data'); % raw data of handwritten letters as features
classified=0;
classifiedtrain=0;
kk=0;
kkk=0;
fileID = fopen('exp.txt','w');
fprintf(fileID,'%6s %12s\n\n\n\t\t\t\t\t','ActualData','Classlabels');
    for k=1:1
        m=size(handwritingData,1);
        index = randperm(m);
        M_part = floor(m/11);
        avgtrain=0;
        avgtest=0;
        for n= 1:11  % cross validation
            test_ind = 1 + M_part*(n-1) : M_part*n;
            test_data_ind =  index(test_ind);
            train_data_ind= setdiff(index,test_data_ind);
            TestData=(handwritingData(test_data_ind,:));
            TrainData=(handwritingData(train_data_ind,:));
            j=0;
            
            %training data of this fold 
            for mulclass=0:25
                j=0;
                jj=0;
                %xtrainpos=0;xtrainneg=0;
                for q = 1:size(TrainData,1)
                    if TrainData(q,1)== mulclass
                        j=j+1;
                        xtrainpos(j,1:129)= TrainData(q,2:end);
                        %cutout(j,1)=q;
                    else
                        jj=jj+1;
                        xtrainneg(jj,1:129)= TrainData(q,2:end);
                    end  
                end
                y_valpos=ones(size(xtrainpos,1),1);
%                 sss=size(xtrainpos)
%                 ss=size(y_valpos)
                index1 = randperm(size(xtrainneg,1));
                xtrainneg=xtrainneg(index1,:);
                if size(xtrainneg,1) > size(xtrainpos,1) | size(xtrainpos,1) < 2000
                    xtrainneg  = xtrainneg(1:size(xtrainpos,1),:);
                    y_valneg = zeros(size(xtrainneg,1),1);
                    xTrainDatanew = [xtrainpos;xtrainneg];
                    y_val = [y_valpos;y_valneg];
                else
                    y_valneg = zeros(size(xtrainneg,1),1);
                    xTrainDatanew = [xtrainpos;xtrainneg];
                    y_val = [y_valpos;y_valneg];
                end 
%                 sssj=size(xtrainneg)
%                 ssj=size(y_valneg)
%                 yvalsz=size(y_val)
                xTrainDatanewsz=size(xTrainDatanew)
                x_traindata = [xTrainDatanew ones( size (xTrainDatanew, 1),1)];%adding bias at the end
                w(1:130,mulclass+1) = zeros(size(x_traindata,2),1);
                m=0;
                while(true) % applying  logistic regression
                    
                    oldw = w(1:130,mulclass+1);
                    p = exp (x_traindata*w(1:130,mulclass+1)) ;
                    p = p./( 1+p ) ;
                    q =(p.*(1-p )) ;
                    for g=1:size(x_traindata,1)
                        for h=1:size(x_traindata,2)
                            U(g,h) = q(g,1).*(x_traindata(g,h));
                        end
                    end
                    %z = x_traindata*w(1:130,mulclass+1) + U\( y_val-p ) ;
                    %w(1:130,mulclass+1) = (x_traindata' * U * x_traindata)\(x_traindata' * U * z ) ;
                    w(1:130,mulclass+1) = w(1:130,mulclass+1)+(x_traindata' * U)\(x_traindata' * ( y_val-p )) ;
                    if (sum((oldw-w(1:130,mulclass+1)).*(oldw-w(1:130,mulclass+1)))<1e-2) || m>30
                        break ;
                    end 
                    m=m+1;
                end
            end
            
            
            %tesing with whole training part of fold
            TrainDatanew = [TrainData ones( size (TrainData, 1),1)];
            num1=size(TrainData,1);
            xTrainDatanew=TrainDatanew(:,2:end);
            for lll=1:size(xTrainDatanew,1)
                for ll=1:size(w,2)
                    pp=  (xTrainDatanew(lll,:)*w(:,ll)) ;
                    fxx(1,ll)= 1/(1+ exp(-pp));
                end 
                [B,I] = sort(fxx,'descend');
                  clslabell = I(1,1);
                  clslabell= clslabell - 1; %as 1 has been added to the class labels during matrix creation so now getting actual label by subtracting that 1
                   if clslabell == TrainData(lll,1)
                     classifiedtrain=classifiedtrain+1;
%                    else
%                      kk=kk+1;  
%                      %predictlabtrain(1:130,kk)= w(:,I(1,1));
%                      actuallabtrain(1,kk)= TrainData(lll,1);
%                      aaa(1,kk)= clslabell;
                   end
            end
            avgtrain=avgtrain+(classifiedtrain/num1);
%             A = [actuallabtrain; aaa];
%             fprintf(fileID,'%6s\n\n\n\t\t\t\t\t','--------------------------Training part misclassifications--------------------------------------------');
%             fprintf(fileID,'%6.2f %12.8f\n\n\n\n\t\t',A);
            %tesing with testing part of fold
            TestDatanew = [TestData ones( size (TestData, 1),1)];
            num=size(TestData,1);
            xTestDatanew=TestDatanew(:,2:end);
            t=size(w)
            for l=1:size(xTestDatanew,1)
                for ll=1:size(w,2)
                    pp=  (xTestDatanew(l,:)*w(:,ll)) ;
                    fx(1,ll)= 1/(1+ exp(-pp));
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
            avgtest=avgtest+(classified/num);
            AA = [actuallabtest; aa];
            fprintf(fileID,'%6s\n\n\n\t\t\t\t\t','--------------------------Testing part misclassifications--------------------------------------------');
            fprintf(fileID,'%6.2f %12.8f\n\n\n\n\t\t',AA);
              
        end
    end 
    fclose(fileID);
    traincrossvalaccuracy=(avgtrain/(1*11))*100
    testcrossvalaccuracy=(avgtest/(1*11))*100
    fileIDD = fopen('accuracy_project.txt','w');
    fprintf(fileIDD,'%6s %12s\n\n\n\t\t\t\t\t','TrainCrossValAccuracy','TestCrossValAccuracy');
    fprintf(fileIDD,'%6.2f   %12.8f\n\n\n',traincrossvalaccuracy,testcrossvalaccuracy);
    fclose(fileIDD);
    
%     for v=1:26%size(predictlabtest,2)/1000
%         figure;
%         meanval= sum(w(:,v));
%         ww(1:130,1)= w(:,v)/meanval;
%         imagesc(reshape(ww(2:129,1),[8 16])');
%         colormap(1.0 - gray);
%         axis equal;
%     end
    disp(w');
end                        
