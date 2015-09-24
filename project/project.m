% ----------------------------------   @psyam ---------------------------------------%
%---------------------only the best model of radial basis function network is used to test the test data-------%

function project()
prompt = 'Enter: 1 (for Testing) or 2 (for Traning and Testing)';
result = input(prompt);
if result==2
    h=load ('handwriting.data'); % copy and paste the 'TEST' file name and extension to be tested here
    e=h(1:1000,:);
    [weights,Centroids]=trainingdata(h((e+1:end),:));
end 
    
    if result==1
        Centroids=load ('Centroids.txt');
        weights=load ('RBFWeights.txt');
    end
    fileIDD = fopen('classlabels.txt','wt');
    fprintf(fileIDD,'%6s\n\n\t\t','Class Labels');
    if result==1
        testdata=load ('handwriting.data');% copy and paste the 'TEST' file name and extension to be tested here
    else    
        testdata=e;
    end    
    TestDatanew = [testdata ones( size (testdata, 1),1)];
    xTestDatanew=TestDatanew(:,2:end);
    sigma_square=25;
    XXX = sum(xTestDatanew.^2, 2);
    XCg = xTestDatanew * Centroids';
    CCg = sum(Centroids.^2, 2)';
    eud = sqrt(bsxfun(@plus, CCg, bsxfun(@minus, XXX, 2*XCg)));
    Ktest = exp(-eud/(2 * sigma_square));
    classified=0;
    
     for l=1:size(Ktest,1)

         if rem(l,50)== 0
             r=size(size(Ktest,1))
         end    
        for ll=1:size(weights,2)
            fx(1,ll)= (Ktest(l,:) * weights(:,ll));
        end 

        [B,I] = sort(fx,'descend');
        clslabel = I(1,1);
        clslabel= clslabel - 1; %as 1 has been added to the class labels during matrix creation so now getting actual label by subtracting that 1
        aa(1,l)= clslabel;
%         if clslabel == testdata(l,1)
%                      classified=classified+1;
%         end  
       
     end
%      acc=classified/size(Ktest,1)*100
    fprintf(fileIDD,'%6.2f\n',aa);
    fclose(fileIDD);
% end
end
