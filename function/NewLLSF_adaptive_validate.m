
function [ BestParameter, BestResult ] = NewLLSF_adaptive_validate( train_data, train_target, oldOptmParameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tune the best parameters for LLSF by crossvalidation
% Input
%   - train_data            : n by d data matrix
%   - trian_target          : n by l lable matrix
%   - oldOptmParameter      : initilization parameter
%
% Output
%   - BestParameter         : a structral variable with searched paramters,
%                             ie. alpha, beta, and gamma
%   - BestResult            : best result on the training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    num_train             = size(train_data,1);
    randorder             = randperm(num_train);
    
    optmParameter         = oldOptmParameter;
    
    alpha_searchrange     = oldOptmParameter.alpha_searchrange;
    beta_searchrange      = oldOptmParameter.beta_searchrange;
    gamma_searchrange     = oldOptmParameter.gamma_searchrange;
    lamda_searchrange     = oldOptmParameter.lamda_searchrange;
    lamda2_searchrange     = oldOptmParameter.lamda2_searchrange;
    
    BestResult = zeros(15,1);
    num_cv = 5;
    index = 1;
    step = 2;
    result = zeros(4,9);
   % ratio = zeros(1,9);
    count = 0;
    %计算遍历所有变量范围所需要的次数（穷举）
    total = length(alpha_searchrange)*length(beta_searchrange)*length(gamma_searchrange)*length(lamda_searchrange)*length(lamda2_searchrange)/step;
    for i=1:length(alpha_searchrange) % alpha
        for j=1:length(beta_searchrange) % beta
            for k = 1:length(gamma_searchrange) % gamma
                for L = 1:length(lamda_searchrange) % lamda
                    for M =1:step:length(lamda2_searchrange)% lamda2
                        count = count+1;
                        fprintf('\n-   %d-th/%d: search parameter alpha and beta for NewLLSF, alpha = %f, beta = %f, gamma = %f,lamda = %f,lamda2 = %f \n',index, total, alpha_searchrange(i), beta_searchrange(j), gamma_searchrange(k),lamda_searchrange(L),lamda2_searchrange(M));
                        index = index + 1;
                        optmParameter.alpha   = alpha_searchrange(i); % label correlation
                        optmParameter.beta    = beta_searchrange(j);  % sparsity
                        optmParameter.gamma   = gamma_searchrange(k); % {0.01, 0.1}
                        optmParameter.lamda   = lamda_searchrange(L);
                        optmParameter.lamda2   = lamda2_searchrange(M);

                        optmParameter.maxIter           = 100;
                        optmParameter.minimumLossMargin = 0.01;
                        optmParameter.outputtempresult  = 0;
                        optmParameter.drawConvergence   = 0;
                        %确定参数，进行训练
                        Result = zeros(15,1);
                        for cv = 1:num_cv%五次交叉验证
                            %得到训练集与测试集
                            [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = NewgenerateCVSet( train_data,train_target',randorder,cv,num_cv);
                            %训练
                            [model_NewLLSF]  = NewLLSF( cv_train_data, cv_train_target,optmParameter);
                            [p,l] = size(model_NewLLSF);
                            %numOfFeatures = sum(model_NewLLSF~=0);
                            %ratio(1,count) = numOfFeatures/(p*l);

                            %计算预测结果
                            Outputs     = (cv_test_data*model_NewLLSF)';
                            %调整阈值
                            fscore                 = (cv_train_data*model_NewLLSF)';
                            [ tau, ~] = TuneThreshold( fscore, cv_train_target', 1, 2);%参数均为转置
                            Pre_Labels             = Predict(Outputs,tau);%参数为转置

                                 %Pre_Labels  = round(Outputs);
                                 %Pre_Labels  = (Pre_Labels >= 1);
                                % Pre_Labels  = double(Pre_Labels);

                            %得到评价指标
                            Result      = Result + EvaluationAll(Pre_Labels,Outputs,cv_test_target');
                        end
                        %得到五次的平均
                        Result = Result/num_cv;
                        %参数分析
                        result(1,count) =Result(5,1); result(2,count) =Result(12,1);
                        result(3,count) =Result(14,1);result(4,count) =Result(15,1);

                        if optmParameter.bQuiet == 0
                            PrintResults(Result)
                        end
                        r = IsBetterThanBefore(BestResult,Result);
                        if r == 1
                            BestResult = Result;
                            %PrintResults(Result);
                            BestParameter = optmParameter;
                        end
                    end 
                end 
            end
        end
    end
    disp(result);
end


function r = IsBetterThanBefore(Result,CurrentResult)
% 1 HammingLoss
% 2 ExampleBasedAPCCuracy
% 3 ExampleBasedPrecision
% 4 ExampleBasedRecall
% 5 ExampleBasedFmeasure
% 6 SubsetAPCCuracy
% 7 LabelBasedAPCCuracy
% 8 LabelBasedPrecision
% 9 LabelBasedRecall
% 10 LabelBasedFmeasure
% 11 MicroF1Measure
% 12 Average_Precision
% 13 OneError
% 14 RankingLoss
% 15 Coverage
% 
%  the combination of Accuracy, F1, Macro F1 and Micro F1. Of course, any evaluation metrics or the combination of them can be used.

    a = CurrentResult(2,1) + CurrentResult(5,1)  + CurrentResult(10,1) + CurrentResult(11,1);
    b = Result(2,1) + Result(5,1) + Result(10,1) + Result(11,1);
    
    %fprintf('CurrentResult=%f,BestResult=%f \n',a,b);
    if a > b
        r =1;
    else
        r = 0;
    end
end
