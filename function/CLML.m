
function [model_NewLLSF] = CLML( X, Y, optmParameter)
   %% optimization parameters
    alpha            = optmParameter.alpha;
    beta             = optmParameter.beta;
    gamma            = optmParameter.gamma;
    lamda            = optmParameter.lamda;
    lamda2            = optmParameter.lamda2;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;
   %% initializtion
    num_dim = size(X,2);
    XT = X';
    XTX = X'*X;
    XTY = X'*Y;
    W_s   = (XTX + gamma*eye(num_dim)) \ (XTY);
    W_s_1 = W_s;
    %% label correlation
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );%R 得到的只是距离，距离与相似度是成反比的
    C = 1 - R;%用1—R得到的就是相似度矩阵
    L1 = diag(sum(C,2)) - C;%计算拉普拉斯矩阵
    %% instance correlation
    S = ins_similarity(X,10);
    L2 = diag(sum(S,2)) - S;
    
    iter    = 1;
    oldloss = 0;
    bk = 1; %bt
    bk_1 = 1; %bt-1
    %% 计算LIP
    A = gradL21(W_s);
    Lip = sqrt(4*(norm(XTX)^2 + norm(alpha*XTX)^2 * norm(L1)^2 + norm(lamda*XT*L2*X)^2)+ norm(lamda2*A)^2);  
   %% proximal gradient
    while iter <= maxIter
      A = gradL21(W_s);
      W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);%W_s_k :W(t)
      gradF = XTX*W_s_k - XTY + alpha * XTX * W_s_k * L1 + lamda * XT * L2 * X * W_s_k + lamda2*A*W_s_k;
      Gw_s_k = W_s_k - 1/Lip *(gradF);% 
       %更新b ，W
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,beta/Lip);
      %% 开始计算损失函数的值 
       predictionLoss = trace((X*W_s - Y)'*(X*W_s - Y));
       F = X*W_s;
       correlation     = trace(F*L1*F');% label correlation
       In_correlation = trace(F'*L2*F);%instance correlation
       sparsity    = sum(sum(W_s~=0));% sparsity of W L1范数  specific features
       sparsity2    = trace(W_s'*A*W_s);
       totalloss = predictionLoss + alpha*correlation + beta*sparsity +lamda*In_correlation + lamda2*sparsity2;
       loss(iter,1) = totalloss;
       if abs(oldloss - totalloss) <= miniLossMargin
           %本次迭代的结果与上次的结果相差少于预订的最小损失间距时，结束循环
           break;
       elseif totalloss <=0  
           break;
       else
           oldloss = totalloss;
       end
       
       iter=iter+1; 
    end
    model_NewLLSF = W_s;

end


%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end

function S = ins_similarity(X,K)
    A     = pdist2(X,X); 
   [num_dim,~] = size(A);
   for i =1:num_dim
      temp = A(i,:);   
      As =sort(temp);
      temp  = (temp <=  As(K));
      A(i,:) = temp;
   end
   S = A;
end

function D = gradL21(W)
    num = size(W,1);
    D = zeros(num,num);
    for i=1:num
       temp = norm(W(i,:),2); 
       if temp~=0 
         D(i,i) = 1/temp;
       else
         D(i,i) = 0;
       end  
    end
end
