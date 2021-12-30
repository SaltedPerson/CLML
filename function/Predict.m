function predict_target = Predict(Outputs,tau)
%% 
 %Outputs: 分类器的预测结果
 %tau:每个标签的阈值
 
    predict_target = zeros(size(Outputs));%构造与预测结果同样的0矩阵用于存放最终结果
    num_class = size(Outputs,1);%得到类标签个数
    for c = 1:num_class %对于每个类标签：
        predict_target(c,:) = Outputs(c,:) >= tau(1,c);%满足tau()条件的值被复制到最终的预测矩阵中
    end
    %predict_target = predict_target*2-1;
end