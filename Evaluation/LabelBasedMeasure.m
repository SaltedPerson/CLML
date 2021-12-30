
function [LabelBasedAccuracy,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure]=LabelBasedMeasure(test_targets,predict_targets)
% syntax
%   [LabelBasedAccuracy,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure]=LabelBasedMeasure(test_targets,predict_targets)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   predict_targets     - L x num_test data matrix of predicted labels
%
% output
%   LabelBasedAccuracy,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure


    [L,~]=size(test_targets);%得到标签个数
    test_targets=double(test_targets==1);%转换类型
    predict_targets=double(predict_targets==1);
    
    LabelBasedAccuracy=0;
    LabelBasedPrecision=0;
    LabelBasedRecall=0;
    LabelBasedFmeasure=0;
    
    for i=1:L%对于每个标签
        intersection=test_targets(i,:)*predict_targets(i,:)';%计算预测值正例与真实正例的 交集 个数
        union=sum(or(test_targets(i,:),predict_targets(i,:)));%计算预测正例与真实正例的 并集 个数
        
        if union~=0
            LabelBasedAccuracy=LabelBasedAccuracy + intersection/union;%准确率 = 预测正确/
        end
        
        if sum(predict_targets(i,:))~=0
            precision_i = intersection/sum(predict_targets(i,:));
        else
            precision_i=0;
        end
        if sum(test_targets(i,:))~=0
            recall_i = intersection/sum(test_targets(i,:));
        else
            recall_i=0;
        end
        LabelBasedPrecision=LabelBasedPrecision + precision_i;
        LabelBasedRecall=LabelBasedRecall + recall_i;
        if recall_i~=0 || precision_i~=0
            LabelBasedFmeasure=LabelBasedFmeasure + 2*recall_i*precision_i/(recall_i+precision_i);
        end
    end
    
    LabelBasedAccuracy=LabelBasedAccuracy/L;
    LabelBasedPrecision=LabelBasedPrecision/L;
    LabelBasedRecall=LabelBasedRecall/L;
    LabelBasedFmeasure=LabelBasedFmeasure/L;
end