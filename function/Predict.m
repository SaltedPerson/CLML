function predict_target = Predict(Outputs,tau)
%% 
 %Outputs: ��������Ԥ����
 %tau:ÿ����ǩ����ֵ
 
    predict_target = zeros(size(Outputs));%������Ԥ����ͬ����0�������ڴ�����ս��
    num_class = size(Outputs,1);%�õ����ǩ����
    for c = 1:num_class %����ÿ�����ǩ��
        predict_target(c,:) = Outputs(c,:) >= tau(1,c);%����tau()������ֵ�����Ƶ����յ�Ԥ�������
    end
    %predict_target = predict_target*2-1;
end