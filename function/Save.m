function [ res ] = Save( Avg_Result,time )
%SAVE 此处显示有关此函数的摘要
%   此处显示详细说明
nums = [1 2 3 5 10 11 12 13 14 15];
res = cell(1,11);
for i=1:size(nums,2)
 temp = [num2str(Avg_Result(nums(1,i),1),'%.3f'), '&',num2str(Avg_Result(nums(1,i),2),'%.3f')];
 res{1,i} = temp;
end
res{1,11} = time;
end

