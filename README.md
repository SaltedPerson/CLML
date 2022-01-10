# Learning Common and Label-Specific Features for Multi-Label classification with correlation information



# Abstract
In multi-label classification, many existing works only pay attention to the label-specific features and label correlation while they ignore the common features and instance correlation, which are also essential for building a competitive classifier. Besides, existing works usually depend on the assumption that they tend to have the similar label-specific features if two labels are correlated. However, this assumption cannot always hold in some cases. Therefore, in this paper, we propose a new approach of learning common and label-specific features for multi-label classification using the correlation information from labels and instances. First, we introduce l2,1-norm and l1-norm regularizers to learn common and label-specific features simultaneously. Second, we use a regularizer to constrain label correlations on label outputs instead of coefficient matrix. Finally, instance correlations are also considered through the k-nearest neighbor mechanism. Comprehensive experiments manifest the superiority of our proposed approach against other well-established multi-label learning algorithms for label-specific features.

# Main contributions
1.Common and label-specific features are considered in our approach simultaneously, meanwhile l1-norm and l2,1-norm regularizers are utilized to select the corresponding features respectively. It is beneficial to improve the classification performance.            

2.We consider the label correlation and the instance correlation simultaneously in our approach. We first illustrate to show that a popular assumption of label correlation used in many existing multi-label learning algorithms cannot always hold in a certain case. Then, we introduce a new assumption to exploit label correlations, that is, if two labels are strongly correlated, they should have similar output rather than coefficient vector. In addition, we adopt the k-nearest neighbor algorithm to evaluate the instance correlation, because KNN is simple and insensitive to the noisy data.          

3.We design an objective function to jointly learning common and label-specific features while exploiting label and instance correlations. Extensive experimental results demonstrate our proposed approach outperforms well-known label-specific learning algorithms on example-based and ranking-based evaluation metrics


# Contact
Peipei Li (peipeili@hfut.edu.cn): Hefei University of Technology.      

Junlong Li (ljl@mail.hfut.edu.cn): Hefei University of Technology.     

Xuegang Hu (jsjxhuxg@hfut.edu.cn): Hefei University of Technology.       

Kui Yu (yukui@hfut.edu.cn): Hefei University of Technology.    

# Publication
 The paper has been accepted by Pattern Recognition.2021

