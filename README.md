1.提供对densenet实现过程的描述：  

想实现一个比较简单的densenet看看效果，就打算用3个block，分别是2,4,6层，growth为32
每个block都是稠密连接，
block之间用连接层transition连接，transition由1*1卷积，2*2平均池化组成
具体实现见quiz-w7-2-densenet\nets\densenet.py

DenseNet核心思想在于建立了不同层之间的连接关系，充分利用了feature，进一步减轻了梯度消失问题，加深网络不是问题，
而且训练效果非常好。另外，利用bottleneck layer，Translation layer以及较小的growth rate使得网络变窄，参数减少，
有效抑制了过拟合，同时计算量也减少了。DenseNet优点很多，而且在和ResNet的对比中优势还是非常明显的。

2.开始训练模型https://www.tinymind.com/executions/4egw7rxo
在 载入点输入  train_image_classifier.py

训练660次，提示没钱就终止运行了，从log图可以看出loss处于下降状态

3.利用训练好的模型来预测https://www.tinymind.com/executions/4ptbgv6j

在 载入点输入  eval_image_classifier.py来验证
虽然最后运行成功，并有正确率输出，但结果很让人失望

但拥有的点数以用完，还是自己有充了5美元，才跑出的结果
