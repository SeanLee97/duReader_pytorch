# 请转至[QANet_dureader](https://github.com/SeanLee97/QANet_dureader)

# duReader_pytorch
基于duReader的阅读理解

这个数据集的不合理之处：
1）标注文章选择不合理，有一些没有被选到的文章，其实也包含了答案。
   在做开放域问答的时候我们会搜索到很多没有答案的文章，模型需要识别出来这些无用文章，所以在训练的时候需要负例，而这里的负例是不合理的。
2）属性写得很模糊..........fake answer其实应该是golden answer，is_selectd也没有被选中作为答案文章


so，还不如刷微软的MS MARCO数据集呢
