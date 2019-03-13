import numpy
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data   #花的属性
iris_Y = iris.target  #花的标签
# print(iris_X[:,:1])
# print(iris_X[:,:])
#
# print(iris_Y)


X_train,X_test,Y_train,Y_test = train_test_split(iris_X,iris_Y,test_size=0.3)
# 上面的这一步骤在进行数据划分的时候，同时进行了数据的打乱操作 ，可以用print来观测验证

knn = KNeighborsClassifier()  #这一步就是定义我们的分类器
knn.fit(X_train,Y_train)  #这里就是把我们要训练的数据放到分类器中, 所有的训练步骤都在这里完成
print(knn.predict(X_test))  #使用训练好的模型，将验证集X_test放进去 预测
print(Y_test)  #这里就是将实际的验证标签Y_test与上面预测的做对比

c = knn.predict(X_test)-Y_test
print(c)