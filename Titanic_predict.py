import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin' # 这里是 path+ Graphviz/bin 即 Graphviz 的 bin目录


# 数据加载
train_data = pd.read_csv('C:/Users/O-O RT/Downloads/Titanic_Data-master/Titanic_Data-master/train.csv')
test_data = pd.read_csv('C:/Users/O-O RT/Downloads/Titanic_Data-master/Titanic_Data-master/test.csv')
# 数据探索
print(train_data.info())
print('--' * 30)
print(train_data.describe())
print('--' * 30)
print(train_data.describe(include=['O']))
print('--' * 30)
print(train_data.head())
print('--' * 30)
print(train_data.tail())
print('--' * 30)
# 数据清洗
print(test_data.info())
print('--' * 30)
# 测试集中类别：age、fare、cabin存在缺失值，训练集中类别：age、cabin、embarked存在缺失值

# 使用平均年龄补充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
# 使用票价的平均值来补齐票价中的nan值
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
# cabin在测试集和训练集中缺失率过高，不进行补充
# 使用登陆最多的港口来代替缺失值
print(train_data['Embarked'].value_counts())  # 登陆最多的是s港口
train_data['Embarked'].fillna('S', inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 决策树模型（ID3）
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(train_features, train_labels)

# 模型预测和评估
# 得到测试集的特征值矩阵
test_features = dvec.transform(test_features.to_dict(orient='record'))
pred_labels = clf.predict(test_features)
# 得到决策树的准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'决策树准确率：%.4lf' % acc_decision_tree)
# 使用K折交叉验证
print(u'cross_val_score准确率为:%.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))

# 决策树可视化
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.view()