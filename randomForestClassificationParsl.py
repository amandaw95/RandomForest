from randomForestClassification import *


print(time.time())

'''
https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

n_estimators represents the number of trees in the forest. 
Usually the higher the number of trees the better to learn the data.

max_depth represents the depth of each tree in the forest. 
The deeper the tree, the more splits it has and it captures more information about the data

min_samples_split represents the minimum number of samples required to split an internal node. 
This can vary between considering at least one sample at each node to considering all of the samples at each node.

min_samples_leaf is The minimum number of samples required to be at a leaf node. 

max_features represents the number of features to consider when looking for the best split.
'''

for i in range(10,30):
	print ("Number of Estimators: " + str(i)) 
	print(rfClassifier(i).result())
	#print(rfClassifier(i))


print(time.time())
