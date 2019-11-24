from parsl import load, python_app
from parsl.configs.local_threads import config
load(config)

import pandas as pd
import numpy as np
import time



df = pd.read_csv("/home/amanda/Downloads/bill_authentication.csv")

from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

maxThreads = 10
local_threads = Config(
    executors=[
        ThreadPoolExecutor(
            max_threads=maxThreads,
            label='local_threads'
        )
    ]
)




@python_app
def rfClassifier(estimators):

	import pandas as pd
	import numpy as np



	dataset = df

	dataset.head()


	X = dataset.iloc[:, 0:4].values
	y = dataset.iloc[:, 4].values

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	# Feature Scaling
	from sklearn.preprocessing import StandardScaler

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	from sklearn.ensemble import RandomForestClassifier

	classifier = RandomForestClassifier(n_estimators=estimators, random_state=0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

	#print(confusion_matrix(y_test,y_pred))
	#print(classification_report(y_test,y_pred))
	#print(accuracy_score(y_test, y_pred))

	return str(confusion_matrix(y_test,y_pred)) + '\n' +(classification_report(y_test,y_pred)) + '\n' + str(accuracy_score(y_test, y_pred))



#print(rfClassifier().result())

