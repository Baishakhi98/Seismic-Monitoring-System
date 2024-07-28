import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import pickle


df=pd.read_csv("extraction_features.csv")
y=df["Label"]
x=df.drop(['Signal','Label'],axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, test_size=0.3,random_state=42)


rfc=ensemble.RandomForestClassifier()
rfc.fit(x_train,y_train)
#pred=rfc.predict(x_test)
y_pred_val = rfc.predict(x_val)
acc=metrics.accuracy_score(y_val,y_pred_val)
#print(acc*100)
cm = confusion_matrix(y_val,y_pred_val)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


#print(cm)


#from sklearn import ensemble

classifier =ensemble.RandomForestClassifier()

# Fit the model
classifier.fit(x_train, y_train)

# Make pickle file of our model
#pickle.dump(classifier, open("model_100.pkl", "wb"))
#model=pickle.load(open('model_100.pkl','rb'))


with open('rfc_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

