import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df=pd.read_csv("data/Bitext_Sample_Customer_Service_Validation_Dataset.csv")
df1=pd.read_csv("data/Bitext_Sample_Customer_Service_Training_Dataset.csv")
df2=pd.read_csv("data/Bitext_Sample_Customer_Service_Testing_Dataset.csv")

df_all = pd.concat([df,df1,df2],ignore_index=True)
df_all = df_all.drop(["category", "tags"], axis=1)


lb=LabelEncoder()
df_all["intent"] = lb.fit_transform(df_all["intent"])
print(len(lb.classes_))

tf = TfidfVectorizer(max_features=2000)
X=tf.fit_transform(df_all["utterance"]).toarray()
y=df_all["intent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(512,activation='relu',input_shape=(X.shape[1],)))
model.add(Dense(256,activation='relu'))
model.add(Dense(len(lb.classes_),activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=20,batch_size=32,validation_data=(X_test,y_test))

model.save("models/customer_model.keras")

with open("models/tfidf.pkl","wb") as f:
    pickle.dump(tf,f)

with open("models/label_encoder.pkl","wb") as f:
    pickle.dump(lb,f)

print("------------------------------------------")
print("SUCCESS: All models and transformers saved!")
print("Location: /models folder")
print("------------------------------------------")


