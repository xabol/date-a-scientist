import pandas as pd
import numpy as np
import re
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

#Create your df here:
df = pd.read_csv('profiles.csv')

df.essay0.head()
df.income.value_counts()
df.dtypes.sample(31)

print(df.columns)

for i in df.columns:
    column = i
    print(column)
    print(df[column][0])


plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

df.sex.value_counts()

def  word_counter(essay):
    words = essay.split()
    count = Counter(words)
    return count['education'] + count['college'] + count['university']

def  avg_word_length(essay):
    words = essay.split()
    total_length = 0
    avg_length = 0
    for i in words:
        if len(i) > 0:
            total_length += len(i)
            avg_length = total_length / len(words)
    return avg_length
        


print(len(df))

drink_mapping = {"not at all": 1, "rarely": 2, "socially": 3,
                 "often": 4, "very often": 5, "desperately": 6}
df["drinks_code"] = df.drinks.map(drink_mapping)

df.drinks_code.value_counts()

smokes_mapping = {"no": 1, "sometimes": 2, "when drinking": 3,
                 "trying to quit": 4, "yes": 5}
df["smokes_code"] = df.smokes.map(smokes_mapping)

df.smokes_code.value_counts()

drugs_mapping = {"never": 1, "sometimes": 2, "often": 3}

df["drugs_code"] = df.drugs.map(drugs_mapping)

df.drugs_code.value_counts()



education_mapping = {"graduated from high school": 2,
                     "dropped out of college/university": 2,
                     "graduated from space camp": 2, 
                     "dropped out of space camp": 2,
                     "working on space camp": 2,
                     "dropped out of two-year college": 2,
                     "dropped out of masters program": 2,
                     "dropped out of ph.d program": 2,
                     "dropped out of high school": 2,
                     "high school": 2,
                     "working on high school": 2,
                     "space camp": 2,
                     "graduated from college/university": 1,
                     "graduated from masters program": 1,
                     "working on college/university": 1,
                     "working on masters program": 1,
                     "graduated from two-year college": 1,
                     "graduated from ph.d program": 1,
                     "graduated from law school": 1,
                     "working on two-year college": 1,
                     "working on ph.d program": 1,
                     "college/university": 1,
                     "graduated from med school": 1,
                     "working on law school": 1,
                     "two-year college": 1,
                     "working on med school": 1,
                     "masters program": 1,
                     "ph.d program": 1,
                     "law school": 1,
                     "med school": 1,
                     "dropped out of law school": 1,
                     "dropped out of med school": 1}

df["education_code"] = df.education.map(education_mapping)

df.education_code.value_counts()

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6",
              "essay7","essay8","essay9"]

all_essays = df[essay_cols].replace(np.nan, '', regex=True)

all_essays = all_essays[essay_cols].replace('br', '', regex=True)

all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

print(all_essays.head())
print(all_essays[0])



df["essay_len"] = all_essays.apply(lambda x: len(x))


df.essay_len.head()

df['new_essays'] = all_essays.apply(lambda x: re.sub(r'\W+', ' ', x))

df['new_essays'].head()
print(df['new_essays'][0])


df['essay_word_freq'] = df['new_essays'].apply(lambda x: word_counter(x))


df['avg_word_length'] = df['new_essays'].apply(lambda x: avg_word_length(x))

print(df['avg_word_length'])

df['avg_word_length'].idxmax()

df['essay_word_freq'].value_counts()

print(df['new_essays'][31734])



print(avg_word_length(df['new_essays'][0]))
print(avg_word_length(df['essay0'][0]))
print(df['essay9'][0])
print(df['new_essays'][0])


df['age_bucket'] = pd.cut(df['age'], [0, 18, 23, 29, 39, 49, 59, 69, 120],
  labels=['0-18', '19-23', '24-29', '30-39', '40-49', '50-59', '60-69', 'Old'])

print(df['essay_len'].value_counts())

age_bucket_map = {
    "0-18": 1,
    "19-23": 2,
    "24-29": 3,
    "30-39": 4,    
    "40-49": 5,
    "50-59": 6,
    "60-69": 7,
    "Old": 8
}

df["age_bucket_code"] = df.age_bucket.map(age_bucket_map)

print(df.age_bucket_code.value_counts())

vectorizer = CountVectorizer()

df['drugs_code'].min()

df['smokes_code'].value_counts()


print (df[['smokes_code']].isnull().values.sum())

#MULTILINEARREGRESSION

regression_one_cols = ['smokes_code', 'drugs_code', 'drinks_code', 'essay_len', 'avg_word_length', 'age']
feature_data = df[regression_one_cols]
feature_data = feature_data[regression_one_cols].dropna(0)
#feature_data[['essay_len', 'age_bucket_code', 'education_code']].replace(np.nan, '', regex=True)

print (feature_data[regression_one_cols].isnull().values.sum())

print(feature_data)

x = feature_data.values

scaler = preprocessing.RobustScaler()
x_scaled = scaler.fit_transform(x)


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)


x = feature_data[['smokes_code', 'drugs_code', 'drinks_code', 'essay_len', 'avg_word_length']]

y = feature_data['age']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()
model = mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)

print(model.coef_)
r2 = model.score(x_train, y_train)
print(r2)

plt.scatter(y_test, y_predict, alpha = 0.4)
plt.xlabel('acutal')
plt.ylabel('predicted')
plt.show()


plt.hist(feature_data['age_bucket_code'], bins=10)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(-10, 10)
plt.show()

# K nearest neigbour regressor

regression_two_cols = ['smokes_code', 'drugs_code', 'drinks_code', 'essay_len', 'avg_word_length', 'age']
feature_data_two = df[regression_two_cols]
feature_data_two = feature_data_two[regression_two_cols].dropna(0)

x = feature_data_two.values

scaler = preprocessing.RobustScaler()
x_scaled = scaler.fit_transform(x)


feature_data_two = pd.DataFrame(x_scaled, columns=feature_data_two.columns)


x2 = feature_data_two[['smokes_code', 'drugs_code', 'drinks_code', 'essay_len', 'avg_word_length']]


y2 = feature_data_two['age']

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size = 0.8, test_size = 0.2, random_state=6)

k_list = []
for i in range(1,20):
    regressor = KNeighborsRegressor(n_neighbors = i, weights = 'distance')
    model2 = regressor.fit(x2_train, y2_train)
    y2_predict = regressor.predict(x2_test)
    r2_2 = model2.score(x2_train, y2_train)
    k_list.append(r2_2)
    print(r2_2)
    
print(k_list)

plt.scatter(y2_test, y2_predict, alpha = 0.4)
plt.xlabel('acutal')
plt.ylabel('predicted')
plt.ylim(-5, 5)
fig5 = plt.gcf()
plt.show()
fig5.savefig('kregressor.png')

plt.plot(k_list)
plt.xlabel("K")
plt.ylabel("R2")
fig6 = plt.gcf()
plt.show()
fig6.savefig('Kvalues.png')


# K CLOSEST NEIGHBOURS

classifier_cols = ['age_bucket_code', 'education_code', 'essay_word_freq', 'essay_len', 'avg_word_length']
feature_data_three = df[classifier_cols]
feature_data_three = feature_data_three[classifier_cols].dropna(0)

x = feature_data_three.values

scaler = preprocessing.RobustScaler()
x_scaled = scaler.fit_transform(x)


feature_data_three = pd.DataFrame(x_scaled, columns=feature_data_three.columns)

x3 = feature_data_three[['age_bucket_code', 'essay_word_freq', 'essay_len', 'avg_word_length']]


y3 = feature_data_three['education_code']

x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, train_size = 0.8, test_size = 0.2, random_state=6)



classifier = KNeighborsClassifier(5)
classifier.fit(x3_train, y3_train)
y3_predict = classifier.predict(x3_test)


print(f1_score(y3_test, y3_predict))
print(precision_score(y3_test, y3_predict))
print(recall_score(y3_test, y3_predict))  

print(classifier.score(x3_test, y3_test))

#SVM

classifier_two_cols = ['age_bucket_code', 'education_code', 'essay_word_freq', 'essay_len', 'avg_word_length']
feature_data_four = df[classifier_two_cols]
feature_data_four = feature_data_four[classifier_two_cols].dropna(0)

x = feature_data_four.values

scaler = preprocessing.RobustScaler()
x_scaled = scaler.fit_transform(x)


feature_data_four = pd.DataFrame(x_scaled, columns=feature_data_four.columns)

x4 = feature_data_four[['age_bucket_code', 'essay_word_freq', 'essay_len', 'avg_word_length']]


y4 = feature_data_four['education_code']

x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, train_size = 0.8, test_size = 0.2, random_state=6)



classifier_two = SVC(kernel = 'rbf', gamma = 0.9)

classifier_two.fit(x4_train, y4_train)

y4_predict = classifier_two.predict(x4_test)


print(f1_score(y4_test, y4_predict))
print(precision_score(y4_test, y4_predict))
print(recall_score(y4_test, y4_predict))  

print(classifier_two.score(x4_test, y4_test))


plt.hist(df['age_bucket_code'], bins=8)
plt.title('Age')
plt.xlabel("Age")
plt.ylabel("Members")
plt.xticks(np.arange(8), ('0-18', '19-23', '24-29', '30-39', '40-49', '50-59', '60-69', 'Old'))
fig1 = plt.gcf()
plt.show()
fig1.savefig('age.png')

plt.hist(feature_data_three['education_code'], bins=2, rwidth = 0.5)
plt.title('Member education')
plt.xlabel("Education")
plt.ylabel("Members")
plt.xticks(np.arange(2), ('College or higher', 'Lower'))
fig2 = plt.gcf()
plt.show()
fig2.savefig('member_education.png')

plt.hist(df['essay_len'], bins=100)
plt.title('Total words in essay')
plt.xlabel("Words in essay")
plt.ylabel("Members")
plt.xlim(0, 10000)
fig3 = plt.gcf()
plt.show()
fig3.savefig('Totalwords.png')

plt.hist(df['avg_word_length'], bins=10000)
plt.title('Average word length in essay')
plt.xlabel("Average word length in essay")
plt.ylabel("Members")
plt.xlim(4, 5)
fig4 = plt.gcf()
plt.show()
fig4.savefig('avgword.png')




















