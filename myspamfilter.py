
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer




#read the data
#df = pd.read_csv("SPAM text message 20170820 - Data.csv")
class spamfiler():
    def __init__(self, data):
        self.df = data


    # two columns, one with the classification and the other with data
    # change the label fro spam -> 1, to cluster spam messages and answer "Is that message spam or not"


    def do(self):
        le = LabelEncoder()
        self.df['Category'] = le.fit_transform(self.df['Category'])
        print(self.df.head())
        wordnet = WordNetLemmatizer()
        filtered_text = []

        def preprocessing(review):
        #for i in range(0, len(self.df)):
            review = re.sub(r'<.*?>', '', review)
            review = re.sub(r'[^a-zA-Z]+', ' ', review)
            review = re.sub(r'[0-9]', '', review)
            review = review.lower()
            review = review.split()
            review = ' '.join(review)
            filtered_text.append(review)
        for i in range(0, len(self.df)):
            preprocessing(self.df["Message"][i])

        print("filtered text", filtered_text[1])
        # A -> Text
        A = pd.DataFrame(filtered_text, columns=['text'])
        # B-> Ham or Spam (0,1)
        B = self.df["Category"]
        print(A)

        # split data into train and test, got from lecture
        from sklearn.model_selection import train_test_split
        Atrain, Atest, B_train, B_test = train_test_split(A, B)

        cv = CountVectorizer(max_features=3000)  # max_feature sets the number of features, if not specfied the number of features will be equal to the the vocab
        #print(Atest)
        A_train = cv.fit_transform(Atrain['text']).toarray()
        A_test = cv.transform(Atest['text']).toarray()



        from sklearn.naive_bayes import MultinomialNB
        nb = MultinomialNB().fit(A_train, B_train)
        B_pred_nb = nb.predict(A_test)
        print(B_pred_nb)



        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        print('Precision Score: ', precision_score(B_test, B_pred_nb))
        print('Recall Score: ', recall_score(B_test, B_pred_nb))
    def tryit(self, text):
        pass

def main():
    df = df = pd.read_csv("SPAM text message 20170820 - Data.csv")

    filter = spamfiler(df)
    filter.do()

if __name__ == "__main__":
    main()