import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# data exploring
class Explore:
    def __init__(self, df):
        self.df = df
        
    # check shape of the data
    def shape(self):
        print('----------------Shape of the Dataset---------------- \n')
        print(self.df.shape)
     
    # check features in the data
    def features(self):
        print('----------------Features in the Dataset---------------- \n')
        print(self.df.columns)
        
    # Check summary statistics
    def stats(self):
        print('----------------Summary Statistics of the Features---------------- \n')
        print(self.df.describe()) 
    
    # check info on dataset
    def info(self):
        print('----------------Dataset Overall Information---------------- \n')
        print(self.df.info())


# data cleaning
class Clean(Explore):
    
    # check for missing values percentage
    def missing_duplicated(self):
    # identify the total missing values per column
    # sort in order 
        miss = self.df.isnull().sum().sort_values(ascending = False)

        # calculate percentage of the missing values
        percentage_miss = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending = False)

        # store in a dataframe 
        missing = pd.DataFrame({"Missing Values": miss, "Percentage(%)": percentage_miss})
    
        print("\n Duplicated Rows:\n")
        duplicate_count = self.df.duplicated().sum()
        print(f"- Total duplicated rows: {duplicate_count} \n \n")

        return missing

    # remove duplicated rows
    def remove_duplicated_rows(self):
        self.df.drop_duplicates(subset=None, keep="first", inplace=True)
        
        # confirm if the duplicated rows have been removed
        duplicates = f'The dataset now has {self.df.duplicated().sum()} duplicate rows'

        return duplicates
    
#Preprocessing Class 
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def generate_text_features(self, df, text_column):
        """
        Adds character count, word count, and sentence count columns.
        """
        df.loc[:, 'char_count'] = df[text_column].astype(str).apply(len)
        df.loc[:, 'word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))
        df.loc[:, 'sentence_count'] = df[text_column].astype(str).apply(lambda x: x.count('.') + 1)
        return df

    def clean_text(self, text):
        """
        Cleans text: lowercase, remove URLs, hashtags, emojis, punctuation, numbers, and extra spaces.
        """
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"#\w+", '', text)

        # Remove emojis
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)

        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        """
        Tokenizes and removes stopwords.
        """
        tokens = word_tokenize(text)
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens):
        """
        Lemmatizes a list of tokens.
        """
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, df, text_column):
        """
        Full preprocessing:
        - Feature engineering
        - Text cleaning
        - Tokenization
        - Lemmatization
        - Save cleaned text, tokens, lemmatized tokens, and document string (from tokens)
        """
        df = df.copy() 

        df = self.generate_text_features(df, text_column)
        df.loc[:, 'cleaned_text'] = df[text_column].astype(str).apply(self.clean_text)
        df.loc[:, 'tokenized_text'] = df['cleaned_text'].apply(self.tokenize)
        df.loc[:, 'lemmatized_text'] = df['tokenized_text'].apply(self.lemmatize)
        df.loc[:, 'document'] = df['tokenized_text'].apply(lambda x: ' '.join(x))

        return df