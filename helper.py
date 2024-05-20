import nltk
import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji
# from nltk.corpus import stopwords
# import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('vader_lexicon')
extract = URLExtract()

def fetch_stats(selected_user, df):

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # 1. fetch the number of messages
    num_messages = df.shape[0]

    # 2. fetch the Number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # 3. fetch the Number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # 4.fetch the number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df =round((df['user'].value_counts()/df.shape[0])*100, 2).reset_index().rename(columns={'user':'users',
                                                                                            'count':'percentage'})
    return x, df


def create_wordcloud(selected_user, df):

    f = open('stopwords.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)

        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

# Most common words

def most_common_words(selected_user, df):

    f = open('stopwords.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        # emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    # emojis = []
    # for message in df['message']:
    #     emojis.extend([c for c in message if c in emoji.UNICODE_DATA and 'Emoji' in emoji.UNICODE_DATA.get(c,
    #                                                                                                        {})])  # Add error handling for missing keys

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap =  df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def sentiment_analyzer(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != "group notification"]
    df = temp[temp['message'] != "<Media omitted>\n"].reset_index(drop=True)

    sentiments = SentimentIntensityAnalyzer()
    df['positive'] = [sentiments.polarity_scores(i)['pos'] for i in df['message']]
    df['negative'] = [sentiments.polarity_scores(i)['neg'] for i in df['message']]
    df['neutral'] = [sentiments.polarity_scores(i)['neu'] for i in df['message']]

    x = sum(df['positive'])
    y = sum(df['negative'])
    z = sum(df['neutral'])

    percent_pos = x * 100 / (x + y + z)
    percent_neg = y * 100 / (x + y + z)
    percent_neu = z * 100 / (x + y + z)

    percent_df = pd.DataFrame([[percent_pos, percent_neg, percent_neu]], columns=["per_pos", "per_neg", "per_neu"])

    return percent_df

nltk.download('stopwords')
nltk.download('punkt')
def transform_text(text):
    # nltk.download('stopwords')
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]

    return " ".join(text)

def spam_classifier(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))


    output = []
    for message in df['message']:

        # input_sms = message

        # if st.button('Predict'):

        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        output.append(result)

    spam_df = pd.DataFrame(output,columns=['output'])

    return spam_df
# def week_activity_map(selected_user, df, k):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#     df = df[df['value'] == k]
#     return df['day_name'].value_counts()
#
# def month_activity_map(selected_user,df,k):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#     df = df[df['value'] == k]
#     return df['month'].value_counts()






# def spam_check_preprocess(selected_user, df):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#
#
#     text = df['message'].lower()
#     text = nltk.word_tokenize(text)
#
#     y = []
#     for message in text:
#
#         if message.isalnum():
#             y.append(message)
#
#         if message not in stopwords.words('english') and message not in string.punctuation:
#             y.append(message)
#             y.append(ps.stem(message))

    # text = y[:]
    # y.clear()

    # for message in text :
    #
    #     if message not in stopwords.words('english') and message not in string.punctuation:
    #         y.append(message)
    #
    # for message in

    # df['text'] =
    #
    # return y






# def
    # x = sum(temp_df['positive'])
    # y = sum(temp_df['negative'])
    # z = sum(temp_df['neutral'])
    #
    #
    # def score(a, b, c):
    #     if a > b and a > c:
    #         print('Positive sentiment')
    #     elif b > a and b > c:
    #         print('negative sentiment')
    #     elif c > a and c > b:
    #         print('neutral sentiment')
    #
    #
    # score(x, y, z)
    #
    # x * 100 / (x + y + z)
    # y * 100 / (x + y + z)
    # z * 100 / (x + y + z)


    # return df








#      1. fetch the number of messages
    #     num_messages = df.shape[0]
    #
    #      2. Number of words
    #     words=[]
    #     for message in df['message']:
    #         words.extend(message.split())
    #
    #     return num_messages, len(words)
    # else:
    #     new_df = df[df['user'] == selected_user]
    #     num_messages =  new_df.shape[0]
    #
    #     # 2. fetch the number of words
    #     words = []
    #     for message in new_df['message']:
    #         words.extend(message.split())
    #
    #     return num_messages, len(words)
