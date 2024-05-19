import streamlit as st
import matplotlib.pyplot as plt
import preprocessor, helper
import seaborn as sns
import pandas as pd
# import pickle
# from nltk.corpus import stopwords
# import string
# from nltk.stem.porter import PorterStemmer
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer


plt.rcParams['font.family'] = 'Segoe UI Emoji'

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)


    #fetch unique users
    user_list = df["user"].unique().tolist()
    user_list.remove('group notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2 :
            st.header("Total Words")
            st.title(words)

        with col3 :
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4 :
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='coral')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='indigo')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='cyan')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='lime')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        # finding busiest users among the group
        if selected_user == 'Overall':
            st.title("Most Busy Users")

            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color="purple")
                plt.xticks(rotation="vertical")
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # word cloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1], color=["fuchsia", "dodgerblue", "lime", "yellow", "cyan"])
        plt.xticks(rotation="vertical")
        st.title("Most Common Words")
        st.pyplot(fig)


        # emoji used
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analyze")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:

            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f", colors=["forestgreen", "dodgerblue", "crimson", "yellow", "indigo"])
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap, cmap='viridis')
        st.pyplot(fig)

        # st.sidebar.subheader("Want to see sentiment analysis")

        # Sentiment Analysis
        st.title("Sentiment Analysis")

        col1, col2 = st.columns(2)
        sentiment_analyze_df = helper.sentiment_analyzer(selected_user, df)
        with col1:

            st.dataframe(sentiment_analyze_df)

        with col2:
            data_sentiment = sentiment_analyze_df.loc[0]
            labels_sentiment = sentiment_analyze_df.keys()
            fig, ax = plt.subplots()
            ax.pie(data_sentiment, labels=labels_sentiment, autopct="%0.2f", colors=["gold", "lime", "cyan"])
            st.pyplot(fig)



        # load saved model and vectorizer

        # tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        # model = pickle.load(open('model.pkl', 'rb'))
        #
        # nltk.download('stopwords')
        # ps = PorterStemmer()
        #
        #
        # def transform_text(text):
        #     text = text.lower()
        #     text = nltk.word_tokenize(text)
        #     text = [word for word in text if word.isalnum()]
        #     text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
        #     text = [ps.stem(word) for word in text]
        #
        #     return " ".join(text)


        # streamlit code
        st.title("Message spam classifier")
        col1, col2 = st.columns(2)
        # st.dataframe(df)
        with col1:
            result = helper.spam_classifier(selected_user, df)
            # spam_df = pd.DataFrame(data=result, columns=['Column1'])
            # type(result)
            # st.dataframe(result)
            spam_df = pd.concat([df, result],axis=1 )
            st.dataframe(spam_df)


        with col2:

            spam = spam_df[spam_df['output'] == 1].value_counts().sum()
            ham = spam_df[spam_df['output'] == 0].value_counts().sum()
            # st.text(spam)
            # st.text(ham)
            output = [spam , ham]
            output_idx = ["spam", "not spam"]
            fig, ax = plt.subplots()

            ax.pie(output, labels=output_idx, autopct="%0.2f", colors=["cyan", "lime"])
            st.pyplot(fig)


              # pass







        # # Monthly activity map
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",
        #                 unsafe_allow_html=True)
        #
        #     busy_month = helper.month_activity_map(selected_user, data, 1)
        #
        #     fig, ax = plt.subplots()
        #     ax.bar(busy_month.index, busy_month.values, color='green')
        #     plt.xticks(rotation='vertical')
        #     st.pyplot(fig)
        # with col2:
        #     st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",
        #                 unsafe_allow_html=True)
        #
        #     busy_month = helper.month_activity_map(selected_user, data, 0)
        #
        #     fig, ax = plt.subplots()
        #     ax.bar(busy_month.index, busy_month.values, color='grey')
        #     plt.xticks(rotation='vertical')
        #     st.pyplot(fig)
        # with col3:
        #     st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",
        #                 unsafe_allow_html=True)
        #
        #     busy_month = helper.month_activity_map(selected_user, data, -1)
        #
        #     fig, ax = plt.subplots()
        #     ax.bar(busy_month.index, busy_month.values, color='red')
        #     plt.xticks(rotation='vertical')
        #     st.pyplot(fig)
        # st.title("Spam or Ham Check")
        # ps = PorterStemmer()
        # def transform_text(text):
        #
        #     text = text.lower()
        #     text = nltk.word_tokenize(text)
        #
        #     y = []
        #     for i in text:
        #         if i.isalnum():
        #             y.append(i)
        #
        #     text = y[:]
        #     y.clear()
        #
        #     for i in text:
        #         if i not in stopwords.words('english') and i not in string.punctuation:
        #             y.append(i)
        #
        #     text = y[:]
        #     y.clear()
        #
        #     for i in text:
        #         y.append(ps.stem(i))
        #
        #     return " ".join(y)
        #
        #
        # tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        # model = pickle.load(open('model.pkl', 'rb'))
        #
        #
        #
        # # st.subheader("Want to check spam or ham?")
        # input_sms = st.text_area("Enter the message")
        # if st.button("Predict"):
        #
        #     # 1 Preprocess
        #     transformed_sms = transform_text(input_sms)
        #     # 2 vectorize
        #     vector_input = tfidf.transform([transformed_sms])
        #     # 3 predict
        #     result = model.predict(vector_input)[0]
        #     # 4 display
        #     if result == 1:
        #         st.header("spam")
        #     else:
        #         st.header("Not Spam")

    #st.sidebar.button("Show sentiment ")

