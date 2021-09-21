import streamlit as st
from streamlit import components
import pandas as pd

# File Processing Pkgs
import pandas as pd
import docx2txt

#Utils
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

from io import BytesIO

# text preprocessing
import re

# Topic Modelling
import gensim
from gensim import corpora
import pyLDAvis
import pickle
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

#Sentiment package
import nltk
# Fxn to download

def csv_downloader(data):
    csvfile = data.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "new_text_file_{}.csv".format(timestr)
    st.markdown("#### Download File ####")
    #link creation using f option passing b64 encoding over the wire to be able to allow it to make it downloadable
    href = f'<a href="data:file/csv;base64,{b64}" download = "{new_filename}"> Click Here!!</a>'
    st.markdown(href, unsafe_allow_html=True)

#class
class FileDownloader(object):

    def __init__(self, data, filename = 'myfile', file_ext='txt'):
        super(FileDownloader, self).__init__()
        self.data = data
        self.filename = filename
        self.file_ext = file_ext

    def download(self):
        b64 = base64.b64encode(self.data.encode()).decode()
        new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
        st.markdown("#### Download File ####")
        #link creation using f option passing b64 encoding over the wire to be able to allow it to make it downloadable
        href = f'<a href="data:file/{self.file_ext};base64,{b64}" download = "{new_filename}"> Click Here!!</a>'
        st.markdown(href, unsafe_allow_html=True)

def get_table_download_link_csv(df):
    csv = df.to_csv(index=False)
    #csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown("#### Download file ####")
    href = f'<a href="data:file/csv;base64;{b64}" download="{output.csv}"> Click Here to download csv file </a>'
    st.markdown(href, unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body{
    background-image:url("data:image/png;base64,%s");
    background-size:cover;
    }
    </style>
    ''' %bin_str

    st.markdown(page_bg_img,unsafe_allow_html=True)
    return
    set_png_as_page_bg('Background.png')

def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|-(https?://[^\s]+))',' ', text)
    text = re.sub('@[^\s]+',' ', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(r'\d+','', text)
    text = re.sub(' +',' ', text)
    return text.strip()
def main():
    #st.title("Sentiment Analyzer")
    menu = ["Home" , "Sentiment & Topic Modelling"]
    choice = st.sidebar.selectbox("Upload Options",menu)
    if choice =="Home":
        st.subheader("")

    elif choice =="Sentiment & Topic Modelling":
        st.subheader("Upload your Dataset")
        data_file = st.file_uploader("Upload CSV",
        type=["csv"])

        if data_file is not None:
            st.write(type(data_file))
            #Methods/Attributes
            #st.write(dir(image_file))
            file_details = {"filename":data_file.name,
            "filetype":data_file.type,"filesize":data_file.size}
            st.write(file_details)

            df = pd.read_csv(data_file)
            st.dataframe(df)


            #if st.button("Analyze & Export"):
            st.markdown(" ##### Sentiment Analysis #####")
            reviews = df["verbatim"].tolist()
            sentiments = SentimentIntensityAnalyzer()
            score = []
            Lable = []
            for sent in reviews:
                #out = out + str(sent)
                ss = sentiments.polarity_scores(str(sent))
                #print(ss)
                for k,v in ss.items():
                    if k == "compound":
                        score.append(str(v))
                        if v >= 0.05:
                            Lable.append("Positive")
                        if v <= -0.05:
                            Lable.append("Negative")
                        if v < 0.05 and v > -0.05:
                            Lable.append("Neutral")
            df["Sentiment_Score"] = score
            df["Sentiment_Lable"] = Lable
            #st.dataframe(df[:-5])
            st.write(df.head(10))
            #st.dataframe(df)
        #if st.button("Export"):
            #st.write(df)
            #st.dataframe(df)
            # csv_downloader(df)
            download = FileDownloader(df.to_csv(),file_ext='csv').download()

            #if st.button("Topic Modelling Analyze"):
            st.markdown(" ##### LDA Analysis #####")

            df['Cleaned_content'] = df['verbatim'].str.replace('http\S+|www.\S+', '', case=False)
            df['Cleaned_content'] = [preprocess_text(str(i)) for i in df['Cleaned_content']]
            stop = set(stopwords.words('english'))
            #Remove stopwords
            def clean(doc):
                stop_free = " ".join([i for i in str(doc).lower().split() if i not in stop])
                return stop_free
            tweets_list = df["Cleaned_content"].to_list()
            doc_clean = [clean(doc).split() for doc in tweets_list]

            with st.form(key='my form'):
                num_topics =  st.number_input(label='Number of Topics', min_value=2,value=3, max_value=10,step=1)
                submit_button = st.form_submit_button(label='Submit')
                if submit_button :
                    st.write("Number of topics chosen is:{}".format(num_topics))
                    ldamodel = Lda(dtm, num_topics= num_topics, id2word = dictionary, passes=20)
                    st.write(ldamodel.print_topics(num_topics=num_topics, num_words=5))

                    # Compute Coherence Score
                    coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
                    coherence_lda = coherence_model_lda.get_coherence()
                    st.write('\nCoherence Score: ', coherence_lda)
                    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
                        coherence_values = []
                        model_list = []
                        for num_topics in range(start, limit, step):
                            model=Lda(corpus=corpus, id2word=dictionary, num_topics=num_topics)
                            model_list.append(model)
                            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                            coherence_values.append(coherencemodel.get_coherence())

                        return model_list, coherence_values
                    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=doc_clean, start=2, limit=10, step=1)
                    limit=10; start=2; step=1;
                    x = range(start, limit, step)
                    plt.plot(x, coherence_values)
                    plt.xlabel("Num Topics")
                    plt.ylabel("Coherence score")
                    plt.legend(("coherence_values"), loc='best')
                    plt.show()


                    def format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=doc_clean):
                        # Init output
                        sent_topics_df = pd.DataFrame()

                        # Get main topic in each document
                        for i, row in enumerate(ldamodel[corpus]):
                            row = sorted(row, key=lambda x: (x[1]), reverse=True)
                            # Get the Dominant topic, Perc Contribution and Keywords for each document
                            for j, (topic_num, prop_topic) in enumerate(row):
                                if j == 0:  # => dominant topic
                                    wp = ldamodel.show_topic(topic_num)
                                else:
                                    break
                        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

                    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=doc_clean)

                    # Format
                    df_dominant_topic = df_topic_sents_keywords.reset_index()
                    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

                    # Show
                    df_dominant_topic.head(10)

                    df_dominant_topic.Dominant_Topic.describe()

                    df["Topics_1"] = df_dominant_topic["Dominant_Topic"]
                    df["Keywords"] = df_dominant_topic["Keywords"]
                    df["Topic_perc_contr"] = df_dominant_topic["Topic_Perc_Contrib"]

                    st.write(df.head(10))
                    download = FileDownloader(df.to_csv(),file_ext='csv').download()

    else:
        st.header("")

if __name__ == '__main__':
    main()
