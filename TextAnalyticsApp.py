import streamlit as st
from streamlit import components
import pandas as pd

# File Processing Pkgs
import pandas as pd
import docx2txt
#import textract
from PIL import Image #pillow
from PyPDF2 import PdfFileReader
import pdfplumber
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
#nltk.data.path.append("C:\\Users\\kavitha.a\\sample files\\nltk_data")
#nltk.download('stopwords')
#nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load Images
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text
# Fxn to download
def text_downloader(raw_text):
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_filename = "new_text_file_{}_.txt".format(timestr)
    st.markdown("#### Download File ####")
    #link creation using f option passing b64 encoding over the wire to be able to allow it to make it downloadable
    href = f'<a href="data:file/txt;base64,{b64}" download = "{new_filename}"> Click Here!!</a>'
    st.markdown(href, unsafe_allow_html=True)

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
    """docstring forFileDownloader.
    >>>> download = FileDownloader(data, filename, file_ext).download()
    """

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

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer,index = False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

    st.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)

@st.cache(allow_output_mutation = True)
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

def my_widget(key):
        clicked = st.button("Click me " + key)


def main():
    #st.title("Sentiment Analyzer")
    menu = ["Home" , "Sentiment by Text", "Image" , "Sentiment & Topic Modelling" , "Document/Files" , "Contact Us" , "About"]
    choice = st.sidebar.selectbox("Upload Options",menu)
    if choice =="Home":
        st.subheader("")
        #Primary accent for interactive elements '#7792E3'
        primaryColor = '#6eb52f'
        # Background color for the main content area
        backgroundColor = '#273346'
        # Background color for sidebar and most interactive widgets secondary
        BackgroundColor = '#B9F1C0'
        # Color used for almost all text
        textColor = '#FFFFFF'
        # Font family for all text in the app, except code blocks # Accepted values (serif | sans serif | monospace) #Default: "sans serif"
        font = "sans serif"
        main_bg = "Background.png"
        main_bg_ext = "png"
        st.markdown( f""" <style> .reportview-container {{ background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg,
        "rb").read()).decode()}) }} </style> """, unsafe_allow_html=True)

    elif choice =="Sentiment by Text":
        st.subheader("Sentiment by Text")
        my_text = st.text_area("Enter Your Message")
        if st.button("Save"):
            st.write(my_text)
            sentiments = SentimentIntensityAnalyzer()
            ss = sentiments.polarity_scores(str(my_text))
            st.write(ss.items())

            #text_downloader(my_text)
            download = FileDownloader(my_text).download()

    elif choice =="Image":
        st.subheader("Image")
        image_file = st.file_uploader("Upload Image",
            type=["png","jpg","jpeg"])
        if image_file is not None:
            st.write(type(image_file))
#Methods/Attributes
            #st.write(dir(image_file))
            file_details = {"filename":image_file.name,
            "filetype":image_file.type,"filesize":image_file.size}
            st.write(file_details)

            st.image(load_image(image_file),width=250)


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
            #new_sw = pd.read_csv("Stopwords_test.csv", encoding="iso-8859-1")
            #new_sw1 = new_sw['words'].tolist()
            #for i in new_sw1:
            #    stop.add(i)
            #    stop.add("coronavirus")
            tweets_list = df["Cleaned_content"].to_list()
            doc_clean = [clean(doc).split() for doc in tweets_list]

            dictionary = corpora.Dictionary(doc_clean)
            dtm = [dictionary.doc2bow(doc) for doc in doc_clean]
            Lda = gensim.models.ldamodel.LdaModel

            #num_topics = st.number_input('Number of Topics',min_value=2,step=1)
            #clicked = my_widget("First")

            with st.form(key='my form'):
                num_topics =  st.number_input(label='Number of Topics', min_value=2,value=3, max_value=10,step=1)
                submit_button = st.form_submit_button(label='Submit')
                if submit_button :
                    st.write("Number of topics chosen is:{}".format(num_topics))


                    ldamodel = Lda(dtm, num_topics= num_topics, id2word = dictionary, passes=20)
                    st.write(ldamodel.print_topics(num_topics=num_topics, num_words=5))

                    ldamodel.save('model.gensim')
                    pickle.dump(dtm, open('dtm1.pkl', 'wb'))
                    dictionary.save('dictionary1.gensim')

                    dictionary = gensim.corpora.Dictionary.load('dictionary1.gensim')
                    corpus = pickle.load(open('dtm1.pkl', 'rb'))
                    lda = gensim.models.ldamodel.LdaModel.load('model.gensim')

                    from gensim.models.coherencemodel import CoherenceModel
                    # Compute Perplexity
                    st.write('\nPerplexity: ', ldamodel.log_perplexity(dtm))  # a measure of how good the model is. lower the better.

                    # Compute Coherence Score
                    coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
                    coherence_lda = coherence_model_lda.get_coherence()
                    st.write('\nCoherence Score: ', coherence_lda)
                    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
                        """
                        Compute c_v coherence for various number of topics

                        Parameters:
                        ----------
                        dictionary : Gensim dictionary
                        corpus : Gensim corpus
                        texts : List of input texts
                        limit : Max num of topics

                        Returns:
                        -------
                        model_list : List of LDA topic models
                        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
                        """
                        coherence_values = []
                        model_list = []
                        for num_topics in range(start, limit, step):
                            model=Lda(corpus=corpus, id2word=dictionary, num_topics=num_topics)
                            model_list.append(model)
                            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                            coherence_values.append(coherencemodel.get_coherence())

                        return model_list, coherence_values
                    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=doc_clean, start=2, limit=10, step=1)
                    # Show graph


                    limit=10; start=2; step=1;
                    x = range(start, limit, step)
                    plt.plot(x, coherence_values)
                    plt.xlabel("Num Topics")
                    plt.ylabel("Coherence score")
                    plt.legend(("coherence_values"), loc='best')
                    plt.show()

                    #import pyLDAvis.gensim - doesn't work
                    lda_display = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, sort_topics=False)
                    pyLDAvis.display(lda_display)
                    #lda_display
                    #st.write(lda_display)
                    html_string = pyLDAvis.prepared_data_to_html(lda_display)
                    #from streamlit import components
                    import streamlit.components.v1 as components
                    st.components.v1.html(html_string, width=1300, height=800, scrolling=True)
                    #st.html(html_string)


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
                                    topic_keywords = ", ".join([word for word, prop in wp])
                                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                                else:
                                    break
                        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

                        # Add original text to the end of the output
                        contents = pd.Series(texts)
                        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
                        return(sent_topics_df)


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

    elif choice =="Document/Files":
        st.subheader("Upload your Document/Files")
        docx_file = st.file_uploader("Upload Document",
        type=["pdf","docx","txt"])
        if st.button("Process"):
            if docx_file is not None:
                file_details = {"filename":docx_file.name,
                "filetype":docx_file.type,"filesize":docx_file.size}
                st.write(file_details)
                if docx_file.type == "text/plain":
                    # Read as bytes
                    #raw_text = docx_file.read()
                    #st.write(raw_text) # works but in bytes
                    #st.text(raw_text,"utf-8") # does work as expected

                    # Read as string (decode bytes as to string)
                    raw_text = str(docx_file.read(),"utf-8") # works
                    #st.write(raw_text) # works
                    st.text(raw_text) # works
                elif docx_file.type == "application/pdf":
                    # try:
                    #     with pdfplumber.open(docx_file) as pdf:
                    #         pages = pdf.pages[0]
                    #         st.write(pages.extract_text())
                    # except:
                    #     st.write("None")
                    #using PyPDF
                    raw_text = read_pdf(docx_file)
                    st.write(raw_text)

                else:
                    raw_text = docx2txt.process(docx_file)
                    st.write(raw_text) # works

                    st.text(raw_text) # works


    elif choice == "ContactUs" :
        st.header("ContactUs")
        st.subheader("For any queries reach out to us:")
        st.info("""
        Pradeep  | Senior Manager Global Analytics
        E-mail: pradeep.p@concentrix.com""")

        st.info("""
        Kavitha A | Senior Data Miner, Global Analytics
        E-mail: kavitha.a@concentrix.com
        """)

    else:
        st.header("About")
        st.subheader("Text Analyzer Tool Development")
        st.info("""VADER Sentiment Analysis
                   VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and works well on texts from other domains.""")

        st.info("""LDA Topic Modelling
                   LDA (Latent Dirichlet Allocation) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.""")
        st.subheader(" Tool Usage")

    # menu2 = ["LDA", "LSA"]
    # choice2 = st.sidebar.selectbox("Topic Modelling",menu2)
    # if choice2 =="LDA":
    #     st.subheader("Topic Modelling")
    #     data_file2 = st.file_uploader("Upload CSV",type=["csv"])
    #
    #     if data_file2 is not None:
    #         st.write(type(data_file2))
    #         #Methods/Attributes
    #         #st.write(dir(image_file))
    #         file_details2 = {"filename":data_file2.name,
    #         "filetype":data_file2.type,"filesize":data_file2.size}
    #         st.write(file_details2)
    #         df2 = pd.read_csv(data_file2)
    #         st.dataframe(df2)



if __name__ == '__main__':
    # import warnings
    # warnings.warn("use 'python -m nltk', not 'python -m nltk.downloader'",         DeprecationWarning)
    # app.run_server(debug=True)
    main()
