import streamlit as st
import os
import io
import tweepy
import webbrowser
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from wordcloud import WordCloud
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import plotly.express as px
from collections import Counter
from arabic_reshaper import reshape 
from bidi.algorithm import get_display
from io import BytesIO, StringIO
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import datetime
#pd.set_option("display.max_colwidth", -1)



def main():
	img=Image.open('zakaria.png')
	st.beta_set_page_config(page_title='Z/E', page_icon = img)

	st.title("  COVID-19 Tweets Dataset  ")
	
	st.image('covid.jpg',use_column_width=True)
	activitie=["Show row data","show random tweet","number of tweets by sentiment","sentiment tweets by country","word cloud"]
	st.sidebar.title('Data Exploration  ')
	

	choice1 = st.sidebar.selectbox("",activitie)
	folder_path="Desktop/hr/covid-19-streamlit-app/data"

	

	#uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
	#print(uploaded_file )
	#st.set_option('deprecation.showfileUploaderEncoding', False)

	
	filenames = os.listdir(folder_path)
	selected_filename = st.selectbox('Select a file', filenames)
	if selected_filename=="first_data_covid19.csv":
		st.write('with this first file the user can  : ')
		st.write('✔️ Visualise Data ')
		st.write('✔️ Random tweets Positive,Negative or Neutral ')
		st.write('✔️ Number of tweets by sentiment')
		st.write('✔️ Number of tweets by country ')
		st.write('✔️ Word cloud Positive,Negative or Neutral')
		st.write('✔️ Split data for multiclass classification ')
		st.write('✔️ Training and classify data with SVM and Naive_BAYES ')
		st.write('✔️ Plot confusion_matrics')
		st.write('❌ Split data for binary classification ')
		st.write('❌ Training and classify data with  LOGISTIC REGRESSION ')
		st.write('❌ Plot Roc Curve,Precision Recall Curve')


	else:
		st.write('with this second file the user can : ')
		st.write('✔️ Visualise Data  ')
		st.write('✔️ Random tweets Positive,Negative or Neutral ')
		st.write('✔️ Number of tweets by sentiment')
		st.write('✔️ Word cloud Positive,Negative or Neutral')
		st.write('✔️ Split data for multiclass classification ')
		st.write('✔️ Split data for binary classification ')
		st.write('✔️ Training and classify data with SVM,Naive_BAYES and LOGISTIC REGRESSION ')
		st.write('✔️ Plot confusion_matrics,Roc Curve,Precision Recall Curve')
		st.write('❌ Number of tweets by country ')

    
    
    
	
	
	

    	
   
    
	@st.cache(persist=True)
	def load_data():

		try:
			data=pd.read_csv(folder_path+'/'+selected_filename)
			return data
		except:
			pass
	
		
	data=load_data()
	if choice1  == "Show row data":
		if st.sidebar.checkbox('show raw data',False):
			st.subheader('RAW DATA')
			st.write(data.head(10000))
				
	elif choice1 == "show random tweet":
		st.sidebar.subheader('show random tweet')
		random_tweet=st.sidebar.radio('sentiment',('Positive','Neutral','Negative'))
		st.sidebar.markdown(data.query('Analysis== @random_tweet')[['text']].sample(n=1).iat[0,0])

	elif choice1 == "number of tweets by sentiment":

		st.sidebar.markdown('number of tweets by sentiment')
		select=st.sidebar.selectbox('Visualization type',['Histogram','Pie chart'],key='1')
		sentiment_count=data['Analysis'].value_counts()
		#st.write(sentiment_count)
		
		sentiment_count=pd.DataFrame({'Sentiment':sentiment_count.index,'Tweets':sentiment_count.values})
		if not st.sidebar.checkbox("Hide",True):
			st.markdown('Number of tweets by sentiment')
			if select == 'Histogram':
				fig=px.bar(sentiment_count,x='Sentiment',y='Tweets',color='Tweets')
				st.plotly_chart(fig)
			else:
				fig=px.pie(sentiment_count,values='Tweets',names='Sentiment')
				st.plotly_chart(fig)
	elif choice1 == "sentiment tweets by country":
		st.sidebar.subheader("sentiment tweets by country")
		choice_country=st.sidebar.multiselect('countries',('USA','India','UK','other','korea','Canada','Australia','France','Nigeria','Mexico','Germany','Pakistan','Switzerland','United States','United Kingdom','Ireland','Austria','Hong Kong'))
		choice_data=data[data.country.isin(choice_country)]
		try:
			fig_choice=px.histogram(choice_data,x='country',y='Analysis',histfunc='count',color='Analysis', facet_col='Analysis',labels={'Analysis':'tweets'},height=600,width=800)
			st.plotly_chart(fig_choice)
		except:
			pass 
             # Prevent the error from propagating into your Streamlit app.
	elif choice1=="word cloud":
		st.sidebar.header("word cloud")
		word_sentiment=st.sidebar.radio('Display word cloud for what sentiment ?',('Positive','Neutral','Negative'))
		if not st.sidebar.checkbox("Hide",True,key='3'):
			st.header('word cloud for %s sentiment'%(word_sentiment))
			df=data[data['Analysis']==word_sentiment]
			words=' '.join([str(tweets) for tweets in df['text']])
			wordcloud=WordCloud(stopwords=STOPWORDS,background_color='white',height=650,width=800).generate(words)
			plt.imshow(wordcloud)
			plt.xticks([])
			plt.yticks([])
			st.pyplot()
			st.set_option('deprecation.showPyplotGlobalUse', False)

	if st.sidebar.checkbox('split data for multiclass classification',False):
		def split(data):
			vectorizer=CountVectorizer()
			Xdata=data['clean_tweets'].fillna('')
			X=vectorizer.fit_transform(Xdata)
			y=data['label']
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
			st.success('shape of training data')
			st.write('',X_train.shape[0])
			st.success('shape of testing data')
			st.write('',X_test.shape[0])

			return X_train,X_test,y_train,y_test
		X_train,X_test,y_train,y_test=split(data)	


	if st.sidebar.checkbox('split data for binary classification',False):
		def splitt(data):
			vectorizer=CountVectorizer()
			Xdata=data['clean_tweets'].fillna('')
			X=vectorizer.fit_transform(Xdata)
			y=data['label2']
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
			st.success('shape of training data')
			st.write('',X_train.shape[0])
			st.success('shape of testing data')
			st.write('',X_test.shape[0])

			return X_train,X_test,y_train,y_test
		X_train,X_test,y_train,y_test=splitt(data)

#*********************************************************************************************************


	
	
		 
		#st.markdown(html_temp, unsafe_allow_html = True) 
		#html_temp = """
		#<div style ="background-color:yellow;padding:13px">
		#<h1 style ="color:black;text-align:center;">Streamlit Classifier ML App </h1> 
		#</div>
		#"""
		

	

		
	


#***********************************************************************************	


	def plot_metrics(metrics_list):
		if'confusion_matrics'in metrics_list:
			st.subheader('Confusion Matrics')
			plot_confusion_matrix(model,X_test,y_test,display_labels=class_names)
			#cm=confusion_matrix(y_test,y_predict)
			st.pyplot()

		if 'Roc Curve' in metrics_list:
			st.subheader('Roc Curve')
			plot_roc_curve(model,X_test,y_test)
			st.pyplot()
		if'Precision Recall curve'in metrics_list:
			st.subheader('Precision Recall curve')
			plot_precision_recall_curve(model,X_test,y_test)
			st.pyplot()
	class_names=['Negative','Neutral','Positive']
	class_names2=['Positive','Negative']

	st.sidebar.title('Training Data')
	st.sidebar.subheader('choose classifier')
	classifier=st.sidebar.selectbox('classifier',('Support vector machine(SVM)','Naive Bayes(NB)','Logistic Regression'))
	if classifier=='Support vector machine(SVM)':
		st.sidebar.subheader('HyperParameters')
		C=st.sidebar.number_input('C(Regularization parameter)',0.01,10.0,step=0.01,key='C')
		kernel=st.sidebar.radio("kernel",("rbf","linear"),key='kernel')
		gamma=st.sidebar.radio("Gamma (kernel coefficient)",("scale","auto"),key='gamma')
		metrics=st.sidebar.multiselect("what metrics to plot",("confusion_matrics","Roc Curve","Precision Recall curve"))
		if st.sidebar.button("Classify",key='classify'):
			st.subheader('Support Vector Machine(SVM) Results:')
			model=SVC(C=C,kernel=kernel,gamma=gamma)
			model.fit(X_train,y_train)
			accuracy=model.score(X_test,y_test)
			y_predict=model.predict(X_test)
			st.write("Accuracy",accuracy.round(2),'%')
			plot_metrics(metrics)

	if classifier=='Naive Bayes(NB)':
		st.sidebar.subheader('Naive Bayes(NB) ')
		metrics=st.sidebar.multiselect("what metrics to plot",("confusion_matrics",""))
		if st.sidebar.button("Classify",key='classify'):
			st.subheader('Naive Bayes (NB) Results:')
			
			model=MultinomialNB()
			model.fit(X_train,y_train)
			accuracy=model.score(X_test,y_test)
			y_predict=model.predict(X_test)
			st.write("Accuracy",(accuracy*100).round(2),'%')


			st.set_option('deprecation.showPyplotGlobalUse', False)
			plot_metrics(metrics)		

	if classifier=='Logistic Regression':
		st.sidebar.subheader('Logistic Regression ')
		metrics=st.sidebar.multiselect("what metrics to plot",("Roc Curve","Precision Recall curve"))
		if st.sidebar.button("Classify",key='classify'):
			st.subheader('Logistic Regression Results:')
			
			model=LogisticRegression()

			model.fit(X_train,y_train)
			accuracy=model.score(X_test,y_test)
			y_predict=model.predict(X_test)
			st.write("Accuracy",(accuracy*100).round(2),'%')


			st.set_option('deprecation.showPyplotGlobalUse', False)
			plot_metrics(metrics)	



#************************************************************************************

	




	
	page_bg_img = '''
    

    <style>
    body {
    background-image: url("")

    }
    </style>
    '''
	hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

	hide_footer_style = """


    <style>

    .reportview-container .main footer {visibility: hidden;} 

    """
    
    
	st.markdown('<style>h1{color:white ;background-color: black ;}</style>', unsafe_allow_html=True)
	st.markdown(hide_footer_style, unsafe_allow_html=True)
	st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
	st.markdown('<style>h3{color:gris;background-color: #D7BE1F;}</style>', unsafe_allow_html=True)
	st.markdown('<style>p{color:Black ;}</style>', unsafe_allow_html=True)
	st.markdown('<style>div{color:black ;}</style>', unsafe_allow_html=True)
	st.markdown('<style>div.element-container{color:yellow ;}</style>', unsafe_allow_html=True)
	st.markdown('<style>body{background-color: #fffd80;}</style>',unsafe_allow_html=True)
	st.sidebar.title('Real Time Twitter Sentiment Analysis')

	url = 'https://sentiment-analysis-rt.herokuapp.com/'
	if st.sidebar.button('Click Here'):
		webbrowser.open_new_tab(url)
    


	st.write(
        """
        <style >
        {color: Blue;}
        div[role="listbox"] 
        </style>
        """
        ,
        unsafe_allow_html=True,
    )
    


    #st.markdown(' <style> body{ color: #fff; background-color: #111; } </style> ' ,  unsafe_allow_html=True)
    	
    	

if __name__ == "__main__":
	main()
	

# Zakaria Elharchaoui
