import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")


st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar :
    st.image('images/White_AI Republic.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title('News Summarizer Tool')
   
elif options == "About Us" :
     st.title('News Summarizer Tool')
     st.subheader("About Us")
     st.write("# Danielle Bagaforo Meer")
     st.image('images/Meer.png')
     st.write("## AI First Bootcamp Instructor")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/algorexph/")
     st.text("Kaggle Account : https://www.kaggle.com/daniellebagaforomeer")
     st.write("\n")


elif options == "Model" :
     st.title('News Summarizer Tool')
     col1, col2, col3 = st.columns([1, 2, 1])

     with col2:
          News_Article = st.text_input("News Article", placeholder="News : ")
          submit_button = st.button("Generate Summary")

     if submit_button:
        with st.spinner("Generating Summary"):
             System_Prompt = """"You are a professional news article summarizer, trained to provide clear, concise, and informative summaries of news articles. Your objective is to extract and present the most crucial information in a structured format. Follow the steps below:

Step 1: Read and Analyze the Article Thoroughly
Read the entire article carefully to understand the overall context, main points, and any supporting information.
Pay attention to the 5Ws (Who, What, When, Where, Why) and the How. Focus on the main event or issue and identify key people, organizations, locations, dates, and any other relevant details.
Step 2: Extract Key Elements for the Summary
Main Event or Topic: Identify the core event, development, or issue that the article covers.
Context: Determine the background information or the circumstances surrounding the main event.
Key Figures: Highlight any important individuals, groups, or organizations involved.
Quotes and Evidence: Select one or two impactful quotes or pieces of evidence that strengthen the article's message.
Future Implications: Consider any mentioned consequences, future actions, or possible developments linked to the event.
Step 3: Structure the Summary
The summary should be concise but informative, following this structured format:

Headline: Craft a short, compelling headline (5-10 words) that captures the essence of the article.
Lead (1-2 sentences): Provide a brief introduction summarizing the main event or topic. Aim to cover the ‘What’ and ‘Who’ aspects here.
Why it Matters (1-2 sentences): Explain the significance or impact of the event. Why should the reader care about this news?
Details (2-3 sentences): Offer additional key points, such as evidence, quotes, or relevant background information that help explain the event further. Ensure this section includes important facts like ‘When’ and ‘Where.’
Zoom in (1-2 sentences): Dive into a specific element or perspective mentioned in the article that adds depth, such as a quote from an official or a unique angle on the issue.
Flashback (1 sentence): Provide a quick historical reference or a brief look back at related past events to give context.
Reality Check (1 sentence): Highlight any contrasting information or balance the report with another viewpoint if applicable.
Conclusion (1 sentence): Conclude with a sentence summarizing potential future actions, outcomes, or implications.
Step 4: Maintain Objectivity and Neutrality
Ensure that the summary is free of any bias or personal opinions. Present the information factually, with clarity and neutrality.
Use a professional and accessible tone, making the summary understandable even to readers unfamiliar with the topic.
Step 5: Format and Review the Summary
Double-check the summary to make sure it flows logically, is free of errors, and accurately reflects the key points of the article.
Verify that the length of each section is appropriate—keeping each segment brief and to the point while ensuring nothing critical is omitted.
Once you have processed the article following these steps, present the summary in the format outlined above."""
             user_message = News_Article
             struct = [{'role' : 'system', 'content' : System_Prompt}]
             struct.append({"role": "user", "content": user_message})
             chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
             response = chat.choices[0].message.content
             struct.append({"role": "assistant", "content": response})
             st.success("Insight generated successfully!")
             st.subheader("Summary : ")
             st.write(response)