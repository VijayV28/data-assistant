# Importing API Keys
import os
from dotenv import find_dotenv, load_dotenv

api_keys = "../.env"
load_dotenv(api_keys)

# Importing Libraries
import streamlit as st
import pandas as pd

from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Main
st.title("Data Lens: Your AI-powered data analysis assistant")
st.write(
    "AI-powered data analysis assistant that helps you understand your data better."
)

# * -> Italics
# ** -> Bold
with st.sidebar:
    st.write(
        """*Data Lens is an AI-powered data analysis assistant that helps you understand your data better.*"""
    )
    st.caption("**Powered by OpenAI**")
    st.divider()
    st.caption(
        "<p style='text-align:center'>Developed by Vijay</p>",
        unsafe_allow_html=True,
    )

# Initializing the session state
if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}


# Function to update the session state
def clicked(button):
    st.session_state.clicked[button] = True


st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:

    # Load data
    user_csv = st.file_uploader("Upload your CSV file", type=["csv"])
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        # LLM Components
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Sidebar function
        @st.cache_data
        def steps_eda():
            steps = llm(
                "What are the steps of exploratory data analysis? Give a short and concise summary."
            )
            return steps

        # Initializing the agent
        pandas_agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True
        )

        # Main Functions
        @st.cache_data
        def function_agent():
            st.write("**Data Overview")

            st.write("Sample from the data:")
            st.write(df.head())

            st.write("**Data Description**")
            columns_meaning = pandas_agent.run(
                "What is the meaning of the columns in the dataset?"
            )
            st.write(columns_meaning)

            missing_values = pandas_agent.run(
                "Are there any missing values in the dataset?"
            )
            st.write(missing_values)

            duplicates = pandas_agent.run("Are there any duplicates in the dataset?")
            st.write(duplicates)

            st.write("**Data Summary**")
            st.write(df.describe())

            correlation = pandas_agent.run(
                "What is the correlation between the columns in the dataset? Do not create any charts."
            )
            st.write(correlation)

            outliers = pandas_agent.run("Are there any outliers in the dataset?")
            st.write(outliers)

            new_features = pandas_agent.run(
                "What new features can be created from the existing features in the dataset?"
            )
            st.write(new_features)

            return

        @st.cache_data
        def function_question_attribute():
            st.line_chart(df, y=[user_question_attribute])
            summary_statistics = pandas_agent.run(
                f"Give me a summary of the statistics of {user_question_attribute}. Do not create any charts."
            )
            st.write(summary_statistics)

            normality = pandas_agent.run(
                f"Is {user_question_attribute} normally distributed? Do not create any charts."
            )
            st.write(normality)

            outliers = pandas_agent.run(
                f"Are there any outliers in {user_question_attribute}?"
            )
            st.write(outliers)

            trends = pandas_agent.run(
                f"What are the trends in {user_question_attribute}? Do not create any charts."
            )
            st.write(trends)

            missing_values = pandas_agent.run(
                f"Are there any missing values in {user_question_attribute}?"
            )
            st.write(missing_values)

            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(
                f"{user_question_dataframe}. You are not allowed to create any charts"
            )
            st.write(dataframe_info)
            return

        # Main
        st.header("Exploratory Data Analysis")
        st.subheader("General information about the dataset")

        with st.sidebar:
            with st.expander("Steps of EDA"):
                st.write(steps_eda())

        function_agent()

        st.subheader("Attribute Information")
        user_question_attribute = st.text_input("What attribute are you interested in?")
        if user_question_attribute is not None and user_question_attribute != "":
            function_question_attribute()

            st.subheader("Further Study")

        if user_question_attribute:
            user_question_dataframe = st.text_input(
                "Is there anything else that you want to know about the data?"
            )
            if user_question_dataframe is not None and user_question_dataframe != "":
                function_question_dataframe()
            if user_question_dataframe is ("no", "No"):
                st.write("")
