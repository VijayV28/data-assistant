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

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain, SimpleSequentialChain
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

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
            return steps.content

        # Initializing the agent
        pandas_agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True
        )

        # Main Functions
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")

            st.write("Sample from the data:")
            st.write(df.head())

            st.write("**Data Description**")
            columns_meaning = pandas_agent.run(
                "What is the meaning of the columns in the dataset?"
            )
            st.write(columns_meaning)

            missing_values = pandas_agent.run("Check for missing values in the dataset")
            st.write(missing_values)

            duplicates = pandas_agent.run("Check for duplicates in the dataset")
            st.write(duplicates)

            st.write("**Data Summary**")
            st.write(df.describe())

            correlation = pandas_agent.run(
                "What is the correlation between the columns in the dataset? Do not create any charts."
            )
            st.write(correlation)

            outliers = pandas_agent.run("Check for outliers in the dataset")
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
                f"Check if {user_question_attribute} is normally distributed. Do not create any charts."
            )
            st.write(normality)

            outliers = pandas_agent.run(
                f"Check for outliers outliers in {user_question_attribute}"
            )
            st.write(outliers)

            trends = pandas_agent.run(
                f"What are the trends in {user_question_attribute}? Do not create any charts."
            )
            st.write(trends)

            missing_values = pandas_agent.run(
                f"How many missing values  are there in {user_question_attribute}?"
            )
            st.write(missing_values)

            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        @st.cache_data
        def google(prompt):
            google = GoogleSearchAPIWrapper()
            response = google.run(prompt)
            return response

        @st.cache_data
        def prompt_templates():
            data_problem_template = PromptTemplate(
                input_variables=["business_problem"],
                template="Convert the following business problem into a data problem: {business_problem}",
            )
            model_selection_template = PromptTemplate(
                input_variables=["data_problem", "google_research"],
                template="Give a list of ML algorithms that are suitable to solve this problem: {data_problem}, while using this google research: {google_research}",
            )
            return data_problem_template, model_selection_template

        @st.cache_resource
        def chains():
            data_problem_prompt, model_selection_prompt = prompt_templates()
            data_problem_chain = LLMChain(
                llm=llm,
                prompt=data_problem_prompt,
                verbose=True,
                output_key="data_problem",
            )
            model_selection_chain = LLMChain(
                llm=llm,
                prompt=model_selection_prompt,
                verbose=True,
                output_key="model_selection",
            )
            sequential_chain = SequentialChain(
                chains=[data_problem_chain, model_selection_chain],
                input_variables=["business_problem", "google_research"],
                output_variables=["data_problem", "model_selection"],
                verbose=True,
            )
            return sequential_chain

        @st.cache_data
        def chains_output(prompt, google_research):
            chain = chains()
            chain_output = chain(
                {"business_problem": prompt, "google_research": google_research}
            )
            data_problem = chain_output["data_problem"]
            model_selection_output = chain_output["model_selection"]
            return data_problem, model_selection_output

        @st.cache_data
        def list_to_selectbox(model_selection_input):
            algorithm_lines = model_selection_input.split("\n")
            # Remove additional split if needed
            algorithms = [
                algorithm.split(":")[-1].split(".")[-1].strip()
                for algorithm in algorithm_lines
                if algorithm.strip()
            ]
            algorithms.insert(0, "Select an algorithm")
            formatted_list_output = [
                f"{algorithm}" for algorithm in algorithms if algorithm
            ]
            return formatted_list_output

        @st.cache_resource
        def python_agent():
            agent_executor = create_python_agent(
                llm=llm,
                tool=PythonREPLTool(),
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                # Add allow_dangerous_code=True to allow the agent to execute code
            )
            return agent_executor

        @st.cache_data
        def python_solution(data_problem, selected_algorithm, user_csv):
            agent = python_agent()
            solution = agent.run(
                f"Write a python code to solve the following data problem: {data_problem} using the {selected_algorithm} algorithm on the dataset {user_csv}."
            )
            return solution

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
            if user_question_dataframe is not None and user_question_dataframe not in (
                "no",
                "No",
                "",
            ):
                function_question_dataframe()
            if user_question_dataframe is ("no", "No"):
                st.write("")

            if user_question_dataframe:
                st.divider()
                st.header("Machine Learning")
                st.write("Let's convert your business problem into a data problem.")

                prompt = st.text_area(
                    "What is the business problem that you would like to solve?"
                )

                if prompt:
                    google_research = google(prompt)
                    data_problem, model_selection_output = chains_output(
                        prompt, google_research
                    )

                    st.write(data_problem)
                    st.write(model_selection_output)

                    formatted_list = list_to_selectbox(model_selection_output)
                    selected_algorithm = st.selectbox(
                        "Select an algorithm", formatted_list
                    )

                    if (
                        selected_algorithm is not None
                        and selected_algorithm != "Select an algorithm"
                    ):
                        st.subheader("Solution")
                        solution = python_solution(
                            data_problem, selected_algorithm, user_csv
                        )
                        st.write(solution)
