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
from langchain_experimental.tools.python.tool import PythonREPLTool, PythonAstREPLTool
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from streamlit_chat import message
from functions import find_match, query_refiner, get_conversation_string


# Main
st.title("Data Lens 🔍: Your AI-powered data analysis assistant")
st.write(
    """Upload your data and let Data Lens reveal hidden insights, trends, and patterns 📈 with the power of AI. 
    Get basic data descriptions, explore specific attributes in depth, query your data with ease, and even perform machine learning - all in one place."""
)

# * -> Italics
# ** -> Bold
with st.sidebar:
    st.write("<p style='text-align:center'>Data Lens</p>", unsafe_allow_html=True)
    st.markdown(
        """ 
    _Effortlessly explore, understand, and model your data._

    * **Comprehensive Exploration:**  Kickstart your analysis with automated EDA, revealing key insights and patterns through interactive visualizations.
    * **Intuitive Querying:** Ask questions about your data in plain language and get instant answers, no coding required. 
    * **Predictive Power:**  Seamlessly transition from data understanding to predictive modeling. Data Lens translates your business objectives into robust machine learning solutions.
    * **AI-Powered Chatbot:**  Have questions about your specific documents? Our new chatbot leverages cutting-edge vector databases to provide accurate and context-aware answers. 

    Data Lens empowers you to unlock the full potential of your data – all in one place.
    """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "<p style='text-align:center'>Developed by Vijay V ✨</p>",
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
    tab1, tab2 = st.tabs(["Data Analysis and Machine Learning", "Chatbot"])

    with tab1:

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

                missing_values = pandas_agent.run(
                    "Check for missing values in the dataset"
                )
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

            # @st.cache_data
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
                    verbose=True,
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

            # with st.sidebar:
            #     with st.expander("Steps of EDA"):
            #         st.write(steps_eda())

            function_agent()

            st.subheader("Attribute Information")
            user_question_attribute = st.text_input(
                "What attribute are you interested in?"
            )
            if user_question_attribute is not None and user_question_attribute != "":
                function_question_attribute()

                st.subheader("Further Study")

            if user_question_attribute:
                user_question_dataframe = st.text_input(
                    "Is there anything else that you want to know about the data?",
                    placeholder="Type here...",
                )
                if (
                    user_question_dataframe is not None
                    and user_question_dataframe
                    not in (
                        "no",
                        "No",
                        "",
                    )
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

    with tab2:
        st.header("Chatbot")
        st.write("Chatbot Assistant for your data")

        st.write("")

        if "responses" not in st.session_state:
            st.session_state["responses"] = ["Hello! How can I help you today?"]
        if "requests" not in st.session_state:
            st.session_state["requests"] = []

        # Initializing the LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Storing in memory the previous 3 messages for context
        if "buffer_memory" not in st.session_state:
            st.session_state.buffer_memory = ConversationBufferWindowMemory(
                k=3, return_messages=True
            )

        system_message_template = SystemMessagePromptTemplate.from_template(
            template="""Answer the question to the best of your abilities using the provided context, 
        and if the answer is not contained within the text below, say 'I don't know' """
        )
        human_message_template = HumanMessagePromptTemplate.from_template(
            template="{input}"
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                system_message_template,
                MessagesPlaceholder(variable_name="history"),
                human_message_template,
            ]
        )

        conversation = ConversationChain(
            memory=st.session_state.buffer_memory,
            verbose=True,
            llm=llm,
            prompt=prompt_template,
        )

        response_container = st.container()
        text_container = st.container()

        with text_container:
            query = st.text_input("", key="input")
            if query:
                with st.spinner("Synthesizing response..."):
                    conversation_string = get_conversation_string()
                    refined_query = query_refiner(conversation_string, query)
                    # st.subheader("Refined Query:")
                    # st.write(refined_query)
                    context = find_match(refined_query)
                    response = conversation.predict(
                        input=f"Context:\n {context} \n\n Query: {query}"
                    )
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

        with response_container:
            if st.session_state["responses"]:
                for i in range(len(st.session_state["responses"])):
                    message(
                        st.session_state["responses"][i],
                        key=str(i),
                    )
                    if i < len(st.session_state["requests"]):
                        message(
                            st.session_state["requests"][i],
                            is_user=True,
                            key=str(i) + "_user",
                        )
