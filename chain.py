from langchain.chat_models import ChatOpenAI
from lanfchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SimpleSequencialChain, SequencialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser


llm = ChatOpenAI(temperture=0.9, model='gpt-3.5-turbo')
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}")

def single_chain(llm, prompt):
    chain = LLMChain(LLM=llm, prompt=prompt)
    print(chain.run("Queen Size Sheet Set"))


def single_sequencial_chain(llm):

    first_prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}")
    chain_one = single_chain(llm, first_prompt)

    second_prompt = ChatPromptTemplate.from_template(
        "Write 20 words description for the following company: {company_name}")
    chain_two = single_chain(llm, second_prompt)

    overall_simple_chain = SimpleSequencialChain(chains=[chain_one, chain_two],
                                                verbose=True)
    print(overall_simple_chain.run(product="Queen Size Sheet Set"))


def sequencial_chain():
    first_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to english:"
        "\n\n{Review}")
    chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

    seconf_prompt = ChatPromptTemplate.from_template(
        "Can you summerize the following review in 1 sentence:"
        "\n\n{English_Review}")
    chain_two = LLMChain(llm=llm, prompt=seconf_prompt, output_key="summary")

    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review: \n\n{Review}")
    chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="labguage")

    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following summary in the specified language:"
        "\n\n Summary:{summary}\n\nLanguage:{language}")
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")
    
    overall_chain = SequencialChain(
        chains=[chain_one, chain_two, chain_three, chain_four])
    print(overall_chain.run("Sample Review"))

def router_chain():
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""


    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:
    {input}"""


    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

    prompt_infos = [
        {
            "name": "physics", 
            "description": "Good for answering questions about physics", 
            "prompt_template": physics_template
        },
        {
            "name": "math", 
            "description": "Good for answering math questions", 
            "prompt_template": math_template
        },
        {
            "name": "History", 
            "description": "Good for answering history questions", 
            "prompt_template": history_template
        },
        {
            "name": "computer science", 
            "description": "Good for answering computer science questions", 
            "prompt_template": computerscience_template
        }
    ]

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain  
        
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    
