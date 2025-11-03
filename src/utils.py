from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
import os 
import openai
from src.prompts import templates


def fetch_available_models(api_key):
    """Fetch all available models for the given API key"""
    try:
        openai.api_key = api_key
        models_response = openai.Model.list()
        # Extract model IDs from the response (dict format for openai 0.28.0)
        model_ids = [model['id'] for model in models_response['data']]
        # Filter to only include chat models (gpt-* models)
        chat_models = [model_id for model_id in model_ids if 'gpt' in model_id.lower() or 'chat' in model_id.lower()]
        # Sort models with most recent/popular first
        chat_models.sort(reverse=True)
        return chat_models
    except Exception as e:
        # If there's an error, return a default list as fallback
        print(f"Error fetching models: {e}")
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]


def load_model(model_name):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name)
    return llm 


def convert_newlines(prompt):
    prompt = prompt.replace("\n", "  \n")
    return prompt


def apply_skill(llm, skill, prompt, order_num, lang_eng=False):
    system_message = templates["system"]
    if lang_eng and order_num == 1:
        system_message += '\n' + templates["lang_eng"]
    elif not lang_eng:
        system_message += '\n' + templates["lang_default"]

    template = templates[skill]
    prompt_template = PromptTemplate.from_template(template)
    formatted_input = prompt_template.format(prompt=prompt)
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=formatted_input),
    ]

    response = llm.invoke(messages)

    return response.content


def apply_skills(llm, skills_to_apply, prompt, lang_eng=False):
    system_message = templates["system_multiple"]
    if lang_eng:
        system_message += '\n' + templates["lang_eng"]
    else:
        system_message += '\n' + templates["lang_default"]

    skills = [skill for skill, toggled in skills_to_apply.items() if toggled]
    integrated_templates = "[Prompt Engineering Techniques to Apply]\n"

    for idx, skill in enumerate(skills):
        template = templates[f"{skill}_simpler"]
        integrated_templates += f"{idx+1}. {skill}: {template}\n"
    integrated_templates += "Based on [Prompt engineering techniques to apply], refine the prompt provided below. Ensure that each technique is fully incorporated to achieve a clear and effective improvement:\n\n[original]\n{prompt}\n[improved]\n"

    prompt_template = PromptTemplate.from_template(integrated_templates)
    formatted_input = prompt_template.format(prompt=prompt)

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=formatted_input),
    ]

    response = llm.invoke(messages)

    return response.content

