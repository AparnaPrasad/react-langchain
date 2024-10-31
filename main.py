
import os
from typing import List, Union;
from dotenv import load_dotenv
from langchain.agents import tool, Tool
from langchain.tools.render import render_text_description
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.schema import  AgentAction, AgentFinish
from langchain.agents.format_scratchpad.log import format_log_to_str

from callback import AgentCallbackHandler

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    print("get_text_length called with", text)
    text = text.strip("'\n'").strip('"') #strip non alphabetic char
    return len(text)


def findTool(tools: List[Tool], tool_name: str) -> str:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError("Cannot find tool")

if __name__ == '__main__':
    load_dotenv()
    print(os.environ['OPENAI_API_KEY'])
    print('hello world2')
    tools=[get_text_length]

    template="""Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools),
                                                                 tool_names=", ".join([t.name for t in tools]))
intermediate_steps = []
llm = ChatOpenAI(temperature=0,stop=["\nObservation"], callbacks=[AgentCallbackHandler()])
agent = {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]) } | prompt | llm | ReActSingleInputOutputParser()

agent_step = ""
while not isinstance(agent_step, AgentFinish):
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of 'GOD' in characters?", "agent_scratchpad": intermediate_steps})
    print("agent_step 1", agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = findTool(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        intermediate_steps.append((agent_step, observation))
        print(f"{observation=}")

# agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of 'GOD' in characters?", "agent_scratchpad": intermediate_steps})

# print("agent_step 2", agent_step)

if isinstance(agent_step, AgentFinish):
    print(agent_step.return_values)