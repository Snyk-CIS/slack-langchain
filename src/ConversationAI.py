import asyncio
from langchain.chains import ConversationChain
from langchain.agents import Agent, Tool, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate,
                               HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from slack_sdk import WebClient

from AsyncStreamingSlackCallbackHandler import AsyncStreamingSlackCallbackHandler
from prompts import SYSTEM_PROMPT_TEMPLATE, HUMAN_PROMPT_TEMPLATE

DEFAULT_MODEL="gpt-3.5-turbo"
UPGRADE_MODEL="gpt-4"
DEFAULT_TEMPERATURE=0.0

class ConversationAI:
    def __init__(
        self,
        bot_name:str,
        slack_client:WebClient,
        existing_thread_history=None,
        model_name:str=None
    ):
        self.bot_name = bot_name
        self.existing_thread_history = existing_thread_history
        self.model_name = None
        self.agent = None
        self.model_temperature = None
        self.slack_client = slack_client
        self.lock = asyncio.Lock()

    async def create_agent(self, sender_user_info, initial_message):
        print(f"Creating new ConversationAI for {self.bot_name}")

        sender_profile = sender_user_info["profile"]
        # TODO: If we are picking up from where a previous thread left off, we shouldn't be looking at the initial message the same way, and should use the original message as the "initial message"

        self.model_name = DEFAULT_MODEL
        self.model_temperature = DEFAULT_TEMPERATURE

        print("Will use model: " + self.model_name)
        print(f"Will use temperature: {self.model_temperature}")

        model_facts = f"You are based on the OpenAI model {self.model_name}. Your 'creativity temperature' is set to {self.model_temperature}."

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    SYSTEM_PROMPT_TEMPLATE
                ).format(
                    model_facts = model_facts,
                    bot_name = self.bot_name
                ),
                HumanMessagePromptTemplate.from_template(
                    HUMAN_PROMPT_TEMPLATE
                ).format(
                    real_name = sender_profile.get("real_name"),
                    user_id = sender_user_info.get("id"),
                    status_emoji = sender_user_info.get("status_emoji"),
                    status_text = sender_user_info.get("status_text"),
                    user_title = sender_user_info.get("title"),
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
        self.callbackHandler = AsyncStreamingSlackCallbackHandler(self.slack_client)

        llm = ChatOpenAI(model_name = self.model_name,
                         temperature=self.model_temperature,
                         request_timeout=60,
                         max_retries=3,
                         streaming=True,
                         verbose=True,
                         callback_manager=AsyncCallbackManager([self.callbackHandler]))
        # This buffer memory can be set to an arbitrary buffer
        memory = ConversationBufferMemory(return_messages=True)

        # existing_thread_history is an array of objects like this:
        # {
        #     "taylor": "Hello, how are you?",
        #     "bot": "I am fine, thank you. How are you?"
        #     "kevin": "@taylor, I'm talking to you now"
        #     "taylor": "@kevin, Oh cool!"
        # }
        # We should iterate through this and add each of these to the memory:
        # Unfortunately, we can't use the human's name because the memory doesn't seem to support that yet
        existing_thread_history = self.existing_thread_history
        if existing_thread_history is not None:
            for message in existing_thread_history:
                sender_name = list(message.keys())[0] # get the first key which is the name (assuming only one key per dictionary)
                message_content = list(message.values())[0] # get the first value which is the message content
                if sender_name == "bot":
                    memory.chat_memory.add_ai_message(message_content)
                else:
                    memory.chat_memory.add_user_message(message_content)

        self.memory = memory
        self.agent = ConversationChain(
            memory=memory,
            prompt=prompt,
            llm=llm,
            verbose=True
        )
        return self.agent

    async def get_or_create_agent(self, sender_user_info, message):
        if self.agent is None:
            self.agent = await self.create_agent(sender_user_info, message)
        return self.agent

    async def respond(self, sender_user_info, channel_id:str, thread_ts:str, message_being_responded_to_ts:str, message:str):
        async with self.lock:
          agent = await self.get_or_create_agent(sender_user_info, message)
          # TODO: This is messy and needs to be refactored...
          print("Starting response...")
          await self.callbackHandler.start_new_response(channel_id, thread_ts)
          # Now that we have a handler set up, just telling it to predict is sufficient to get it to start streaming the message response...
          response = await self.agent.apredict(input=message)
          return response
