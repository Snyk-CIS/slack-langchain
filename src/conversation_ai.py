'''
AI Class
'''
import os
import asyncio
from typing import List
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain.globals import set_debug
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
        PromptTemplate,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
)
from langchain.callbacks.manager import (
        AsyncCallbackManager,
        CallbackManager,
        collect_runs
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import LanguageModelLike
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from slack_sdk import WebClient
from callback_handler import AsyncStreamingSlackCallbackHandler
from prompts import (HUMAN_PROMPT_TEMPLATE,
                     REPHRASE_TEMPLATE)
from chatapp import Chatapp, Config

def format_docs(docs: List):
    '''
    Format the docs
    '''
    return "\n\n".join([d.page_content for d in docs])

def create_rephrase_chain(llm) -> Runnable:
    '''
    Create the rephrase chain
    '''
    rephrase_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    rephrase_question_chain = (
        rephrase_question_prompt | llm | StrOutputParser()
    ).with_config(
        run_name="RephraseQuestion",
    )
    return RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("history"))).with_config(
                    run_name="HasChatHistoryCheck"),
                RunnablePassthrough.assign(
                    input=rephrase_question_chain.with_config(
                        run_name="RephraseQuestionWithHistory"))
            ),
            (
                RunnablePassthrough()
                .with_config(run_name="RephraseQuestionNoHistory")
            )
    ).with_config(run_name="RouteDependingOnChatHistory")

def create_retriever_chain(
        retriever: BaseRetriever,
        ) -> Runnable:
    '''
    Create the retriever chain
    '''
    return (
            RunnableLambda(itemgetter("input")).with_config(
                run_name="Itemgetter:input"
            )
            | retriever).with_config(run_name="RetrievalChain")

def create_welcome_message(
        bot_user_name,
        bot_user_id,
        user_name,
        user_id,
        profile):
    '''
    Generate a welcome message
    '''
    llm_gpt3_turbo = ChatOpenAI(
            temperature=1,
            model_name="gpt-3.5-turbo",
            request_timeout=30,
            max_retries=5,
            verbose=True
            )
    prompt = PromptTemplate.from_template(os.environ.get("WELCOME_PROMPT"))
    print(prompt)
    chain = prompt | llm_gpt3_turbo
    output = chain.invoke({
        "bot_name": bot_user_name,
        "bot_user_id": bot_user_id,
        "user_name": user_name,
        "user_id": user_id,
        "user_title": profile.get("title"),
        "status_emoji": profile.get("status_emoji"),
        "status_text": profile.get("status_text")})
    return output.content

# pylint: disable=too-many-instance-attributes
class ConversationAI(Chatapp):
    '''
    Conversation class
    '''
    def __init__(
        self,
        bot_name:str,
        slack_client:WebClient,
        existing_thread_history=None,
    ):
        self.bot_name = bot_name
        self.existing_thread_history = existing_thread_history
        self.agent = None
        self.memory = None
        self.callback_handler = None
        self.slack_client = slack_client
        self.lock = asyncio.Lock()
        self.run_ids = {}

        super().__init__(Config())

        set_debug(self.config.langchain_debug)

    def init_model(self) -> LanguageModelLike:
        '''
        Configure the LLM
        '''
        # pylint: disable=not-callable
        if self.model_type == 'openai':
            llm = ChatOpenAI(
                    model_name = self.openai.openai_model_name,
                    temperature = self.temperature,
                    request_timeout = 60,
                    max_retries = 3,
                    streaming = True,
                    verbose = True,
                    callback_manager = AsyncCallbackManager([self.callback_handler])
            )
        else:
            llm = Ollama(
                    model = "llama2",
                    verbose = True,
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
            )
        return llm

    def init_rephrase_llm(self) -> LanguageModelLike:
        '''
        Configure the LLM
        '''
        llm = ChatOpenAI(
                model_name = self.openai.openai_model_name,
                temperature = self.temperature,
                request_timeout = 60,
                max_retries = 3,
                streaming = True,
                verbose = True)
        return llm

    def init_embeddings(self) -> Embeddings:
        '''
        Initialise the embedding engine
        '''
        if self.vector.embed_type == 'openai':
            embeddings = OpenAIEmbeddings(
                    openai_api_key = self.openai.openai_api_key
                    )
        else:
            embeddings = GPT4AllEmbeddings()
        return embeddings

    def init_retriever(self) -> BaseRetriever:
        '''
        Initialise the vector and retriever
        '''
        # pylint: disable=not-callable
        if self.vector.vector_type == 'chroma':
            chroma_db_path = "vectorstores/db/"
            vectorstore = Chroma(
                    collection_name = self.vector.collection_name,
                    embedding_function = self.init_embeddings(),
                    persist_directory = chroma_db_path)
        if self.vector.vector_type == 'pinecone':
            vectorstore = Pinecone(
                    embedding = self.init_embeddings(),
                    index_name=self.vector.collection_name)
        else:
            vectorstore = PGVector(
                    collection_name = self.vector.collection_name,
                    connection_string = self.vector.database_url,
                    embedding_function = self.init_embeddings(),
            )
        return vectorstore.as_retriever(
                  search_kwargs = {'k': 30}
              )

    async def create_agent(self, sender_user_info):
        '''
        Create the AI
        '''
        print(f"Creating new ConversationAI for {self.bot_name}")

        sender_profile = sender_user_info["profile"]

        print(f"Will use model: {self.openai.openai_model_name}")
        print(f"Will use temperature: {self.temperature}")
        model_facts = f"You are based on the OpenAI model {self.openai.openai_model_name}. \
                Your 'creativity temperature' is set to {self.temperature}."


        prompt = ChatPromptTemplate.from_messages(
            [
                ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    ]).partial(
                   bot_name = self.bot_name,
                   model_facts = model_facts
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
        self.callback_handler = AsyncStreamingSlackCallbackHandler(self.slack_client)

        if self.config.enable_rephrase:
            rephrase_llm = self.init_rephrase_llm()
            rephrase_chain = create_rephrase_chain(rephrase_llm)
        else:
            rephrase_chain = RunnablePassthrough().with_config(run_name="RephraseQuestionDisabled")

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
        # Unfortunately, we can't use the human's name
        # because the memory doesn't seem to support that yet
        existing_thread_history = self.existing_thread_history
        if existing_thread_history is not None:
            for message in existing_thread_history:
                # get the first key which is the name (assuming only one key per dictionary)
                sender_name = list(message.keys())[0]
                # get the first value which is the message content
                message_content = list(message.values())[0]
                if sender_name == "bot":
                    memory.chat_memory.add_ai_message(message_content)
                else:
                    memory.chat_memory.add_user_message(message_content)

        self.memory = memory

        retriever_chain = create_retriever_chain(
                self.init_retriever()
                )

        history = (
                RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
                .with_config(run_name="AddHistory")
        )
        context = (
                RunnablePassthrough.assign(docs=retriever_chain)
                .assign(context=lambda x: format_docs(x["docs"]))
                .with_config(run_name="RetrieveDocs")
        )

        self.agent = (
        history
        | rephrase_chain
        | context
        | prompt
        | self.init_model()
        | StrOutputParser()
        )
        return self.agent

    async def get_or_create_agent(self, sender_user_info):
        '''
        Get or create the agent
        '''
        if self.agent is None:
            self.agent = await self.create_agent(sender_user_info)
        return self.agent

    async def respond(self,
                      sender_user_info,
                      channel_id:str,
                      thread_ts:str,
                      message:str):
        '''
        Respond to a message
        '''
        async with self.lock:
            await self.get_or_create_agent(sender_user_info)
            print("Starting response...")
            await self.callback_handler.start_new_response(channel_id, thread_ts)
            with collect_runs() as cb:
                response = await self.agent.ainvoke({"input": message})
                run_id = cb.traced_runs[0].id
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response)
            if run_id:
                # Store ts to run_id mapping for feedback
                self.run_ids[self.callback_handler.last_ts] = run_id
            return response
