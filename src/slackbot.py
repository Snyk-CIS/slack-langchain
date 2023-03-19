#!/usr/bin/env python3


import logging
import os
import asyncio
from slack_sdk.errors import SlackApiError
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from dotenv import load_dotenv
from pprint import pprint
import re
from pathlib import Path

from ConversationAI import ConversationAI

import modal

# Get the folder this file is in:
this_file_folder = os.path.dirname(os.path.realpath(__file__))
# Get the parent folder of this file's folder:
parent_folder = os.path.dirname(this_file_folder)


image = modal.Image.debian_slim().pip_install(
    "openai",
    "slack_bolt",
    "python-dotenv",
    "langchain==0.0.115"
)

stub = modal.Stub(
    name="slackbot",
    image=image,
    secrets=[modal.Secret.from_name("slackbot_ynai")],
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv(Path(parent_folder) / ".env")

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get('SLACK_APP_TOKEN')
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

class SlackBot:
    def __init__(self, slack_app):
        self.threads_bot_is_participating_in = {}
        self.app = slack_app
        self.client = self.app.client
        self.id_to_name_cache = {}

    async def start(self):
        print("Looking up bot user_id. (If this fails, something is wrong with the auth)")
        response = await self.app.client.auth_test()
        self.bot_user_id = response["user_id"]
        print("Bot user id: ", self.bot_user_id)

        self.bot_user_name = await self.get_mention_username(self.bot_user_id)

        await AsyncSocketModeHandler(app, SLACK_APP_TOKEN).start_async()

    async def get_mention_username(self, user_id):
        ret_val = self.id_to_name_cache.get(user_id, None)
        if ret_val is not None:
            return ret_val
        
        print("Getting username for user_id: ", user_id)
        user_info_response = await self.app.client.users_info(user=user_id)
        user_info = user_info_response['user']
        ret_val = None
        if (user_info['is_bot']):
            ret_val = user_info_response['user']['profile']['real_name']
        else:
            ret_val = user_info_response['user']['profile']['display_name']
        self.id_to_name_cache[user_id] = ret_val
        return ret_val

    async def reply_to_slack(self, channel_id, thread_ts, response):
        await self.client.chat_postMessage(channel=channel_id, text=response, thread_ts=thread_ts)

    async def confirm_message_received(self, channel, thread_ts, user_id_of_sender):
        # React to the message with a thinking face emoji:
        await self.client.reactions_add(channel=channel, name="thinking_face", timestamp=thread_ts)
        #await self.reply_to_slack(channel, thread_ts, ":thinking_face: ...")

    async def respond_to_message(self, channel_id, thread_ts, user_id, text):
        try:
            conversation_ai = self.threads_bot_is_participating_in.get(thread_ts, None)
            if conversation_ai is None:
                raise Exception("No AI found for thread_ts")
            text = await self.clean_up_message(text)
            sender_username = await self.get_mention_username(user_id)
            response = conversation_ai.respond(sender_username, text)
            await self.reply_to_slack(channel_id, thread_ts, response)
        except Exception as e:
            response = f":exclamation::exclamation::exclamation: Error: {e}"
            await self.reply_to_slack(channel_id, thread_ts, response)

    @staticmethod
    def is_parent_thread_message(message_ts, thread_ts):
        return message_ts == thread_ts

    async def clean_up_message(self, text):
        # Replace every @mention of a user id with their actual name:
        # First, use a regex to find @mentions that look like <@U123456789>:
        matches = re.findall(r"<@(U[A-Z0-9]+)>", text)
        for match in matches:
            mention_string = f"<@{match}>"
            mention_name = await self.get_mention_username(match)
            if mention_name is not None:
                text = text.replace(mention_string, "@"+mention_name)

        return text

    async def add_ai_to_thread(self, channel_id, thread_ts, message_ts):
        if thread_ts in self.threads_bot_is_participating_in:
            return


        processed_history = None
        # Is this thread_ts the very first message in the thread? If so, we need to create a new AI for it.
        if not self.is_parent_thread_message(message_ts, thread_ts):
            print("It looks like I am not the first message in the thread. I should get the full thread history from Slack and add it to my memory.")
            # This is not the very first message in the thread
            # We should figure out a way to boostrap the memory:
            # Get the full thread history from Slack:
            thread_history = await client.conversations_replies(channel=channel_id, ts=thread_ts)
            # Iterate through the thread history, adding each of these to the ai_memory:
            processed_history = []
            for message in thread_history.data['messages']:
                text = message['text']
                text = await self.clean_up_message(text)
                user_id = message['user']
                #user_name = await self.get_mention_username(user_id)
                #print(f"Adding message: '{user_name}: {text}")
                if (user_id == self.bot_user_id):
                    processed_history.append({"bot":text})
                else:
                    processed_history.append({"human":text})

        ai = ConversationAI(self.bot_user_name, processed_history)
        self.threads_bot_is_participating_in[thread_ts] = ai


    def is_ai_participating_in_thread(self, thread_ts, message_ts):
        if thread_ts in self.threads_bot_is_participating_in:
            return True
        return False

    async def on_app_mention(self, payload, say):
        # If the user mentions us AND we are not already participating in the thread, we should add ourselves to the thread:
        # However, if we are already participating in the thread, we should ignore this message.
        print(f"Received mention event: {payload}")

        thread_ts = payload.get('thread_ts') or payload.get('ts')
        message_ts = payload.get('ts')
        channel_id = payload.get('channel')
        if self.is_ai_participating_in_thread(thread_ts, message_ts):
            # We're already in this thread - nothing to do. The other handler will take it from here!
            return
        
        # Let's add ourselves to this thread and pass the message to the other handler:
        await self.add_ai_to_thread(channel_id, thread_ts, message_ts)
        await self.on_message(payload, say)

    async def on_message(self, event, say):
        message_ts = event['ts']
        thread_ts = event.get('thread_ts', message_ts) 
        try:
            print(f"Received message event: {event}")
            # At first I thought we weren't told about our own messages, but I don't think that's true. Let's make sure we aren't hearing about our own:
            if event['user'] == self.bot_user_id:
                print("Not handling message event since I sent the message.")
                return
            #if 'thread_ts' in event and 'subtype' not in event:

            # And are we participating in it?
            if thread_ts in self.threads_bot_is_participating_in:
                channel_id = event['channel']
                user_id = event['user']
                text = event['text']
                await self.confirm_message_received(channel_id, message_ts, user_id)
                await self.respond_to_message(channel_id, thread_ts, user_id, text)
        except Exception as e:
            response = f":exclamation::exclamation::exclamation: Error: {e}"
            await say(text=response, thread_ts=thread_ts)

app = AsyncApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
client = app.client
slack_bot = SlackBot(app)

@app.event('app_mention')
async def on_app_mention(payload, say):
    await slack_bot.on_app_mention(payload, say)

@app.event("message")
async def on_message(payload, say):
    await slack_bot.on_message(payload, say)

async def start():
    await slack_bot.start()

if __name__ == "__main__":
    asyncio.run(start())


@stub.webhook(
    method="GET"
)
def status(request):
    return "Good!"

@stub.local_entrypoint
def main():
    asyncio.run(start())