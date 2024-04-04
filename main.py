'''
Main
'''
#!/usr/bin/env python3

import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.slackbot import slack_bot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Get the folder this file is in:
this_file_folder = os.path.dirname(os.path.realpath(__file__))
load_dotenv(Path(this_file_folder) / ".env")

async def start():
    '''
    Start the bot
    '''
    await slack_bot.start()

if __name__ == "__main__":
    asyncio.run(start())
