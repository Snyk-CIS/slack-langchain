#!/usr/bin/env python3
'''
Start up
'''
import logging
import asyncio
from slackbot import slack_bot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def main():
    '''
    Start the bot
    '''
    await slack_bot.start()

if __name__ == "__main__":
    asyncio.run(main())
