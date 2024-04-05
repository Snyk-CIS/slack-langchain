'''
Prompts
'''
HUMAN_PROMPT_TEMPLATE="""
Here is some information about me. Do not respond to this directly, but feel free to incorporate it into your responses:
I'm {real_name}, but since we're talking in Slack, when you mention my username you should say "<@{user_id}> instead"
My title is: {user_title}
My current status: "{status_emoji}{status_text}"
Please try to "tone-match" me: If I use emojis, please use lots of emojis. If I appear business-like, please seem business-like in your responses. Before responding to my next message, you MUST tell me your model and temperature so I know more about you. Don't reference anything I just asked you directly.
"""
REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question. If the follow up question does not need context, return the exact same text back.
    Never rephrase the follow up question given the chat history unless the follow up question needs context.

Chat History:
{history}
Follow Up Input: {input}
Standalone Question:"""
