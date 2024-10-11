import os
# import threading
from dotenv import load_dotenv
from webex_bot.webex_bot import WebexBot
from src.webexbot.commands import OpenAI, UserAuth


user_tokens = {}

# (Optional) Proxy configuration
# Supports https or wss proxy, wss prioritized.
# proxies = {
#     'https': 'http://proxy.esl.example.com:80',
#     'wss': 'socks5://proxy.esl.example.com:1080'
# }

load_dotenv()

# Create a Bot Object
bot = WebexBot(
    teams_bot_token=os.getenv("WEBEX_TEAMS_ACCESS_TOKEN"),
    # approved_rooms=['06586d8d-6aad-4201-9a69-0bf9eeb5766e'],
    # approved_users=[os.getenv("WEBEX_TEAMS_USER_ID")],
    bot_name="AI-Assistant",
    bot_help_subtitle="",
    threads=False,
    help_command=OpenAI(),
)
# run the bot
bot.run()

