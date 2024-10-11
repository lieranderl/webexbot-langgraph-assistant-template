import os
from dotenv import load_dotenv
from webex_bot.webex_bot import WebexBot  # type: ignore
from .commands import OpenAI

# (Optional) Proxy configuration
# Supports https or wss proxy, wss prioritized.
# proxies = {
#     'https': 'http://proxy.esl.example.com:80',
#     'wss': 'socks5://proxy.esl.example.com:1080'
# }

# Load environment variables
load_dotenv()

# Create a Bot Object
bot = WebexBot(
    teams_bot_token=os.getenv("WEBEX_TEAMS_ACCESS_TOKEN"),
    # approved_rooms=['06586d8d-6aad-4201-9a69-0bf9eeb5766e'],
    # approved_users=[os.getenv("WEBEX_TEAMS_USER_ID")],
    approved_domains=[os.getenv("WEBEX_TEAMS_DOMAIN")],
    bot_name="AI-Assistant",
    bot_help_subtitle="",
    threads=False,
    help_command=OpenAI(),
)
# run the bot
bot.run()
