import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS

from google import genai

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    chat = client.chats.create(model="gemini-2.0-flash")
    response = chat.send_message_stream("I have 2 dogs in my house.")
    # for chunk in response:
    #     print(chunk.text, end="")
    print(response)
    response = chat.send_message_stream("How many paws are in my house?")
    for chunk in response:
        print(chunk.text, end="")
    for message in chat._curated_history:
        print(f'role - ', message.role, end=": ")
        print(message.parts[0].text)