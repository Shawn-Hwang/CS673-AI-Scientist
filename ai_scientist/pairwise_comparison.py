import json
import os
import os.path as osp
import time
from typing import List, Dict, Union, Any

import backoff
import requests

from llm import get_response_from_llm, extract_json_between_markers, create_client

# S2_API_KEY = os.getenv("S2_API_KEY")
def compare_two_ideas(
        idea_1: tuple[str, Any],
        idea_2: tuple[str, Any],
        client,
        model, 
        experiment
):
    evaluate_prompt = """
        You are an expert evaluator tasked with comparing two hypotheses.
        Evaluate the two provided ideas (idea 1 and idea 2) 
        for modifying and improving the following code: 
            
        <experiment.py>
        {code}
        </experiment.py>
        Determine which idea is superior based on expected improvement in performance and efficiency.

        Idea 1:
        {idea_1}

        Idea 2:
        {idea_2}

        First, generate a list of reasons why each idea might increase or decrease performance and
        reasons it might be considered efficient or inefficient.
        The format should be as follows:

        ## Idea 1:
        Reasons for potential performance improvement:
        * ...

        Reasons for potential performance decrease:
        * ...

        Reasons for efficiency:
        * ...

        Reasons for inefficiency:
        * ...

        ...

        ## Idea 2:
        Reasons for potential performance improvement:
        * ...
        ...

        Last, concisely reason through which idea is better. 

        End with "better idea: <1 or 2>"
        """
    system_prompt = "You are a distinguished researcher in computer science, artificial intelligence, " \
                    "and deep learning with decades of experience evaluating research ideas. " \
                    "Apply your scientific judgment to critically analyze the proposed ideas. " \
                    "Consider theoretical soundness, novelty, feasibility, and potential impact. " \
                    "Think methodically and provide balanced assessments as you would when reviewing for top conferences. " \
                    "Your evaluation should reflect the nuanced thinking of an elite researcher in the field."

    with open(osp.join("templates", experiment, "experiment.py"), "r") as f:
        code = f.read()

    text, msg_history = get_response_from_llm(
        evaluate_prompt.format(idea_1=idea_1[0], idea_2=idea_2[0], code=code),
        client=client,
        model=model,
        system_message=system_prompt
    )

    last_line = text.split("\n")[-2]
    # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
    # print(last_line)
    # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
    if "1" in last_line:
        return idea_1[1], idea_2[1]
    elif "2" in last_line:
        return idea_2[1], idea_1[1]
    else:
        return None, None


