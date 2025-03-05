import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

S2_API_KEY = os.getenv("S2_API_KEY")
def sort_ideas(
        idea_1: tuple[str, any],
        idea_2: tuple[str, any],
        client,
        model
):
    evaluate_prompt = """
You are an expert evaluator tasked with comparing two hypotheses.
Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one
is superior based on the specified quality, novelty, and feasibility.

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

First, generate a list of pro's and cons for each hypothesis regarding quality, novelty, and feasibility.
The format should be as follows:

## Hypothesis 1:
Quality pros:
* ...

Quality cons:
* ...

Feasibility pros:
* ...

...

## Hypothesis 2:
Quality pros:
* ...
...

Last, concisely reason through which hypothesis is better. 

End with "better hypothesis: <1 or 2>"
"""

    text, msg_history = get_response_from_llm(
        evaluate_prompt.format(hypothesis_1=idea_1[0], hypothesis_2=idea_2[0]),
        client=client,
        model=model,
        system_message=""
    )

    last_line = text.split("\n")[-1]
    if "1" in last_line:
        return idea_1[1], idea_2[1]
    elif "2" in last_line:
        return idea_2[1], idea_1[1]
    else:
        return None, None


