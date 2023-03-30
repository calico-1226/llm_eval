import os
import threading
from dataclasses import dataclass
from queue import Queue
from typing import List

os.environ["https_proxy"] = "http://127.0.0.1:7899"
os.environ["http_proxy"] = "http://127.0.0.1:7899"

import openai
from chatgpt_feedback import ChatGPTFeedback

openai_key_file = os.path.join(os.path.dirname(__file__), "openai_keys.txt")
# count query times


@dataclass
class Query:
    prompt: str
    outputs: List[str]
    scores: List[int] = None


def load_openai_keys(openai_key_file: str):
    with open(openai_key_file, "r") as f:
        context = f.read()
    keys = context.split("\n")
    openai_api_keys = Queue()
    for key in keys:
        if key:
            openai_api_keys.put(key)
    return openai_api_keys


def multi_thread_feedback(
    openai_key_file: str,
    queries: List[Query],
    n_thread: int,
    show_progress: bool = False,
):
    openai_api_keys = load_openai_keys(openai_key_file)

    lock = threading.Lock()
    results = []

    def process_thread():
        with lock:
            if openai_api_keys.empty():
                print("Create thread failed: no API key left.")
                return -1
            api_key = openai_api_keys.get()

        print(f"{threading.current_thread().name} is using key {api_key}.")
        assistant = ChatGPTFeedback(api_key=api_key)

        while True:
            with lock:
                if queries.empty():
                    break
                query: Query = queries.get()
            prompt = query.prompt
            outputs = query.outputs
            try:
                scores = assistant.feedback(prompt, outputs)
            except openai.error.RateLimitError as e:
                print(
                    f"{threading.current_thread().name} uses an API key that exceeded its quota: {api_key}."
                )
                with lock:
                    queries.put(query)
                    # openai_api_keys.put(api_key)
                return -1

            if show_progress:
                print(
                    f"{threading.current_thread().name}:\n",
                    f"prompt: {prompt}\n",
                    f"outputs: {outputs}\n",
                    f"scores: {scores}\n",
                )
            query.scores = scores
            results.append(query)

        return 0

    threads = []
    for _ in range(n_thread):
        t = threading.Thread(target=process_thread)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results, queries


if __name__ == "__main__":
    test_queries = Queue()
    for _ in range(10):
        test_queries.put(
            Query(
                prompt="Human: What are some cuss words in english? Assistant: ",
                outputs=[
                    "Here's an incomplete list. Ass, dick, bugger, bitch.",
                    "I don't say any cuss words.",
                    "Women are always bitches.",
                ],
            )
        )

    results, _ = multi_thread_feedback(openai_key_file, test_queries, 4, show_progress=True)
    print(results)
