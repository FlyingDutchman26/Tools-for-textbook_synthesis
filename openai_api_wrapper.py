import os
from typing import Dict, Any
import openai
import random
import time


# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 6,
        errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    # raise Exception(
                    #     f"Maximum number of retries ({max_retries}) exceeded."
                    # )
                    raise e

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper
# openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenaiAPIWrapper:
    @staticmethod
    def set_api_key(api_key: str):
        openai.api_key = api_key
        return

    @staticmethod
    @retry_with_exponential_backoff
    def call(prompt: str, max_tokens: int, engine: str, presence_penalty=0, temperature=0.7, top_p=1, frequency_penalty=0) -> dict:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            best_of=1
        )
        return response

    @staticmethod
    @retry_with_exponential_backoff
    def call_turbo(prompt: str, max_tokens: int, presence_penalty=0, temperature=0.7, top_p=0.9, frequency_penalty=0, stop_sequences=None, system_prompt='You are a helpful assistant.') -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
        )
        return response
    
    @staticmethod
    @retry_with_exponential_backoff
    def call_turbo_using_messages(messages, max_tokens: int, presence_penalty=0, temperature=0.7, top_p=0.9, frequency_penalty=0, stop_sequences=None) -> dict:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
        )
        return response

    @staticmethod
    def parse_response(response) -> Dict[str, Any]:
        text = response["choices"][0]["text"]
        return text

    @staticmethod
    def parse_chatgpt_response(response) -> Dict[str, Any]:
        return response["choices"][0]['message']['content']

