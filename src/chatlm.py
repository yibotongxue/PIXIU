import os
import asyncio
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils


async def _call_api(client, model, msg, max_tokens, temperature, semaphore):
    """Single API call with semaphore for concurrency control."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": msg}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


async def _call_api_with_progress(client, model, msg, max_tokens, temperature, semaphore, pbar):
    """Ensure the progress bar advances whenever a request finishes."""
    try:
        return await _call_api(client, model, msg, max_tokens, temperature, semaphore)
    finally:
        pbar.update(1)


async def oa_completion(**kwargs):
    """Query OpenAI API for completion using async openai library with tqdm progress."""
    from tqdm.auto import tqdm

    client = kwargs["client"]
    messages = kwargs["messages"]
    model = kwargs["model"]
    max_tokens = kwargs["max_tokens"]
    temperature = kwargs["temperature"]
    max_concurrent = kwargs.get("max_concurrent", 20)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create progress bar
    pbar = tqdm(total=len(messages), desc="API Requests", leave=False)

    try:
        # Create all tasks explicitly so gather preserves order while progress is tracked per completion.
        tasks = [
            asyncio.create_task(
                _call_api_with_progress(
                    client,
                    model,
                    msg,
                    max_tokens,
                    temperature,
                    semaphore,
                    pbar,
                )
            )
            for msg in messages
        ]

        # Wait for every request so the progress bar can reach completion even if some requests fail.
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        pbar.close()

    exceptions = [result for result in results if isinstance(result, Exception)]
    if exceptions:
        raise exceptions[0]

    return results


def run_async(coro):
    """Run async coroutine from synchronous code."""
    return asyncio.run(coro)


class ChatLM(BaseLM):

    def __init__(self, model, truncate=False, base_url=None, max_gen_toks=256, temperature=0.0, max_concurrent=20, timeout=600):
        """

        :param model: str
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        :param base_url: str
            Base URL for OpenAI API compatible server (e.g., http://localhost:8000/v1)
        :param max_gen_toks: int
            Maximum tokens to generate
        :param max_concurrent: int
            Maximum number of concurrent API requests
        :param timeout: int
            Timeout in seconds for API requests (default 600)
        """
        super().__init__()

        from openai import AsyncOpenAI

        self.model = model
        self.truncate = truncate
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self._max_gen_toks = max_gen_toks
        self._temperature = temperature
        self._max_concurrent = max_concurrent
        self._timeout = timeout

        # Check if using local vLLM server
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            # For local vLLM server, use dummy API key
            api_key = "EMPTY"
        else:
            # For OpenAI API
            api_key = os.environ.get("OPENAI_API_SECRET_KEY", "")

        # Create async openai client with timeout
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=self._timeout)

        # Use a dummy tokenizer for API-based models
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # Note: the OpenAI API supports up to 2049 tokens, with the first token being the first input token
        return 4096

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def temperature(self):
        return self._temperature

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        raise NotImplementedError()

    def greedy_until(self, requests):
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        # Collect all inputs
        all_inputs = [context[0] for context in re_ord.get_reordered()]

        # Process all requests with async concurrency control
        if all_inputs:
            responses = run_async(oa_completion(
                client=self.client,
                model=self.model,
                messages=all_inputs,
                max_tokens=self.max_gen_toks,
                temperature=self.temperature,
                max_concurrent=self._max_concurrent,
            ))

            for resp, context in zip(responses, re_ord.get_reordered()):
                s = resp

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, "</s>"), s)

                res.append(s)

        return re_ord.get_original(res)

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
