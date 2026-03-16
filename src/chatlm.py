import os
import asyncio
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils


async def _call_api(client, model, msg, max_tokens, temperature, semaphore, max_retries=2):
    """Single API call with semaphore for concurrency control, using streaming.

    Args:
        max_retries: Maximum number of retry attempts on exception (default 2)
    """
    import traceback

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": msg}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                # Collect streamed chunks
                full_content = ""
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                return full_content
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2  # Wait 2, 4 seconds between retries
                await asyncio.sleep(wait_time)
                continue
            else:
                # All retries failed, return exception info instead of raising
                tb_str = traceback.format_exc()
                return f"Error: {type(e).__name__}: {str(e)}\nTraceback:\n{tb_str}"
    # Should not reach here, but just in case
    tb_str = traceback.format_exc()
    return f"Error: {type(last_error).__name__}: {str(last_error)}\nTraceback:\n{tb_str}"


async def _call_api_with_progress(client, model, msg, max_tokens, temperature, semaphore, pbar, max_retries=2):
    """Ensure the progress bar advances whenever a request finishes."""
    try:
        return await _call_api(client, model, msg, max_tokens, temperature, semaphore, max_retries)
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
    max_retries = kwargs.get("max_retries", 2)

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
                    max_retries,
                )
            )
            for msg in messages
        ]

        # Wait for every request so the progress bar can reach completion even if some requests fail.
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        pbar.close()

    # Do not raise any exceptions - continue evaluation with error results
    # Results that are strings contain error information, which will be handled by the evaluation
    return results


def run_async(coro):
    """Run async coroutine from synchronous code."""
    return asyncio.run(coro)


class ChatLM(BaseLM):

    def __init__(self, model, truncate=False, base_url=None, max_gen_toks=256, temperature=0.0, max_concurrent=20, timeout=600, max_retries=2):
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
        :param max_retries: int
            Maximum number of retry attempts on timeout (default 2)
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
        self._max_retries = max_retries

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
                max_retries=self._max_retries,
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
