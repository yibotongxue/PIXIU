import os
import asyncio
import numpy as np
import transformers
from lm_eval.base import BaseLM
from lm_eval import utils
from tqdm import tqdm
import time

BACKOFF_TIME = 0.1

async def oa_completion(**kwargs):
    """Query OpenAI API for completion using async openai library with concurrency control."""
    client = kwargs["client"]
    messages = kwargs["messages"]
    model = kwargs["model"]
    max_tokens = kwargs["max_tokens"]
    temperature = kwargs["temperature"]
    max_concurrent = kwargs.get("max_concurrent", 20)

    # Process in chunks to control concurrency
    results = []
    for i in range(0, len(messages), max_concurrent):
        chunk = messages[i:i + max_concurrent]
        tasks = [
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for msg in chunk
        ]
        chunk_results = await asyncio.gather(*tasks)
        results.extend([r.choices[0].message.content for r in chunk_results])
    return results


class ChatLM(BaseLM):
    REQ_CHUNK_SIZE = 20

    def __init__(self, model, truncate=False, base_url=None, max_gen_toks=256, temperature=0.0, max_concurrent=20):
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
        """
        super().__init__()

        from openai import AsyncOpenAI

        self.model = model
        self.truncate = truncate
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self._max_gen_toks = max_gen_toks
        self._temperature = temperature
        self._max_concurrent = max_concurrent

        # Check if using local vLLM server
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            # For local vLLM server, use dummy API key
            api_key = "EMPTY"
        else:
            # For OpenAI API
            api_key = os.environ.get("OPENAI_API_SECRET_KEY", "")

        # Create async openai client
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)

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

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = "</s>"
            for x in xs:
                if len(ret) >= size:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = "</s>"
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, until in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.REQ_CHUNK_SIZE))
        ):
            inps = []
            for context in chunk:
                inps.append(context[0])

            responses = asyncio.run(oa_completion(
                client=self.client,
                model=self.model,
                messages=inps,
                max_tokens=self.max_gen_toks,
                temperature=self.temperature,
                max_concurrent=self._max_concurrent,
            ))

            for resp, context in zip(responses, chunk):
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
