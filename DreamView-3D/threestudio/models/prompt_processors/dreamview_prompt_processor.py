import os
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, CLIPTextModel
import torch.multiprocessing as mp

import threestudio
from threestudio.models.prompt_processors.base import hash_prompt
from threestudio.utils.misc import barrier, cleanup, get_rank
from threestudio.utils.typing import *
from threestudio.utils.base import BaseObject


@threestudio.register("dreamview-prompt-processor")
class DreamviewPromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        prompt_global: str = "a hamburger"

        prompt_front: Optional[str] = None
        prompt_right: Optional[str] = None
        prompt_back: Optional[str] = None
        prompt_left: Optional[str] = None

        negative_prompt: str = ""
        pretrained_model_name_or_path: str = ""
        front_threshold: float = 60
        back_threshold: float = 60

        view_dependent_prompt: bool = True
        use_cache: bool = True
        spawn: bool = True

    cfg: Config

    def configure(self) -> None:
        self._cache_dir = ".threestudio_cache/text_embeddings"

        self.prompt_global = self.cfg.prompt_global
        self.prompt_front = self.cfg.prompt_front
        self.prompt_right = self.cfg.prompt_right
        self.prompt_back = self.cfg.prompt_back
        self.prompt_left = self.cfg.prompt_left
        self.negative_prompt = self.cfg.negative_prompt

        # compute the text embeddings and save them
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = [self.cfg.prompt_global]
        if self.cfg.view_dependent_prompt:
            self.view_prompts = [self.cfg.prompt_front, self.cfg.prompt_right, self.cfg.prompt_back,
                                 self.cfg.prompt_left]
            all_prompts += self.view_prompts
        else:
            self.view_prompts = None

        all_prompts.append(self.cfg.negative_prompt)

        prompts_to_process = []
        for prompt in all_prompts:
            if self.cfg.use_cache:
                cache_path = os.path.join(self._cache_dir,
                                          f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt")
                if os.path.exists(cache_path):
                    threestudio.info(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing.")
                    continue
            prompts_to_process.append(prompt)

        threestudio.info(f"Prompt to process:{prompts_to_process}")

        if len(prompts_to_process) > 0:
            if self.cfg.spawn:
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(target=self.spawn_func, args=(
                self.cfg.pretrained_model_name_or_path, prompts_to_process, self._cache_dir,), )
                subprocess.start()
                subprocess.join()
            else:
                self.spawn_func(self.cfg.pretrained_model_name_or_path, prompts_to_process, self._cache_dir, )
            cleanup()

        # load the saved text embeddings
        self.load_text_embeddings()

    def load_text_embeddings(self):
        barrier()
        self.global_text_embeddings = self.load_from_cache(self.prompt_global)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[None, ...]
        if self.cfg.view_dependent_prompt:
            self.view_text_embeddings = torch.stack([self.load_from_cache(prompt) for prompt in self.view_prompts],
                                                    dim=0)
            threestudio.info(f"Loaded view text embeddings, size: {self.view_text_embeddings.size()}")
        else:
            self.view_text_embeddings = None
        threestudio.info(f"Loaded text embeddings.")

    def load_from_cache(self, prompt):
        cache_path = os.path.join(self._cache_dir, f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found.")
        return torch.load(cache_path, map_location=self.device)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",
                                                     device_map="auto")

        with torch.no_grad():
            tokens = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                               return_tensors="pt")
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(embedding, os.path.join(cache_dir, f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt"))

        del text_encoder

    def __call__(self):
        return DreamviewPromptProcessorOutput(global_text_embeddings=self.global_text_embeddings,
                                              uncond_text_embeddings=self.uncond_text_embeddings,
                                              view_text_embeddings=self.view_text_embeddings,
                                              front_threshold=self.cfg.front_threshold,
                                              back_threshold=self.cfg.back_threshold)


@dataclass
class DreamviewPromptProcessorOutput:
    global_text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]
    view_text_embeddings: Float[Tensor, "Nv N Nf"]
    front_threshold: float
    back_threshold: float

    def get_text_embeddings(self, azimuth):
        batch_size = azimuth.shape[0]

        global_text_embeddings = self.global_text_embeddings.expand(batch_size, -1, -1)
        uncond_text_embeddings = self.uncond_text_embeddings.expand(batch_size, -1, -1)

        if self.view_text_embeddings is None:
            view_text_embeddings = None
        else:
            view_text_embeddings = []
            for i in range(batch_size):
                # range (0, 360)
                if 90 - self.front_threshold <= azimuth[i] <= 90 + self.front_threshold:  # 30 - 150 is front
                    idx = 0
                elif 90 + self.front_threshold < azimuth[i] < 270 - self.back_threshold:  # 150 - 210 side
                    idx = 1
                elif 270 - self.back_threshold <= azimuth[i] <= 270 + self.back_threshold <= azimuth[i]:  # 210 - 330
                    idx = 2
                else:
                    idx = 3
                view_text_embeddings.append(self.view_text_embeddings[idx])

            view_text_embeddings = torch.stack(view_text_embeddings, dim=0)

        return global_text_embeddings, uncond_text_embeddings, view_text_embeddings

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        # return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
