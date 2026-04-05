import warnings
import dataclasses
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizer


@dataclasses.dataclass
class Conversation:
    """Wrapper for conversation that can use either custom templates or HF chat templates"""

    system: str
    roles: Dict[str, str]
    messages: List[Tuple[str, str]]
    tokenizer: PreTrainedTokenizer
    do_generation: bool

    def reset(self):
        self.messages = []

    def append_message(self, role: str, content: str):
        self.messages.append((role, content))

    def get_prompt_chunks(self):
        """
        Returns:
            List[(text, is_target)]
        """
        assert self.messages[0][0] == "human", "First message must be human"

        if hasattr(self.tokenizer, "chat_template"):
            if self.tokenizer.chat_template is not None:
                return self._hf_chat_template_format()
            else:
                warnings.warn(
                    "Tokenizer has no chat_template defined. Falling back to plain format.",
                    category=UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Tokenizer does not support chat_template. Falling back to plain format.",
                category=UserWarning,
                stacklevel=2,
            )

        return self._plain_format()

    def _hf_chat_template_format(self):
        chunks = []
        messages_so_far = []

        if self.system:
            messages_so_far.append({"role": "system", "content": self.system})
            system_chunk_text = self.tokenizer.apply_chat_template(
                messages_so_far, tokenize=False, add_generation_prompt=False
            )
            chunks.append((system_chunk_text, False))
            prev_prompt_len = len(system_chunk_text)
        else:
            prev_prompt_len = 0

        for idx, (role, content) in enumerate(self.messages):
            add_generation_prompt = self.do_generation and (idx + 1) == len(
                self.messages
            )

            messages_so_far.append(
                {"role": self.roles.get(role, role), "content": content}
            )

            full_prompt = self.tokenizer.apply_chat_template(
                messages_so_far,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

            chunk_text = full_prompt[prev_prompt_len:]  # type: ignore
            is_target = role == "gpt"

            chunks.append((chunk_text, is_target))
            prev_prompt_len = len(full_prompt)

        return chunks

    def _plain_format(self):
        chunks = []
        for role, msg in self.messages:
            chunks.append((msg + "\n", role == "gpt"))
        return chunks


def create_conversation(
    tokenizer: PreTrainedTokenizer, system_prompt: str, do_generation: bool = False
) -> Conversation:
    roles = {"human": "user", "gpt": "assistant"}

    return Conversation(
        system=system_prompt,
        roles=roles,
        messages=[],
        tokenizer=tokenizer,
        do_generation=do_generation,
    )
