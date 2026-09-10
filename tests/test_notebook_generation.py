"""CPU regression check for the notebook's actual response helper."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


class NotebookGenerationTest(unittest.TestCase):
    def test_generation_accepts_chat_mapping_and_decodes_only_completion(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "chemistry_llm_colab.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "def generate_response(" in "".join(cell["source"]))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "generate_response")
        backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=256, n_embd=8, n_layer=1, n_head=1,
                                         bos_token_id=1, eos_token_id=1, pad_token_id=1)).eval()
        namespace = {"tokenizer": tokenizer, "model": model, "torch": torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_helper", "exec"), namespace)
        original_generate = model.generate
        captured = {}

        def generate(*args, **kwargs):
            output = original_generate(*args, **kwargs)
            captured.update(kwargs=kwargs, output=output)
            return output

        for prompt in ("hello", "hello world hello"):
            with self.subTest(prompt=prompt), patch.object(model, "generate", side_effect=generate):
                result = namespace["generate_response"](prompt)
                inputs = captured["kwargs"]
                self.assertIn("attention_mask", inputs)
                self.assertEqual(inputs["input_ids"].shape, inputs["attention_mask"].shape)
                expected = tokenizer.decode(captured["output"][0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
