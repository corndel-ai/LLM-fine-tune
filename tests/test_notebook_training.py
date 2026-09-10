"""Validate the notebook's training arguments against the installed API on CPU."""
import ast
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import TrainingArguments


class NotebookTrainingTest(unittest.TestCase):
    def test_training_arguments_preserve_five_percent_warmup(self):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "chemistry_llm_colab.ipynb").read_text()
        )
        cells = ["".join(cell["source"]) for cell in notebook["cells"]]
        config = next(source for source in cells if source.startswith("SEED ="))
        training = next(source for source in cells if "training_args = TrainingArguments(" in source)
        constants = [node for node in ast.parse(config).body if isinstance(node, ast.Assign)]
        assignment = next(
            node for node in ast.parse(training).body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "training_args"
                for target in node.targets
            )
        )
        # Keep every notebook argument; only select CPU and disable GPU precision.
        assignment.value.keywords.append(ast.keyword(arg="use_cpu", value=ast.Constant(True)))
        module = ast.fix_missing_locations(ast.Module(body=constants + [assignment], type_ignores=[]))
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "torch": torch, "TrainingArguments": TrainingArguments,
                "COMPUTE_DTYPE": torch.float32, "OUTPUT_DIR": Path(directory),
            }
            exec(compile(module, "notebook_training_arguments", "exec"), namespace)
            args = namespace["training_args"]
            for total_steps in (args.max_steps, 50, 100, 101):
                self.assertEqual(args.get_warmup_steps(total_steps), math.ceil(0.05 * total_steps))
            self.assertEqual(args.eval_steps, args.save_steps)
            self.assertTrue(args.load_best_model_at_end)
            self.assertEqual(args.optim.value, "paged_adamw_8bit")


if __name__ == "__main__":
    unittest.main()
