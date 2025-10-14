# *** AI Studio experiment test ***
# workspaces:
#  - local_genai
# ******

import unittest


class TestBertNotebook(unittest.TestCase):

    def test_torch_installation(self):
        try:
            import torch

            self.assertTrue(
                torch.__version__ is not None,
                "Torch is not installed or version is not accessible.",
            )
            self.assertTrue(hasattr(torch, "cuda"), "Torch does not have CUDA support.")
            self.assertTrue(
                torch.cuda.is_available(),
                "CUDA is not available with the installed Torch version.",
            )
        except ImportError:
            self.fail("Torch is not installed.")


if __name__ == "__main__":
    unittest.main()
