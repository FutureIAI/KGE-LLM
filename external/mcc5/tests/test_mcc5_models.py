import unittest

import torch

from external.mcc5.models import MeasurementLocationFusion, build_mcc5_model


class MCC5ModelTest(unittest.TestCase):
    def test_measurement_fusion_shapes(self):
        model = MeasurementLocationFusion()
        output = model(torch.randn(2, 256, 6), torch.randn(2, 4))
        self.assertEqual(tuple(output["system_logits"].shape), (2, 20))
        self.assertEqual(tuple(output["fine_logits"].shape), (2, 10))

    def test_compact_transformer_shapes(self):
        model = build_mcc5_model(
            "m2",
            local_mode="measurement_correct",
            context_on=True,
            llama_path="",
            active_layers=4,
        )
        output = model(torch.randn(2, 512, 6), torch.randn(2, 4))
        self.assertEqual(tuple(output["fine_multilabel_logits"].shape), (2, 10))
        self.assertEqual(tuple(output["system20_logits"].shape), (2, 20))


if __name__ == "__main__":
    unittest.main()
