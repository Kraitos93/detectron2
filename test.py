from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("bottle_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "bottle_test")
inference_on_dataset(trainer.model, val_loader, evaluator)