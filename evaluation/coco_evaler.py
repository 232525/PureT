import os
import sys
import tempfile
import json
from json import encoder
from lib.config import cfg

# sys.path.append(cfg.INFERENCE.COCO_PATH)
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

class COCOEvaler(object):
    def __init__(self, annfile):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        if not os.path.exists(cfg.TEMP_DIR):
            os.mkdir(cfg.TEMP_DIR)

    def eval(self, result):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=cfg.TEMP_DIR)
        # save results to tmp file
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval
    
    def eval_no_spice(self, result):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=cfg.TEMP_DIR)
        # save results to tmp file
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate_no_spice()
        os.remove(in_file.name)
        return cocoEval.eval