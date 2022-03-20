import os
import sys
import numpy as np
import torch
import tqdm
import json
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg

# 模型验证
class OnlineTester(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats
    ):
        super(OnlineTester, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        # './mscoco/txt/coco_val_image_id.txt'  Karpathy验证集  5K张图像
        # './mscoco/txt/coco_test_image_id.txt' Karpathy测试集  5K张图像
        # './mscoco/txt/coco_test4w_image_id.txt' MSCOCO在线测试集 4W张图像
        self.eval_ids = np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats)

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs
        
    def __call__(self, model, rname):
        model.eval()
        
        results = []
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask) in enumerate(tqdm.tqdm(self.eval_loader)):
                ids = self.eval_ids[indices]
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = model.module.decode_beam(**kwargs)
                else:
                    seq, _ = model.module.decode(**kwargs)
                sents = utils.decode_sequence(self.vocab, seq.data)
                for sid, sent in enumerate(sents):
                    # 构造模型验证结果 {'image_id': ***, 'caption': 'word1 word2 word3 ...'}
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    results.append(result)

        # 在线测试不需要评估，直接保存模型输出结果即可
        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()