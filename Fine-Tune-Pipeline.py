import torch
from FineTuneConfig import *
from FineTuner import *
from BPETokenizer import *
import gc

ADAM_W = 'AdamW'

SEMANTIC_ANALOGY = 'Semantic Analogy'
UPAMA = 'Upama'

configs = []

config_nos = 1

for i in range(config_nos):
    configs.append(FineTuneConfig   (
                                        # Model:
                                        # tokens_batch_size = 1024 * 8 * 1,
                                        # batch_size = 8,
                                        # dec_context_size = 1024, 
                                        # betas = (0.9, 0.95),
                                        # vocab_size = 12000,
                                        # d_model = 1024,
                                        # num_heads = 16,
                                        # num_decoder_blocks = 18,
                                        # num_freeze_blocks = 9,
                                        # pos_enc_dropout = 0,
                                        # drop_prob = 0,
                                        # weight_decay = None,
                                        # d_feedfwd = 3072,
                                        # mask_attention = True,
                                        # pre_norm = True,
                                        # ===========================================================================================
                                        tokens_batch_size = 512 * 8 * 1,
                                        batch_size = 8,
                                        dec_context_size = 512,
                                        betas = (0.9, 0.95),
                                        vocab_size = 12000,
                                        d_model = 768,
                                        num_heads = 12,
                                        num_decoder_blocks = 12,
                                        num_freeze_blocks = 8,
                                        pos_enc_dropout = 0,
                                        drop_prob = 0,
                                        weight_decay = None,
                                        d_feedfwd = 512 * 4,
                                        mask_attention = True,
                                        pre_norm = True,

                                        # Data Loader and Checkpointing:
                                        load_check_point = True,
                                        checkpoint_path = './CheckPoints/',
                                        checkpoint_name = 'Pipe-17-12000-CFG-0-Complete-Sanskrit-Iter-300000-2024-09-12 17-02-20',
                                        checkpoint_save_iter = None,
                                        num_iters = None,
                                        eval_val_set = True,
                                        val_eval_iters = None,
                                        val_eval_interval = None,
                                        
                                        # Optimization:
                                        optimizer_name = None,
                                        max_lr = None,
                                        min_lr = None,
                                        model_name = 'Fine-Tune-',
                                        warmup_iters = None,
                                        clip_grad_norm = 1.0,

                                        # Training Files:
                                        replacements = {},
                                        file_name = None,
                                        file_path = "./Eval_Tasks/",
                                        vocab_path = "./Tokenizer/",
                                        load_merge_info_name = 'Final-Corpus-Tokenizer-Merge-Info-NL-12000-2024-08-31 03-04-04',
                                        load_vocab_name = 'Final-Corpus-Tokenizer-Vocab-NL-12000-2024-08-31 03-04-04',
                                        data_path = "./Eval_Tasks/Eval_Tensors/",
                                        x_train_name = None,
                                        y_train_name = None,
                                        mask_train_name = None,
                                        x_val_name = None,
                                        y_val_name = None,
                                        mask_val_name = None,
                                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                        
                                        # Mention the fine tuning task here
                                        fine_tuning_task = SEMANTIC_ANALOGY,
                                        pad_token = None,
                                        sep_token = None,
                                        eos_token = None,
                                        topk_sampling_k = 50
                                    ))

Tokenizer = BPETokenizer()
Tokenizer.load(configs[0].vocab_path, configs[0].load_merge_info_name, configs[0].load_vocab_name)
pad_token = Tokenizer.special_tok('<pad>')
sep_token = Tokenizer.special_tok('<sep>')
eos_token = Tokenizer.special_tok('<eos>')

for i in range(config_nos):
    configs[i].pad_token = pad_token
    configs[i].sep_token = sep_token
    configs[i].eos_token = eos_token

    if configs[i].fine_tuning_task == SEMANTIC_ANALOGY:
        configs[i].file_name = 'Semantic Analogies.csv'

        configs[i].x_train_name = 'X-Eval-12000-Semantic-Analogies-Train-2024-09-14 22-18-08'
        configs[i].y_train_name = 'Y-Eval-12000-Semantic-Analogies-Train-2024-09-14 22-18-08'
        configs[i].mask_train_name = 'Mask-Eval-12000-Semantic-Analogies-Train-2024-09-14 22-18-08'
        configs[i].x_val_name = 'X-Eval-12000-Semantic-Analogies-Val-2024-09-14 22-18-08'
        configs[i].y_val_name = 'Y-Eval-12000-Semantic-Analogies-Val-2024-09-14 22-18-08'
        configs[i].mask_val_name = 'Mask-Eval-12000-Semantic-Analogies-Val-2024-09-14 22-18-08'

    elif configs[i].fine_tuning_task == UPAMA:
        configs[i].file_name = 'Upama Data.csv'

        configs[i].x_train_name = 'X-Eval-12000-Upama-V0.9-Train-2024-09-17 01-08-40'
        configs[i].y_train_name = 'Y-Eval-12000-Upama-V0.9-Train-2024-09-17 01-08-40'
        configs[i].mask_train_name = 'Mask-Eval-12000-Upama-V0.9-Train-2024-09-17 01-08-40'
        configs[i].x_val_name = 'X-Eval-12000-Upama-V0.9-Val-2024-09-17 01-08-40'
        configs[i].y_val_name = 'Y-Eval-12000-Upama-V0.9-Val-2024-09-17 01-08-40'
        configs[i].mask_val_name = 'Mask-Eval-12000-Upama-V0.9-Val-2024-09-17 01-08-40'

    else:
        raise ValueError("Wrong task selected!")

config_nos = config_nos

pipe_indx = 0
cfg_no = -1

cfg_no += 1
configs[cfg_no].fine_tuning_task = UPAMA
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 12000
configs[cfg_no].model_name = 'FT-Ext-Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-' + configs[cfg_no].fine_tuning_task
configs[cfg_no].warmup_iters = 1000
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 8 * 1
configs[cfg_no].batch_size = 8
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].drop_prob = 0
configs[cfg_no].checkpoint_save_iter = 2000
configs[cfg_no].num_iters = 20000
configs[cfg_no].val_eval_iters = 200
configs[cfg_no].val_eval_interval = 200

# ====================================================================================================================================================

cfg_no += 1
configs[cfg_no].fine_tuning_task = SEMANTIC_ANALOGY
configs[cfg_no].optimizer_name = ADAM_W
configs[cfg_no].max_lr = 2e-4
configs[cfg_no].min_lr = 2e-5
configs[cfg_no].vocab_size = 12000
configs[cfg_no].model_name = 'FT-Ext-Pipe-' + str(pipe_indx) + '-CFG-' + str(cfg_no) + '-' + configs[cfg_no].fine_tuning_task
configs[cfg_no].warmup_iters = 700
configs[cfg_no].weight_decay = 1e-2
configs[cfg_no].tokens_batch_size = 512 * 8 * 1
configs[cfg_no].batch_size = 8
assert configs[cfg_no].tokens_batch_size % (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
configs[cfg_no].gradient_accum_iters = configs[cfg_no].tokens_batch_size // (configs[cfg_no].batch_size * configs[cfg_no].dec_context_size)
configs[cfg_no].drop_prob = 0
configs[cfg_no].checkpoint_save_iter = 720
configs[cfg_no].num_iters = 7250
configs[cfg_no].val_eval_iters = 80
configs[cfg_no].val_eval_interval = 72

for i in range(config_nos):
    fine_tuner = FineTuner(configs[i])
    fine_tuner.train()
    del fine_tuner
    gc.collect()