import torch

class FineTuneConfig:
    def __init__(
                    self,
                    tokens_batch_size: int,
                    batch_size: int,
                    dec_context_size: int,
                    betas: tuple,
                    vocab_size: int,
                    d_model: int,
                    num_heads: int,
                    num_decoder_blocks: int,
                    num_freeze_blocks: int,
                    pos_enc_dropout: float,
                    drop_prob: float,
                    weight_decay: float,
                    d_feedfwd: int,
                    mask_attention: bool,
                    pre_norm: bool,
                    load_check_point: bool,
                    checkpoint_path: str,
                    checkpoint_name: str,
                    checkpoint_save_iter: int,
                    num_iters: int,
                    eval_val_set: bool,
                    val_eval_iters: int,
                    val_eval_interval: int,
                    optimizer_name: str,
                    max_lr: float,
                    min_lr: float,
                    model_name: str,
                    warmup_iters: int,
                    clip_grad_norm,
                    replacements: dict,
                    file_name: str,
                    file_path: str,
                    vocab_path: str,
                    load_merge_info_name: str,
                    load_vocab_name: str,
                    data_path: str,
                    x_train_name: str,
                    y_train_name: str,
                    mask_train_name: str,
                    x_val_name: str,
                    y_val_name: str,
                    mask_val_name: str,
                    device: torch.device,
                    fine_tuning_task: str,
                    pad_token: int,
                    sep_token: int,
                    eos_token: int,
                    topk_sampling_k: int
                ):

        assert tokens_batch_size % (batch_size * dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
        
        self.gradient_accum_iters = tokens_batch_size // (batch_size * dec_context_size)

        self.tokens_batch_size = tokens_batch_size
        self.batch_size = batch_size
        self.dec_context_size = dec_context_size
        self.betas = betas
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks
        self.num_freeze_blocks = num_freeze_blocks
        self.pos_enc_dropout = pos_enc_dropout
        self.drop_prob = drop_prob
        self.weight_decay = weight_decay
        self.d_feedfwd = d_feedfwd
        self.mask_attention = mask_attention
        self.pre_norm = pre_norm
        self.load_check_point = load_check_point
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.checkpoint_save_iter = checkpoint_save_iter
        self.num_iters = num_iters
        self.eval_val_set = eval_val_set
        self.val_eval_iters = val_eval_iters
        self.val_eval_interval = val_eval_interval
        self.optimizer_name = optimizer_name
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.model_name = model_name
        self.warmup_iters = warmup_iters
        self.clip_grad_norm = clip_grad_norm
        self.replacements = replacements
        self.file_name = file_name
        self.file_path = file_path
        self.vocab_path = vocab_path
        self.load_merge_info_name = load_merge_info_name
        self.load_vocab_name = load_vocab_name
        self.data_path = data_path
        self.x_train_name = x_train_name
        self.y_train_name = y_train_name
        self.mask_train_name = mask_train_name
        self.x_val_name = x_val_name
        self.y_val_name = y_val_name
        self.mask_val_name = mask_val_name
        self.device = device
        
        self.fine_tuning_task = fine_tuning_task
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.eos_token = eos_token
        self.topk_sampling_k = topk_sampling_k