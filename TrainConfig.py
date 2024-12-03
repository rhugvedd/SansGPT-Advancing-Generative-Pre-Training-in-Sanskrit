import torch

class TrainConfig:
    def __init__(
                    self,
                    tokens_batch_size: int,
                    batch_size: int,
                    dec_context_size: int,
                    batch_overlap: int,
                    betas: tuple,
                    vocab_size: int,
                    d_model: int,
                    num_heads: int,
                    num_decoder_blocks: int,
                    pos_enc_dropout: float,
                    drop_prob: float,
                    weight_decay: float,
                    d_feedfwd: int,
                    mask_attention: bool,
                    pre_norm: bool,
                    x_data_loader_dtype: torch.dtype,
                    y_data_loader_dtype: torch.dtype,
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
                    train_shard_names: str,
                    val_name: str,
                    device: torch.device,
                    load_shard_indx = 0,
                    load_train_batch_indx = 0
                ):

        assert tokens_batch_size % (batch_size * dec_context_size) == 0, "The Tokens Batch Size must me a multiple of Batch Size and context size"
        
        self.gradient_accum_iters = tokens_batch_size // (batch_size * dec_context_size)

        self.tokens_batch_size = tokens_batch_size
        self.batch_size = batch_size
        self.dec_context_size = dec_context_size
        self.batch_overlap = batch_overlap
        self.betas = betas
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks
        self.pos_enc_dropout = pos_enc_dropout
        self.drop_prob = drop_prob
        self.weight_decay = weight_decay
        self.d_feedfwd = d_feedfwd
        self.mask_attention = mask_attention
        self.pre_norm = pre_norm
        self.x_data_loader_dtype = x_data_loader_dtype
        self.y_data_loader_dtype = y_data_loader_dtype
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
        self.train_shard_names = train_shard_names
        self.val_name = val_name
        self.device = device

        self.load_shard_indx = load_shard_indx
        self.load_train_batch_indx = load_train_batch_indx