import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import *
from DataLoader import DataLoader
import datetime
import time
from FineTuneConfig import FineTuneConfig
from sklearn.metrics import precision_score, recall_score, f1_score

ADAM_W = 'AdamW'

SEMANTIC_ANALOGY = 'Semantic Analogy'
UPAMA = 'Upama'

class FineTuner:
    def __init__(
                    self,
                    fine_tune_config: FineTuneConfig
                ):
        self.config = fine_tune_config

    def train(self):

        print("Initializing")
        print(f"Total Tokens Batch Size: {self.config.tokens_batch_size}")
        print(f"Iterations for Gradient Accumulation: {self.config.gradient_accum_iters}")

        torch.cuda.empty_cache()

        self.GPT = Decoder  (
                                vocab_size = self.config.vocab_size,
                                d_model = self.config.d_model,
                                context_size = self.config.dec_context_size,
                                pos_enc_dropout = self.config.pos_enc_dropout,
                                num_decoder_blocks = self.config.num_decoder_blocks,
                                num_heads = self.config.num_heads,
                                drop_prob = self.config.drop_prob,
                                d_feedfwd = self.config.d_feedfwd,
                                pre_norm = self.config.pre_norm,
                                mask_attention = self.config.mask_attention
                            )

        m = self.GPT.to(self.config.device)

        print(f"No. of Parameters: {sum(p.numel() for p in m.parameters()) / 1e6} M parameters\n")

        #=================================================================================================================================================

        print("Loading Data")
        self.data_loader = DataLoader(self.config.data_path)
        
        self.data_loader.load_fine_tune_data (   
                                batch_size = self.config.batch_size,
                                x_train_name = self.config.x_train_name,
                                y_train_name = self.config.y_train_name,
                                mask_train_name = self.config.mask_train_name,
                                x_val_name = self.config.x_val_name,
                                y_val_name = self.config.y_val_name,
                                mask_val_name = self.config.mask_val_name
                            )

        print("Data Loading Complete\n")

        #====================================================================================================================

        print("Configuring Optimzer and Learning Rate Scheduler")

        iters = torch.arange(self.config.num_iters + 1)
        
        if self.config.optimizer_name == ADAM_W:
            lr_schedule = (self.config.max_lr * ((iters < self.config.warmup_iters) * (iters + 1) / self.config.warmup_iters)) + ((self.config.min_lr + (0.5 * (self.config.max_lr - self.config.min_lr) * (1 + torch.cos((iters - self.config.warmup_iters) * torch.pi / (self.config.num_iters - self.config.warmup_iters))))) * (iters >= self.config.warmup_iters))
        else:
            raise ValueError('Wrong Optimizer Name..!')

        decay_params = [param for param in self.GPT.parameters() if (param.requires_grad and (param.dim() >= 2))]
        no_decay_params = [param for param in self.GPT.parameters() if (param.requires_grad and (param.dim() < 2))]

        print(f"Number of parameter tensors with weight decay: {len(decay_params)}, totaling {sum(p.numel() for p in decay_params):,} parameters")
        print(f"Number of parameter tensors without weight decay: {len(no_decay_params)}, totaling {sum(p.numel() for p in no_decay_params):,} parameters")

        if self.config.optimizer_name == ADAM_W:
            print('\nUsing AdamW')
            print(f'Max LR: {self.config.max_lr}')
            print(f'Min LR: {self.config.min_lr}')
            print(f'Warmup Iters: {self.config.warmup_iters}\n')
            optimizer = torch.optim.AdamW(
                                        [
                                            {'params': decay_params, 'weight_decay': self.config.weight_decay},
                                            {'params': no_decay_params, 'weight_decay': 0.0}
                                        ],
                                        lr = lr_schedule[0],
                                        betas = self.config.betas,
                                        eps = 1e-8
                                    )
        else:
            raise ValueError("Wrong Optimizer Name!!")

        st_iter = 0
        val_losses = []
        total_loss_list = []
        total_norm_list = []
        sem_anlg_eval = []
        upama_eval = []
        print("Configuration Complete\n")

        # ====================================================================================================================

        if self.config.load_check_point:
            print("\nLoading Checkpoint")
            checkpoint = torch.load(self.config.checkpoint_path + self.config.checkpoint_name + '.pth')
            self.GPT.load_state_dict(checkpoint['model_state_dict'])   
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            val_losses = checkpoint['val_losses']
            total_loss_list = checkpoint['total_loss_list']
            total_norm_list = checkpoint['total_norm_list']

            try:
                sem_anlg_eval = checkpoint['sem_anlg_eval']
            except:
                None

            try:
                upama_eval = checkpoint['upama_eval']
            except:
                None

            print('\nLoaded Checkpoint: ' + self.config.checkpoint_path + self.config.checkpoint_name)

            print(f'Starting Iter for Training: {st_iter}')

            lr = optimizer.param_groups[0]['lr']
            print("Learning rate of loaded model:", lr)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[st_iter]
            
            print("Setting learning rate to:", optimizer.param_groups[0]['lr'])

        # ====================================================================================================================   
        
        if self.config.fine_tuning_task == SEMANTIC_ANALOGY:
            
            def calc_loss(preds, targets, mask):
                B, T, C = preds.shape
                targets = targets.to(torch.int64)

                return F.cross_entropy(preds.view(B*T, C), targets.view(B*T), ignore_index = self.config.pad_token)

            @torch.no_grad()
            def eval_task():
                self.GPT.eval()
                loss_values = torch.zeros(self.config.val_eval_iters)
                self.data_loader.shuffle('val', reset_batch_index = True)
            
                correct_predictions = 0
                total_predictions = 0

                all_preds = []
                all_targets = []

                for eval_iter in range(self.config.val_eval_iters):
                    print('-', end="")
                    
                    X, Y = self.data_loader.get_val_batch(self.config.device)
                    mask = self.data_loader.get_val_mask(self.config.device)

                    preds = self.GPT(X, mask)

                    loss_values[eval_iter] = calc_loss(preds, Y, mask).item()

                    preds = F.softmax(preds, dim=-1)

                    top_probs, top_indices = preds.topk(self.config.topk_sampling_k, dim=-1)

                    flattened_top_probs = top_probs.view(-1, top_probs.size(-1))
                    flattened_next_tok_ids = torch.multinomial(flattened_top_probs, num_samples=1)
                    next_tok_ids = flattened_next_tok_ids.view(top_probs.size(0), top_probs.size(1), -1)

                    next_toks = top_indices.gather(-1, next_tok_ids)

                    next_toks = next_toks.squeeze(-1)

                    B, T = X.shape

                    # Flip the tensor along the last dimension (reverse the sequence for each row)
                    X_flipped = torch.flip(X, dims=[1])  # Flipping along the token dimension (T)

                    for i in range(B):
                        # Find the first occurrence of <eos> and <sep> in the flipped row
                        eos_pos_flipped = (X_flipped[i] == self.config.eos_token).nonzero(as_tuple=True)[0]
                        sep_pos_flipped = (X_flipped[i] == self.config.sep_token).nonzero(as_tuple=True)[0]

                        # Get the indices in the original row by reversing the flipped index
                        eos_index_original = (T - 1 - eos_pos_flipped[0].item()) if len(eos_pos_flipped) > 0 else -1
                        sep_index_original = (T - 1 - sep_pos_flipped[0].item()) if len(sep_pos_flipped) > 0 else -1

                        target_toks = Y[i][sep_index_original: (eos_index_original - 1)]
                        pred_toks = next_toks[i][sep_index_original: (eos_index_original - 1)]

                        all_targets.extend(target_toks.cpu().numpy())
                        all_preds.extend(pred_toks.cpu().numpy())

                        correct_predictions += torch.sum(target_toks == pred_toks).item()
                        total_predictions += len(pred_toks)
                        
                val_loss = loss_values.mean().item()
                accuracy = (correct_predictions / total_predictions) * 100
                precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

                self.GPT.train()
                
                log_to_print = f"\nVal Loss: {val_loss} | Acc: {accuracy} | Precision: {precision} | Recal: {recall} | F1: {f1}"

                return val_loss, (accuracy, precision, recall, f1), log_to_print

        elif self.config.fine_tuning_task == UPAMA:
            
            def calc_loss(preds, targets, mask):
                B, T, C = preds.shape
                targets = targets.to(torch.int64)

                return F.cross_entropy(preds.view(B*T, C), targets.view(B*T), ignore_index = self.config.pad_token)

            @torch.no_grad()
            def eval_task():
                self.GPT.eval()
                loss_values = torch.zeros(self.config.val_eval_iters)
                self.data_loader.shuffle('val', reset_batch_index = True)
            
                correct_predictions = 0
                total_predictions = 0

                all_preds = []
                all_targets = []

                for eval_iter in range(self.config.val_eval_iters):
                    print('-', end="")
                    
                    X, Y = self.data_loader.get_val_batch(self.config.device)
                    mask = self.data_loader.get_val_mask(self.config.device)

                    preds = self.GPT(X, mask)

                    loss_values[eval_iter] = calc_loss(preds, Y, mask).item()

                    preds = F.softmax(preds, dim=-1)

                    top_probs, top_indices = preds.topk(self.config.topk_sampling_k, dim=-1)

                    flattened_top_probs = top_probs.view(-1, top_probs.size(-1))
                    flattened_next_tok_ids = torch.multinomial(flattened_top_probs, num_samples=1)
                    next_tok_ids = flattened_next_tok_ids.view(top_probs.size(0), top_probs.size(1), -1)

                    next_toks = top_indices.gather(-1, next_tok_ids)

                    next_toks = next_toks.squeeze(-1)

                    B, T = X.shape

                    # Flip the tensor along the last dimension (reverse the sequence for each row)
                    X_flipped = torch.flip(X, dims=[1])  # Flipping along the token dimension (T)

                    for i in range(B):
                        # Find the first occurrence of <eos> and <sep> in the flipped row
                        eos_pos_flipped = (X_flipped[i] == self.config.eos_token).nonzero(as_tuple=True)[0]
                        sep_pos_flipped = (X_flipped[i] == self.config.sep_token).nonzero(as_tuple=True)[0]

                        # Get the indices in the original row by reversing the flipped index
                        eos_index_original = (T - 1 - eos_pos_flipped[0].item()) if len(eos_pos_flipped) > 0 else -1
                        sep_index_original = (T - 1 - sep_pos_flipped[0].item()) if len(sep_pos_flipped) > 0 else -1

                        target_toks = Y[i][sep_index_original: (eos_index_original - 1)]
                        pred_toks = next_toks[i][sep_index_original: (eos_index_original - 1)]

                        all_targets.extend(target_toks.cpu().numpy())
                        all_preds.extend(pred_toks.cpu().numpy())

                        correct_predictions += torch.sum(target_toks == pred_toks).item()
                        total_predictions += len(pred_toks)

                val_loss = loss_values.mean().item()
                accuracy = (correct_predictions / total_predictions) * 100
                precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

                self.GPT.train()
                
                log_to_print = f"\nVal Loss: {val_loss} | Acc: {accuracy} | Precision: {precision} | Recal: {recall} | F1: {f1}"

                return val_loss, (accuracy, precision, recall, f1), log_to_print

        else:
            raise ValueError('Wrong Fine Tuning Task Selected')

        # ====================================================================================================================

        if self.config.num_freeze_blocks > 0:
            for dec_block in self.GPT.decoder[:self.config.num_freeze_blocks]:
                for param in dec_block.parameters():
                    param.requires_grad = False

        torch.cuda.empty_cache()
        print("Computing Started")

        for iter in range(st_iter, self.config.num_iters + 1):
            st_time = time.time()
            
            cumulative_loss = 0.0
            optimizer.zero_grad()

            for mini_iter in range(self.config.gradient_accum_iters):

                train_x, train_y = self.data_loader.get_ft_train_batch(self.config.device)
                attention_mask = self.data_loader.get_train_mask(self.config.device)
                batch_no = self.data_loader.train_batch_index

                preds = self.GPT(train_x, attention_mask)

                loss = calc_loss(preds, train_y, attention_mask)

                loss = loss / self.config.gradient_accum_iters
                cumulative_loss += loss.detach()
                loss.backward()

            if self.config.clip_grad_norm != False:
                norm = torch.nn.utils.clip_grad_norm_(self.GPT.parameters(), self.config.clip_grad_norm)
                total_norm_list.append(norm.item())
            else:
                norm = 0
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iter]

            optimizer.step()

            if self.config.device.type == 'cuda': 
                torch.cuda.synchronize()
            
            total_loss_list.append(cumulative_loss.item())

            if self.config.eval_val_set and ((iter % self.config.val_eval_interval == 0) or (iter == self.config.num_iters - 1)):

                print("\nEvaluating Task on Validation Set")
                
                if self.config.fine_tuning_task == SEMANTIC_ANALOGY:
                    val_loss, metric, log_to_print = eval_task()
                    sem_anlg_eval.append(metric)

                    log_msg = 'Sem-Anlg:'
                    log_val = f"Acc: {metric[0]:.3f} | P: {metric[1]:.3f} | R: {metric[2]:.3f} | F1: {metric[3]:.3f}"

                elif self.config.fine_tuning_task == UPAMA:
                    val_loss, metric, log_to_print = eval_task()

                    upama_eval.append(metric)
                    log_msg = 'UPAMA'

                    log_val = f"Acc: {metric[0]:.3f} | P: {metric[1]:.3f} | R: {metric[2]:.3f} | F1: {metric[3]:.3f}"
                else:
                    raise ValueError('Wrong Fine Tuning Task Selected')
                
                val_losses.append(val_loss)

                print(log_to_print)
            
            if (iter % self.config.checkpoint_save_iter == 0) and (iter != 0):
                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

                torch.save  ({
                                'iter': iter,
                                'model_state_dict': self.GPT.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_losses': val_losses,
                                'sem_anlg_eval': sem_anlg_eval,
                                'upama_eval': upama_eval,
                                'total_loss_list': total_loss_list,
                                'total_norm_list': total_norm_list,
                                'train_config': self.config,
                                'train_batch_index': self.data_loader.train_batch_index
                            }, self.config.checkpoint_path + self.config.model_name + '-Iter-' + str(iter) + '-' + date_time + '.pth')
                print("Checkpoint Saved")

            time_taken = time.time() - st_time
            token_throughput = self.config.batch_size * self.config.dec_context_size * self.config.gradient_accum_iters / time_taken

            print(f"Iter: {iter:4d} | Loss: {cumulative_loss.item():.5f} | Norm: {norm:.4f} | {log_msg} {log_val} | Batch No: ({batch_no:4d}/{self.data_loader.train_num_batches:4d}) | Time: {time_taken*1000:.2f}ms | LR: {lr_schedule[iter]:.3e} | {self.config.model_name}")

        print("Training Complete")