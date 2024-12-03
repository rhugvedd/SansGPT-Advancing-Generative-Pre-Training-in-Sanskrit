import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import *
from DataLoader import DataLoader
import datetime
import time
from TrainConfig import TrainConfig

ADAM_W = 'AdamW'

class Trainer:
    def __init__(
                    self,
                    trainer_config: TrainConfig
                ):
        self.config = trainer_config

    @torch.no_grad()
    def estimate_val_loss(self):
        self.GPT.eval()
        loss_values = torch.zeros(self.config.val_eval_iters)
        self.data_loader.shuffle('val', reset_batch_index = True)

        for loss_iter in range(self.config.val_eval_iters):
            print('-', end="")
            
            X, Y = self.data_loader.get_val_batch(self.config.device)
        
            preds = self.GPT(X)
            
            B, T, C = preds.shape
            preds = preds.view(B*T, C)
            Y = Y.view(B*T)
            loss = F.cross_entropy(preds, Y)

            loss_values[loss_iter] = loss.item()

        val_loss = loss_values.mean().item()

        self.GPT.train()
        
        return val_loss

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
        self.data_loader.load_data  (
                                        batch_size = self.config.batch_size,
                                        context_size = self.config.dec_context_size,
                                        train_shard_names = self.config.train_shard_names,
                                        batch_overlap = self.config.batch_overlap,
                                        x_dtype = self.config.x_data_loader_dtype,
                                        y_dtype = self.config.y_data_loader_dtype,
                                        val_name = self.config.val_name if self.config.eval_val_set else None,
                                        load_shard_indx = self.config.load_shard_indx,
                                        load_train_batch_indx = self.config.load_train_batch_indx
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

        print("Configuration Complete\n")

        #====================================================================================================================

        if self.config.load_check_point:
            print("\nLoading Checkpoint")
            checkpoint = torch.load(self.config.checkpoint_path + self.config.checkpoint_name + '.pth')
            self.GPT.load_state_dict(checkpoint['model_state_dict'])   
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            val_losses = checkpoint['val_losses']
            total_loss_list = checkpoint['total_loss_list']
            total_norm_list = checkpoint['total_norm_list']
            
            print('\nLoaded Checkpoint: ' + self.config.checkpoint_path + self.config.checkpoint_name)
            
            st_iter = checkpoint['iter'] + 1

            print(f'Starting Iter for Training: {st_iter}')

            lr = optimizer.param_groups[0]['lr']
            print("Learning rate of loaded model:", lr)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[st_iter]
            
            print("Setting learning rate to:", optimizer.param_groups[0]['lr'])

        #====================================================================================================================   

        torch.cuda.empty_cache()
        print("Computing Started")

        for iter in range(st_iter, self.config.num_iters + 1):
            st_time = time.time()
            
            cumulative_loss = 0.0
            optimizer.zero_grad()

            for mini_iter in range(self.config.gradient_accum_iters):

                train_x, train_y = self.data_loader.get_train_batch(self.config.device)
                batch_no = self.data_loader.train_batch_index

                preds = self.GPT(train_x)

                B, T, C = preds.shape
                preds = preds.view(B*T, C)
                train_y = train_y.view(B*T)

                loss = F.cross_entropy(preds, train_y)

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
                print("\nEvaluating Val Loss")
                val_loss = self.estimate_val_loss()
                val_losses.append(val_loss)
                print(f"\nIter {iter}: Val loss: {val_loss:.4f}\n")
            
            if (iter % self.config.checkpoint_save_iter == 0) and (iter != 0):
                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

                torch.save  ({
                                'iter': iter,
                                'model_state_dict': self.GPT.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_losses': val_losses,
                                'total_loss_list': total_loss_list,
                                'total_norm_list': total_norm_list,
                                'train_config': self.config,
                                'shard_indx': self.data_loader.shard_indx,
                                'train_batch_index': self.data_loader.train_batch_index
                            }, self.config.checkpoint_path + self.config.model_name + '-Iter-' + str(iter) + '-' + date_time + '.pth')
                print("Checkpoint Saved")

            time_taken = time.time() - st_time
            token_throughput = self.config.batch_size * self.config.dec_context_size * self.config.gradient_accum_iters / time_taken

            print(f"Iter: {iter:4d} | Loss: {cumulative_loss.item():.5f} | Norm: {norm:.4f} | Batch No: ({batch_no:4d}/{self.data_loader.train_num_batches:4d}) | Token Throughput: {token_throughput:.2f} | Time: {time_taken*1000:.2f}ms | LR: {lr_schedule[iter]:.3e} | Shard Index: {self.data_loader.shard_indx:1d} | {self.config.model_name}")

        print("Training Complete")