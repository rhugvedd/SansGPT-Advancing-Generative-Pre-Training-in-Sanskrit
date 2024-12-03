import torch
from Transformer import *
from BPETokenizer import BPETokenizer
from DataLoader import DataLoader
import torch.nn.functional as F

SEMANTIC_ANALOGY = 'Semantic Analogy'
UPAMA = 'Upama'

# Mention the task here
fine_tune_task = UPAMA

vocab_size = 12000
d_model = 768
dec_context_size = 512
pos_enc_dropout = 0
num_decoder_blocks = 12
num_heads = 12
drop_prob = 0
d_feedfwd = 512 * 4
pre_norm = True
mask_attention = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_path = "./Final Tokenizer/"
load_merge_info_name = 'Final-Corpus-Tokenizer-Merge-Info-NL-12000-2024-08-31 03-04-04'
load_vocab_name = 'Final-Corpus-Tokenizer-Vocab-NL-12000-2024-08-31 03-04-04'

batch_size = 1
data_path = "./Eval_Tasks/Eval_Tensors/"

num_generations = 10
topk_sampling_k = 50

if fine_tune_task == UPAMA:
    checkpoint_path = './Fine Tuning CheckPoints/FT-Ext-Pipe-22-CFG-0-Upama-Iter-10000-2024-09-25 20-32-28.pth'
        
    x_train_name = 'X-Eval-12000-Upama-V0.9-Train-2024-09-17 01-08-40'
    y_train_name = 'Y-Eval-12000-Upama-V0.9-Train-2024-09-17 01-08-40'
    mask_train_name = 'Mask-Eval-12000-Upama-V0.9-Train-2024-09-17 01-08-40'
    x_val_name = 'X-Eval-12000-Upama-V0.9-Val-2024-09-17 01-08-40'
    y_val_name = 'Y-Eval-12000-Upama-V0.9-Val-2024-09-17 01-08-40'
    mask_val_name = 'Mask-Eval-12000-Upama-V0.9-Val-2024-09-17 01-08-40'

elif fine_tune_task == SEMANTIC_ANALOGY:
    checkpoint_path = './Fine Tuning CheckPoints/FT-Ext-Pipe-21-CFG-0-Semantic Analogy-Iter-7200-2024-09-25 19-51-07.pth'

    x_train_name = 'X-Eval-12000-Semantic-Analogies-Train-2024-09-14 22-18-08'
    y_train_name = 'Y-Eval-12000-Semantic-Analogies-Train-2024-09-14 22-18-08'
    mask_train_name = 'Mask-Eval-12000-Semantic-Analogies-Train-2024-09-14 22-18-08'
    x_val_name = 'X-Eval-12000-Semantic-Analogies-Val-2024-09-14 22-18-08'
    y_val_name = 'Y-Eval-12000-Semantic-Analogies-Val-2024-09-14 22-18-08'
    mask_val_name = 'Mask-Eval-12000-Semantic-Analogies-Val-2024-09-14 22-18-08'

def generate_analogies():

    data_loader.shuffle('val', reset_batch_index = True)

    for gen_iter in range(num_generations):
        X, Y = data_loader.get_val_batch(device)
        mask = data_loader.get_val_mask(device)
        
        B, T = X.shape

        sep_pos_flipped = (torch.flip(X, dims=[1])[0] == sep_token).nonzero(as_tuple=True)[0]
        sep_index_original = (T - 1 - sep_pos_flipped[0].item()) if len(sep_pos_flipped) > 0 else -1

        X = X[:, :(sep_index_original + 1)]

        for tok in X[0]:
            if tok == bos_token:
                # print('<bos> ', end='')
                print('', end='')
            elif tok == eos_token:
                print('<eos>', end='')
            elif tok == sep_token:
                print(' : ', end='')
            elif tok == pad_token:
                print('<pad>', end='')
            else:
                print(Tokenizer.Decode([tok.tolist()]), end="")

        while True:
            
            X = X[:, -dec_context_size:]
            
            preds = Generator(X)

            preds = preds[:, -1, :]

            preds = F.softmax(preds, dim=-1)

            top_probs, top_indices = preds.topk(topk_sampling_k, dim=-1)

            next_tok_id = torch.multinomial(top_probs, num_samples=1)

            next_tok = top_indices.gather(-1, next_tok_id)
            next_tok = next_tok
            
            if next_tok[0,0] == eos_token:
                # print(' <eos>')
                print('')
                break
            
            print(Tokenizer.Decode([next_tok[0,0].tolist()]), end="")

            X = torch.cat((X, next_tok), dim=1)

    Generator.train()

# ===========================================================================================================================

def generate_upamas():

    data_loader.shuffle('val', reset_batch_index = True)

    for gen_iter in range(num_generations):
        X, Y = data_loader.get_val_batch(device)
        mask = data_loader.get_val_mask(device)
        
        B, T = X.shape

        sep_pos_flipped = (torch.flip(X, dims=[1])[0] == sep_token).nonzero(as_tuple=True)[0]
        sep_index_original = (T - 1 - sep_pos_flipped[0].item()) if len(sep_pos_flipped) > 0 else -1

        X = X[:, :(sep_index_original + 1)]
        cnt = 0

        for tok in X[0]:
            if tok == bos_token:
                # print('<bos> ', end='')
                print('', end='')
            elif tok == eos_token:
                print('<eos>', end='')
            elif tok == sep_token:
                # print(' : ', end='')
                # print('<sep> ', end='')
                print('', end='')
            elif tok == pad_token:
                print('<pad>', end='')
            elif tok == unk_token:
                print('<sep2>', end='')
            else:
                print(Tokenizer.Decode([tok.tolist()]), end="")
        
        print("\n\nWord Indicating Similarity: ", end='')

        while True:
            
            X = X[:, -dec_context_size:]
            
            preds = Generator(X)

            preds = preds[:, -1, :]

            preds = F.softmax(preds, dim=-1)

            top_probs, top_indices = preds.topk(topk_sampling_k, dim=-1)

            next_tok_id = torch.multinomial(top_probs, num_samples=1)

            next_tok = top_indices.gather(-1, next_tok_id)
            next_tok = next_tok
            
            if next_tok[0,0] == eos_token:
                # print(' <eos>\n\n')
                print('')
                break
            elif next_tok[0,0] == unk_token:
                # print(' <sep2>', end="")
                print('', end='')
                
                cnt += 1

                if cnt % 2 == 0:
                    if cnt == 2:
                        print("\nUpameya: ", end='')
                    elif cnt == 4:
                        print("\nUpamaana: ", end='')

            else:
                if cnt % 2 == 0:
                    print(Tokenizer.Decode([next_tok[0,0].tolist()]), end='')

            X = torch.cat((X, next_tok), dim=1)

        print("===========================================================================================================")

    Generator.train()

# ===========================================================================================================================

print("Initializing")

Generator = Decoder(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    context_size=dec_context_size,
                    pos_enc_dropout=pos_enc_dropout,
                    num_decoder_blocks=num_decoder_blocks,
                    num_heads=num_heads,
                    drop_prob=drop_prob,
                    d_feedfwd=d_feedfwd,
                    pre_norm=pre_norm,
                    mask_attention=mask_attention
                )

m = Generator.to(device)

print(f"No. of Parameters: {sum(p.numel() for p in m.parameters()) / 1e6} M parameters")
print("Loading Checkpoint")
checkpoint = torch.load(checkpoint_path)
Generator.load_state_dict(checkpoint['model_state_dict'])

Generator.eval()

Tokenizer = BPETokenizer()
Tokenizer.load(vocab_path, load_merge_info_name, load_vocab_name)

bos_token = Tokenizer.special_tok('<bos>')
sep_token = Tokenizer.special_tok('<sep>')
pad_token = Tokenizer.special_tok('<pad>')
eos_token = Tokenizer.special_tok('<eos>')
unk_token = Tokenizer.special_tok('<sep2>')

# ===========================================================================================================================

print("Loading Data")
data_loader = DataLoader(data_path)

data_loader.load_fine_tune_data (   
                        batch_size = batch_size,
                        x_train_name = x_train_name,
                        y_train_name = y_train_name,
                        mask_train_name = mask_train_name,
                        x_val_name = x_val_name,
                        y_val_name = y_val_name,
                        mask_val_name = mask_val_name
                    )

print("Data Loading Complete\n")

# ===========================================================================================================================

print("Ready to Generate\n")

if fine_tune_task == SEMANTIC_ANALOGY:
    generate_analogies()
elif fine_tune_task == UPAMA:
    generate_upamas()

print("Generation Complete\n")