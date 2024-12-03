import torch

SEMANTIC_ANALOGY = 'Semantic Analogy'
UPAMA = 'Upama'

# Please mention a single file at a time in the list 'checkpoints_names'

# checkpoints_names = [
#                         './CheckPoints/FT-Ext-Pipe-21-CFG-0-Semantic Analogy-Iter-7200-2024-09-25 19-51-07.pth'
#                     ]

checkpoints_names = [
                        './CheckPoints/FT-Ext-Pipe-22-CFG-0-Upama-Iter-10000-2024-09-25 20-32-28.pth'
                    ]

task = UPAMA

all_tot_loss = []
all_val_loss = []
all_norms = []

tot_loss_file = open("./Total_Loss.csv", "w")
val_loss_file = open("./Val_Loss.csv", "w")
norm_file = open("./Norms.csv", "w")

if task == UPAMA:
    upama_eval_file = open("./Upama_Eval.csv", "w")
    all_upama_acc = []
    all_upama_precision = []
    all_upama_recall = []
    all_upama_f1 = []
elif task == SEMANTIC_ANALOGY:
    sem_anlg_eval_file = open("./SemAnlg_Eval.csv", "w")
    all_sem_acc = []
    all_sem_precision = []
    all_sem_recall = []
    all_sem_f1 = []

for checkpoints_name in checkpoints_names:
    checkpoint = torch.load(checkpoints_name)

    model_name = checkpoint['train_config'].model_name if checkpoint['train_config'].model_name != None else 'None'
    optimizer_name = checkpoint['train_config'].optimizer_name if checkpoint['train_config'].optimizer_name != None else 'None'
    max_lr = checkpoint['train_config'].max_lr if checkpoint['train_config'].max_lr != None else 'None'
    min_lr = checkpoint['train_config'].min_lr if checkpoint['train_config'].min_lr != None else 'None'
    warmup_iters = checkpoint['train_config'].warmup_iters if checkpoint['train_config'].warmup_iters != None else 'None'
    tokens_batch_size = checkpoint['train_config'].tokens_batch_size
    weight_decay = checkpoint['train_config'].weight_decay

    print(model_name)
    print(optimizer_name)
    print(f"max_lr: {max_lr}")
    print(f"min_lr: {min_lr}")
    print(f"warmup_iters: {warmup_iters}")
    print(f"Tokens Batch Size: {tokens_batch_size}")
    print(f"Weight Decay: {weight_decay}")
    print('================================================================')

    all_tot_loss.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26]] + checkpoint['total_loss_list'])
    all_val_loss.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26]] + checkpoint['val_losses'])
    all_norms.append([optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26]] +  checkpoint['total_norm_list'])

    if task == UPAMA:
        upama_eval = checkpoint['upama_eval']

        all_upama_acc.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])
        all_upama_precision.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])
        all_upama_recall.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])
        all_upama_f1.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])

        for metric in upama_eval:
            all_upama_acc.append(metric[0])
            all_upama_precision.append(metric[1])
            all_upama_recall.append(metric[2])
            all_upama_f1.append(metric[3])

    elif task == SEMANTIC_ANALOGY:
        sem_anlg_eval = checkpoint['sem_anlg_eval']
        
        all_sem_acc.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])
        all_sem_precision.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])
        all_sem_recall.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])
        all_sem_f1.append(optimizer_name + '-' + str(max_lr) + '-' + str(min_lr) + '-' + str(tokens_batch_size) + '-' + str(weight_decay) + '-' + checkpoints_name[14:26])
        
        for metric in sem_anlg_eval:
            all_sem_acc.append(metric[0])
            all_sem_precision.append(metric[1])
            all_sem_recall.append(metric[2])
            all_sem_f1.append(metric[3])

    print('Done')

    del checkpoint

all_tot_loss = list(zip(*all_tot_loss))
all_val_loss = list(zip(*all_val_loss))
all_norms = list(zip(*all_norms))

for sub_list in all_tot_loss:
    for item in sub_list:
        tot_loss_file.write(str(item) + ',')
        
    tot_loss_file.write('\n')
tot_loss_file.close()

for sub_list in all_val_loss:
    for item in sub_list:
        val_loss_file.write(str(item) + ',')
        
    val_loss_file.write('\n')
val_loss_file.close()

for sub_list in all_norms:
    for item in sub_list:
        norm_file.write(str(item) + ',')

    norm_file.write('\n')
norm_file.close()

if task == UPAMA:
    for acc, precision, recall, f1 in zip(all_upama_acc, all_upama_precision, all_upama_recall, all_upama_f1):
        upama_eval_file.write(str(acc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + (','))
        
        upama_eval_file.write('\n')
    upama_eval_file.close()

elif task == SEMANTIC_ANALOGY:
    for acc, precision, recall, f1 in zip(all_sem_acc, all_sem_precision, all_sem_recall, all_sem_f1):
        sem_anlg_eval_file.write(str(acc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + (','))
        
        sem_anlg_eval_file.write('\n')
    sem_anlg_eval_file.close()