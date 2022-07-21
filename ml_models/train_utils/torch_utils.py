import torch 
import numpy as np
import random
import copy 
import time

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def weight_init(m, initializer = nn.init.xavier_uniform_):
    # https://discuss.pytorch.org/t/crossentropyloss-expected-object-of-type-torch-longtensor/28683/8?u=ptrblck
    # https://pytorch.org/docs/stable/nn.init.html
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        initializer(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
# model = model.apply(weight_init)


def train(model, loss_fn, optimizer, scheduler, dataloader, val_dataloader=None, epochs = 10, patience=5, debug = False, device='cpu', target = 'reg', verbose=True):
    best_loss = 0
    best_model = None
    cur_patience = 0
    train_loss_list = []
    val_loss_list = []

    print('start training....\n')
    if verbose:
        if target == 'reg':
            print(
                f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Elapsed':^9} | {'Cur lr':^9}" 
            )
        else:
            print(
                f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9} | {'Cur lr':^9}" 
            )
        print("-"*95)
    
    for epoch_i in range(epochs):
        #training....
        t0_epoch = time.time()
        total_loss = 0

        # put the model into the training mode
        model.train()
        for step, batch in enumerate(dataloader):
            # load batch to gpu
            b_tuple = tuple(t.to(device) for t in batch)
            b_labels = b_tuple[-1]
            b_tuple = b_tuple[:-1]

            # zero out any previously calculated gradients 
            model.zero_grad()
            # forward pass
            logits = model(*b_tuple)
            # comput loss and accumulate
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            # perform a backward pass to cal gradients 
            loss.backward()
            # update parameters 
            optimizer.step()
            if debug:
                break
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        # cal the average loss over the entire training data
        avg_train_loss = total_loss / len(dataloader)

        ##
        # Evaluate....
        ##
        if val_dataloader is not None:
            val_loss, val_acc, _ = evaluate(model, loss_fn, val_dataloader, device, target)

            # track the best acc
            if val_loss > best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
                cur_patience = 0
            else:
                cur_patience += 1 
            
            # print the performance:
            time_elapsed = time.time() - t0_epoch
            if verbose:
                if target == 'reg':
                    print(
                        f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {time_elapsed:^9.2f} | {cur_lr:^9.4f}"
                    )
                else:
                    print(
                        f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_acc:^9.2f} | {time_elapsed:^9.2f} | {cur_lr:^9.4f}"
                    )
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(val_loss)

        if cur_patience == patience:
            print('early stopping..., Best loss is ', round(best_loss, 4))
            return best_model
    
    print('\n')
    print(f"Training complete! Best loss: {best_loss:.4f}.")

    return best_model, [train_loss_list, val_loss_list]


def evaluate(model, loss_fn, val_dataloader, device, target = 'reg'):
    model.eval()

    # tracking variables 
    val_acc = []
    val_loss = 0
    val_labels_list = []
    val_pred_logits_list = []

    # for each batch in our dev set....
    for i, batch in enumerate(val_dataloader):
        # load 
        b_tuple = tuple(t.to(device) for t in batch)
        b_labels = b_tuple[-1]
        b_tuple = b_tuple[:-1]

        # compute logits 
        with torch.no_grad():
            logits = model(*b_tuple)
        loss = loss_fn(logits, b_labels)
        # get predictions
        preds = torch.argmax(logits, dim=1).flatten()
        
        ## calculate the acc rate
        if target == 'clf':
            acc = (preds == b_labels).cpu().numpy().mean()*100
            val_acc.append(acc)
        val_loss+=loss.item()
        val_labels_list.append(b_labels)

        val_pred_logits_list.append(logits)

    val_loss = val_loss / len(val_dataloader)
    
    if target=='clf':
        val_acc = np.mean(val_acc)
    else:
        val_acc = None

    val_labels = torch.cat(val_labels_list)
    val_pred_proba = torch.softmax(torch.cat(val_pred_logits_list), 1)

    deep_info = [val_labels, 
                 val_pred_proba]

    return val_loss, val_acc, deep_info
    
# def validation(dataloader, model, loss_fn, device, target = 'reg'):
#     size = len(dev_dataloader.dataset)
#     num_batches = len(dev_dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dev_dataloader:
#             X, y = X.to(device), torch.flatten(y.to(device))
#             pred = model(X)

#             if target == 'clf':
#                 pred = pred.argmax(1)
#                 pred = map_class_to_y(pred)
#             #else do nothing...
#             pred_arr = pred.cpu().detach().numpy()
#             y_arr = y.cpu().detach().numpy()
#             test_loss += MAE_with_log_smooth(y_arr, pred_arr) * X.size()[0]
#             correct += count_of_correct_torch(y, pred)

#     test_loss /= size
#     correct = (correct.float() / size).item()
#     return test_loss, correct, [pred, y]

###test 跟 validate 相同


###
def train_multiloss(model, loss_fn_list, optimizer, scheduler, dataloader, val_dataloader=None, epochs = 10, patience=5, debug = False, device='cpu', target = 'reg'):
    best_accuracy = 0
    best_roc_auc = 0
    best_model = None
    cur_patience = 0

    print('start training....\n')
    print(
        f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9} | {'Cur lr':^9}" 
    )
    print("-"*95)
    
    for epoch_i in range(epochs):
        #training....
        t0_epoch = time.time()
        total_loss = 0

        # put the model into the training mode
        model.train()
        for step, batch in enumerate(dataloader):
            # load batch to gpu
            b_tuple = tuple(t.to(device) for t in batch)
            b_labels = b_tuple[-1]
            b_tuple = b_tuple[:-1]

            # zero out any previously calculated gradients 
            model.zero_grad()
            # forward pass
            outputs = model(*b_tuple)
            # comput loss and accumulate
            loss1 = loss_fn_list[0](outputs[0], b_tuple[0])
            loss2 = loss_fn_list[1](outputs[1], b_labels)
            loss3 = loss_fn_list[2](outputs[2], b_labels)
            
            total_loss += loss1.item() + loss2.item() + loss3.item()
            # perform a backward pass to cal gradients 
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss3.backward()

            # update parameters 
            optimizer.step()
            if debug:
                break
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        # cal the average loss over the entire training data
        avg_train_loss = total_loss / len(dataloader)

        ##
        # Evaluate....
        ##
        if val_dataloader is not None:
            val_loss, val_acc = evaluate_multiloss(model, loss_fn_list, val_dataloader, device)

            # track the best acc
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model = copy.deepcopy(model)
                cur_patience = 0
            else:
                cur_patience += 1 
            
            # print the performance:
            time_elapsed = time.time() - t0_epoch
            print(
                f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_acc:^9.2f} | {time_elapsed:^9.2f} | {cur_lr:^9.4f}"
            )

        if cur_patience == patience:
            print('early stopping..., Best acc is ', round(best_accuracy, 4))
            return best_model
    
    print('\n')
    print(f"Training complete! Best acc: {best_accuracy:.4f}%.")

    return best_model


def evaluate_multiloss(model, loss_fn_list, val_dataloader, device):
    model.eval()

    # tracking variables 
    val_acc = []
    val_loss = 0
    val_labels_list = []
    val_pred_logits_list = []

    # for each batch in our dev set....
    for batch in val_dataloader:
        # load 
        b_tuple = tuple(t.to(device) for t in batch)
        b_labels = b_tuple[-1]
        b_tuple = b_tuple[:-1]

        # compute logits 
        with torch.no_grad():
            result = model(*b_tuple)
        logits = result[-1]
        loss = loss_fn_list[-1](logits, b_labels)
        # get predictions
        preds = torch.argmax(logits, dim=1).flatten()

        ## calculate the acc rate
        
        acc = (preds == b_labels).cpu().numpy().mean()*100
        val_loss+=loss.item()
        val_acc.append(acc)
        val_labels_list.append(b_labels)

        val_pred_logits_list.append(logits)

    val_loss = val_loss / len(val_dataloader)
    val_acc = np.mean(val_acc)

    val_labels = torch.cat(val_labels_list)
    val_pred_proba = torch.softmax(torch.cat(val_pred_logits_list), 1)

    return val_loss, val_acc

