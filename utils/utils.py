import os
import sys
import json
import pytz
import argparse
import datetime
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from .dataset import FewShotDataset

def get_model(args, pengi, trainers):
    print(f"Using Method: '{args.model_name.upper()}'\n")

    try:
        model = getattr(trainers, args.model_name)(args, pengi)
    except AttributeError:
        raise ValueError(f"Model '{args.model_name}' not found in trainers. Please check if 'trainers/{args.model_name}.py' exists.")

    return model


def get_dataloaders(args):
    train_dataset = FewShotDataset(args.dataset_root, 'train' , num_shots=args.num_shots, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, is_half_labels=args.is_half_labels, train_csv=args.train_csv, test_csv=args.test_csv, args=args)
    test_dataset  = FewShotDataset(args.dataset_root, 'test'  , num_shots=-1, repeat=args.repeat , process_audio_fn=args.process_audio_fn, resample=args.resample, is_half_labels=False, train_csv=args.train_csv, test_csv=args.test_csv, args=args)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def save_model(args, model, save_model_path):
    print(f"Saving Context Weights for Method: '{args.model_name.upper()}'\n")
    checkpoint = {'prompt_learner': model.prompt_learner.state_dict()}
    torch.save(checkpoint, save_model_path)


    
def load_model(args, model):
    load_model_path = get_load_model_path(args)
    checkpoint = torch.load(load_model_path)

    if args.cross_dataset:
        pl_ckpt = checkpoint['prompt_learner']


        if 'svd' in args.model_name:
            model.prompt_learner.ctx = nn.Parameter(pl_ckpt['ctx'])
            model.prompt_learner.Vh_init = nn.Parameter(pl_ckpt['Vh_init'])
            print("Loading the weights of the ctx and Vh_init")
        elif 'text_adapter' in args.model_name:
            # Extract adapter weights from the prompt_learner state_dict
            adapter_state_dict = {k.replace('adapter.', ''): v for k, v in pl_ckpt.items() if k.startswith('adapter.')}
            model.prompt_learner.adapter.load_state_dict(adapter_state_dict)
            
        else:
            model.prompt_learner.ctx.data.copy_(pl_ckpt['ctx'])
            print("Loading the weights of the ctx")
        if any(key.startswith('meta_net.') for key in pl_ckpt.keys()):
            # linear1
            model.prompt_learner.meta_net.linear1.weight.data.copy_(
                pl_ckpt['meta_net.linear1.weight']
            )
            model.prompt_learner.meta_net.linear1.bias.data.copy_(
                pl_ckpt['meta_net.linear1.bias']
            )
            # linear2
            model.prompt_learner.meta_net.linear2.weight.data.copy_(
                pl_ckpt['meta_net.linear2.weight']
            )
            model.prompt_learner.meta_net.linear2.bias.data.copy_(
                pl_ckpt['meta_net.linear2.bias']
            )
            print("Loading the weights of the meta_net")

    else:
        model.prompt_learner.load_state_dict(checkpoint['prompt_learner'], strict=False)

    

def get_save_model_path(args):
    save_model_path = args.log_dir.replace('logs', 'weights')
    
    if not os.path.exists(save_model_path): os.makedirs(save_model_path)
    save_model_path = os.path.join(save_model_path, f"{args.exp_name+'-SEED'+str(args.seed)}.pth")
    return save_model_path

def get_load_model_path(args):
        if args.load_model_abs_path is not None:
            load_model_path = args.load_model_abs_path
        
        if not os.path.exists(load_model_path): 
            raise ValueError(f"Model file '{load_model_path}' does not exist. Specify the correct path to the model file.")
        
        return load_model_path


def get_args():
    parser = argparse.ArgumentParser(description='SubT')
    parser.add_argument('--model_name', type=str, default='', help='Model Name (default: None)', required=True)
    parser.add_argument('--save_model', help='Save the trained model (default: False)', action='store_true')
    parser.add_argument('--save_model_path', type=str, default='weights', help='Path to save the trained model (default: None)')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to the pre-trained model (learnable context) weights (default: None)')
    parser.add_argument('--load_model_abs_path', type=str, default=None, help='Absolute path to the pre-trained model (learnable context) weights (default: None)')
    parser.add_argument('--dataset_root', type=str, default='', help='Path to the dataset root directory (default: None)', required=True)
    parser.add_argument('--train_csv', type=str, default='', help='Path to the train csv file (default: None)', required=True)
    parser.add_argument('--test_csv', type=str, default='', help='Path to the test csv file (default: None)', required=True)
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (default: 0)')
    parser.add_argument('--freq_test_model', type=int, default=10, help='Frequency of testing the model (default: 10)')
    parser.add_argument('--spec_aug', help='Apply Spectrogram Augmentation (default: False)', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.0125, help='Learning Rate (default: 0.0125)')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed (default: 0)')
    parser.add_argument('--eval_only', help='Evaluate the model only (default: False)', action='store_true')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name', required=True)
    parser.add_argument('--do_logging', help='Disable Logging (default: False)', action='store_true')
    parser.add_argument('--prompt_prefix', type=str, default='The is a recording of', help='Prompt Prefix (default: The is a recording of )')

    parser.add_argument('--n_ctx', type=int, default=16, help='Number of context tokens (default: 16)')
    parser.add_argument('--ctx_dim', type=int, default=512, help='Dimension of the context vector (default: 512)')

    # Few-Shot Learning Arguments
    parser.add_argument('--num_shots', type=int, default=16, help='Number of shots (default: 16)')
    parser.add_argument('--resample', type=bool, default=True, help='Resample samples if needed (default: True)')
    parser.add_argument('--repeat', type=bool, default=False, help='Repeat samples if needed (default: False)')
    parser.add_argument('--is_half_labels', action='store_true', help='Use half of the labels (default: False)')

    # Cross-dataset Arguments
    parser.add_argument('--cross_dataset', action='store_true', help='Cross dataset task')

    args = parser.parse_args()

    # Sanity check on Arguments
    if not os.path.exists(args.dataset_root):
        raise ValueError(f"\n\nDirectory '{args.dataset_root}' does not exist. Specify the correct path to the dataset.\n\n")
    if args.save_model and not os.path.exists(args.save_model_path):
        raise ValueError(f"\n\nDirectory '{args.save_model_path}' does not exist. Create or specify the correct the directory to save the trained model.\n\n")
    if args.eval_only:
        load_model_path = get_load_model_path(args)
        if not os.path.exists(load_model_path): raise ValueError(f"\n\nEvaluation Mode: Model file '{load_model_path}' does not exist. Specify the correct path to the model file.\n\n")
    
    if args.model_name == 'zeroshot': args.eval_only = True
    if args.is_half_labels == False:
        args.save_model = True
    return args



def print_total_time(now_start, now_end):
	print(f'\nEnd Time & Date = {now_end.strftime("%I:%M %p")} , {now_end.strftime("%d_%b_%Y")}\n')
	duration_in_s = (now_end - now_start).total_seconds()
	days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
	hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
	minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
	seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds
	print(f"Total Time => {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds\n\n")




def print_dataset_info(train_dataloader, test_dataloader):
	n_classes = train_dataloader.dataset.n_classes
	num_batches_train = len(train_dataloader)
	num_batches_test = len(test_dataloader)

	print("\n########################\nDataset Information\n########################\n")
	print("Length of the Train Dataset: ", len(train_dataloader.dataset))
	print("Length of the Test Dataset: ", len(test_dataloader.dataset))
	print("Train Batch Size: ", train_dataloader.batch_size)
	print("Test Batch Size: ", test_dataloader.batch_size)
	print("Number of Batches in Train Dataloader: ", num_batches_train)
	print("Number of Batches in Test Dataloader: ", num_batches_test)
	print("Number of Classes: ", n_classes)
     

def get_scores(actual_labels, predicted_labels, classnames):
    cls_report = classification_report(actual_labels, predicted_labels, target_names=classnames, output_dict=True)
    accuracy = cls_report['accuracy']
    f1_score = cls_report['macro avg']['f1-score']
    precision = cls_report['macro avg']['precision']
    recall = cls_report['macro avg']['recall']
    return accuracy, f1_score, precision, recall


def print_scores(accuracy, f1_score, precion, recall, avg_loss):
    print(f"{'Accuracy':<15} = {accuracy:0.4f}")
    print(f"{'F1-Score':<15} = {f1_score:0.4f}")
    print(f"{'Precision':<15} = {precion:0.4f}")
    print(f"{'Recall':<15} = {recall:0.4f}")
    print(f"{'Average Loss':<15} = {avg_loss:0.4f}\n\n")
    
def print_scores_b2n(total_scores, new_scores, base_scores, harmonic_score, eval_loss_list, train_loss=None):
    print(f"{'Base Accuracy':<15} = {base_scores[0]:0.4f}")
    print(f"{'New Accuracy':<15} = {new_scores[0]:0.4f}")
    print(f"{'Total Accuracy':<15} = {total_scores[0]:0.4f}")
    print(f"{'Harmonic Score':<15} = {harmonic_score:0.4f}")
    print(f"{'Total F1-Score':<15} = {total_scores[1]:0.4f}")
    print(f"{'Base F1-Score':<15} = {base_scores[1]:0.4f}")
    print(f"{'New F1-Score':<15} = {new_scores[1]:0.4f}")
    print(f"{'Total Precision':<15} = {total_scores[2]:0.4f}")
    print(f"{'Base Precision':<15} = {base_scores[2]:0.4f}")
    print(f"{'New Precision':<15} = {new_scores[2]:0.4f}")
    print(f"{'Total Recall':<15} = {total_scores[3]:0.4f}")
    print(f"{'Base Recall':<15} = {base_scores[3]:0.4f}")
    print(f"{'New Recall':<15} = {new_scores[3]:0.4f}")
    if train_loss is not None: print(f"{'Train AVG Loss':<15} = {train_loss:0.4f}")
    print(f"{'Eval Total Loss':<15} = {eval_loss_list[0]:0.4f}")
    print(f"{'Eval Base Loss':<15} = {eval_loss_list[2]:0.4f}")
    print(f"{'Eval New Loss':<15} = {eval_loss_list[1]:0.4f}")
    

def save_scores(seed, epoch, accuracy, f1_score, precision, recall, avg_loss, json_file_path):
    if not os.path.exists(json_file_path):
        # create the file if it doesn't exist
        with open(json_file_path, "w") as file:
            file.write("{}")
        
    # load existing results
    with open(json_file_path, "r") as file:
        scores_json = json.load(file)

    scores_json[f"seed_{seed}"] = {"accuracy": f"{accuracy:0.4f}", "f1_score": f"{f1_score:0.4f}", "precision": f"{precision:0.4f}", "recall": f"{recall:0.4f}", "avg_loss": f"{avg_loss:0.4f}", "epoch": epoch}

    for metric in scores_json[f"seed_{seed}"].keys():
        if metric != 'epoch': scores_json[f"seed_{seed}"][metric] = float(scores_json[f"seed_{seed}"][metric])


    # save updated results
    with open(json_file_path, "w") as file:
        json.dump(scores_json, file, indent=2) 

def save_b2n_scores(seed, epoch, total_scores, new_scores, base_scores, harmonic_score, eval_loss_list, json_file_path):
    if not os.path.exists(json_file_path):
        # create the file if it doesn't exist
        with open(json_file_path, "w") as file:
            file.write("{}")
    with open(json_file_path, "r") as file:
        scores_json = json.load(file)
    scores_json[f"seed_{seed}"] = {"accuracy": {"total": total_scores[0], "new": new_scores[0], "base": base_scores[0]}, 
                                   "f1_score": {"total": total_scores[1], "new": new_scores[1], "base": base_scores[1]}, 
                                   "precision": {"total": total_scores[2], "new": new_scores[2], "base": base_scores[2]}, 
                                   "recall": {"total": total_scores[3], "new": new_scores[3], "base": base_scores[3]},
                                   "harmonic_score": harmonic_score, "eval_loss": {"total": eval_loss_list[0].item(), "new": eval_loss_list[1].item(), "base": eval_loss_list[2].item()},
                                   "epoch": epoch}

    # save updated results
    with open(json_file_path, "w") as file:
        json.dump(scores_json, file, indent=2) 


# Decorator to measure the time taken by a function
def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        
        duration_in_s = end - start
        days  = divmod(duration_in_s, 86400)   # Get days (without [0]!)
        hours = divmod(days[1], 3600)          # Use remainder of days to calc hours
        minutes = divmod(hours[1], 60)         # Use remainder of hours to calc minutes
        seconds = divmod(minutes[1], 1)        # Use remainder of minutes to calc seconds


        date_now = datetime.datetime.now(pytz.timezone('Asia/Dubai'))
        print(f'\n\nTime & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")}  GST')
        print(f"\nTotal Time => {int(days[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds\n\n")
        return result
        
    return wrapper


############################################################################################################
# Logging Functions
############################################################################################################


# Define a Tee class to duplicate output to both stdout and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
 
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
 
    def flush(self):
        for file in self.files:
            file.flush()

# Define a function to redirect stdout and stderr to a log file
def redirect_output_to_log(log_file):
    log = open(log_file, 'a')
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)

    return log

# Define a function to setup logging
def setup_logging(args):
    if args.is_half_labels:
        log_dir = os.path.join('logs', 'b2n', args.model_name)
    else:
        log_dir = os.path.join('logs', 'general', args.model_name) 
    
    log_dir = os.path.join(log_dir, f'n_shots_{args.num_shots}_lr_{args.lr}')
    
    if args.cross_dataset:
        log_dir = '/'.join(args.load_model_abs_path.split('/')[:-1]).replace('weights', 'logs').replace('general', 'cross_dataset')
        
    print(f"Logging to directory: {log_dir}")
    args.log_dir = log_dir
    
    if args.do_logging:
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, f"{args.exp_name+'-SEED'+str(args.seed)}.log")
        log_file_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)
        if os.path.exists(log_file_path): os.remove(log_file_path)
        json_file_path = os.path.join(log_dir, f"{args.exp_name}.json")
        args.json_file_path = json_file_path
        print(f"\nLogging to '{log_file_path}'\n")
        log_file = redirect_output_to_log(log_file_path) 
    else:
        log_file =None

    return log_file