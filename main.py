import os
import random
import numpy as np
import datetime
import pytz
from tqdm import tqdm
from pprint import pprint

import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn

import trainers
from pengi import pengi


from utils import trainer
from utils.utils import print_total_time, get_args, get_dataloaders, get_model, setup_logging, get_scores, print_scores, save_scores, load_model, print_scores_b2n, save_b2n_scores

# to solve  the issue of : the current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):

    print(f"\n\n{'Model:':<10}{args.model_name.upper()}")
    print(f"{'Dataset:':<10}{args.dataset_root.split('/')[-1]}")
    print(f"{'Seed:':<10}{args.seed}\n\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.process_audio_fn = pengi.preprocess_audio

    # to ensure reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    train_dataloader, test_dataloader = get_dataloaders(args)
    args.classnames = train_dataloader.dataset.classnames
    if args.is_half_labels:
        args.all_classnames = test_dataloader.dataset.classnames
        args.train_classnames = train_dataloader.dataset.classnames
        args.new_classnames = args.all_classnames[len(args.train_classnames):]
    else:
        args.all_classnames = args.classnames
        args.train_classnames = args.classnames
        assert train_dataloader.dataset.classnames == test_dataloader.dataset.classnames, "Classnames in train and test datasets are different."

    model = get_model(args, pengi, trainers)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    print("\nArguments:\n")
    for arg in vars(args): print(f"{arg:<25}: {getattr(args, arg)}")
    print("\n\n")


    if args.eval_only:
        if args.model_name != "zeroshot":
            args.load_model_abs_path=f"weights/b2n/{args.model_name}/n_shots_16_lr_0.0125/{args.exp_name}-SEED{args.seed}.pth"
            print(args.load_model_abs_path)
            load_model(args, model)
        # if "zeroshot" not in args.model_name: load_model(args, model)
        if args.is_half_labels:
            loss_list, actual_labels_list, predicted_labels_list, predicted_labels_b2n_list = trainer.run_b2n_evaluation(model, test_dataloader, criterion, device, args)
            total_loss, new_loss, base_loss = loss_list
            new_actual_labels, base_actual_labels = actual_labels_list
            new_predicted_labels, base_predicted_labels = predicted_labels_list
            new_predicted_labels_b2n, base_predicted_labels_b2n = predicted_labels_b2n_list
            total_accuracy, total_f1_score, total_precision, total_recall =  get_scores(np.concatenate([new_actual_labels, base_actual_labels]), np.concatenate([new_predicted_labels, base_predicted_labels]), args.all_classnames)
            new_actual_labels = [x-len(args.train_classnames) for x in new_actual_labels]
            new_accuracy, new_f1_score, new_precision, new_recall =  get_scores(new_actual_labels, new_predicted_labels_b2n, args.new_classnames)
            base_accuracy, base_f1_score, base_precision, base_recall =  get_scores(base_actual_labels, base_predicted_labels_b2n, args.train_classnames)
            total_scores = [total_accuracy, total_f1_score, total_precision, total_recall]
            new_scores = [new_accuracy, new_f1_score, new_precision, new_recall]
            base_scores = [base_accuracy, base_f1_score, base_precision, base_recall]
            harmonic_score = 2 * (new_accuracy * base_accuracy) / (new_accuracy + base_accuracy)
            eval_loss_list = [total_loss, new_loss, base_loss]
            
            print(f"\n\n-------------------------------\nTest Evaluation\n-------------------------------\n")
            print_scores_b2n(total_scores, new_scores, base_scores, harmonic_score, eval_loss_list) 
            
        else:
            test_loss, actual_labels, predicted_labels = trainer.run_evaluation(model, test_dataloader, criterion, device, args)
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTest Evaluation\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, test_loss)
        if args.do_logging:
            print("Saving Results ...") 
            if args.is_half_labels:
                save_b2n_scores(args.seed, -1, total_scores, new_scores, base_scores, harmonic_score, eval_loss_list, args.json_file_path)
            else:
                save_scores(args.seed, -1, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
            print("Results Saved\n\n")
    else:
        try:
            for p in model.prompt_learner.text_encoder.parameters():
                p.requires_grad = False
        except:
            pass
        optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=0.9)

        print("=== Trainable parameters in model.prompt_learner ===")
        total = 0
        for name, p in model.prompt_learner.named_parameters():
            if p.requires_grad:
                n = p.numel()
                total += n
                print(f"{name:60s}  shape={tuple(p.shape)}  numel={n}")
        print(f"Total trainable numel in prompt_learner: {total}")

        trainer.run_training(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=args.n_epochs, args=args)



if __name__ == "__main__":

    args = get_args()
    log_file = setup_logging(args)

    print("\n\n##############################################")
    print("Spectral Tuning in Audio Language Models")
    print("##############################################\n\n")
    date_now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")}  KST\n')

    main(args)

