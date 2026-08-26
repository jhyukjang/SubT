import os
import torch
import numpy as np
from tqdm import tqdm

from .utils import get_scores, print_scores, save_scores, timeit, save_model, get_save_model_path, print_scores_b2n, save_b2n_scores
from torch.nn import functional as F


def run_epoch(model, dataloader, optimizer, criterion, device, args=None):
    model.train()
    model.audio_encoder.base.htsat.bn0.eval()
    for p in model.audio_encoder.base.htsat.bn0.parameters():
        p.requires_grad = False
    losses = []
    actual_labels = []
    predicted_labels = []

    for i, (audio, label) in enumerate(dataloader):

        audio = audio.to(device).squeeze(1)
        label = label.to(device)
        model_output = model(audio)
        logits = model_output
        loss = criterion(logits, label)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        actual_labels.extend(label.cpu().numpy())
        predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_evaluation(model, dataloader, criterion, device, args):
    model.eval()

    losses = []
    actual_labels = []
    predicted_labels = []
    
    print("\n\nEvaluating the model ...")
    with torch.no_grad():
        for i, (audio, label) in enumerate(dataloader):
            print(f"Batch {i+1}/{len(dataloader)}")

            audio = audio.to(device).squeeze(1)
            label = label.to(device)
            model_output = model(audio)
            logits = model_output
            loss = criterion(logits, label)

            losses.append(loss.item())

            actual_labels.extend(label.cpu().numpy())
            predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels

def run_b2n_evaluation(model, dataloader, criterion, device, args):
    model.eval()

    new_logits = []
    base_logits = []
    new_targets = []
    base_targets = []
    
    for i, (audio, label) in enumerate(dataloader):
        audio = audio.to(device).squeeze(1)
        label = label.to(device)
        
        with torch.no_grad():
            logits = model(audio, is_test_b2n=args.is_half_labels)
            base_logits.append(logits[label<len(args.train_classnames)])
            new_logits.append(logits[label>=len(args.train_classnames)])
            new_targets.append(label[label>=len(args.train_classnames)])
            base_targets.append(label[label<len(args.train_classnames)])

    new_logits = torch.cat(new_logits)
    base_logits = torch.cat(base_logits)
    new_targets = torch.cat(new_targets)
    base_targets = torch.cat(base_targets)

    new_loss = F.cross_entropy(new_logits, new_targets)
    base_loss = F.cross_entropy(base_logits, base_targets)
    total_loss = new_loss + base_loss

    
    loss_list = [total_loss, new_loss, base_loss]
    actual_labels_list = [new_targets.cpu().numpy(), base_targets.cpu().numpy()]
    predicted_labels_list = [new_logits.argmax(dim=1).cpu().numpy(), base_logits.argmax(dim=1).cpu().numpy()]
    predicted_labels_b2n_list = [new_logits[:, len(args.train_classnames):].argmax(dim=1).cpu().numpy(), base_logits[:, :len(args.train_classnames)].argmax(dim=1).cpu().numpy()]
    
    return loss_list, actual_labels_list, predicted_labels_list, predicted_labels_b2n_list


@timeit
def run_training(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=50, args=None):
    
    best_train_loss = float('inf')
    best_epoch = -1

    for epoch in tqdm(range(epochs), total=epochs):

        train_loss, actual_labels, predicted_labels = run_epoch(model, train_dataloader, optimizer, criterion, device, args=args)

        if (epoch+1)%5 == 0:
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTrain Evaluation (Epoch {epoch + 1}/{epochs})\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, train_loss) 
            
        if (epoch+1)%1 == 0 and train_loss < best_train_loss:
            if not args.is_half_labels:
                test_loss, actual_labels, predicted_labels = run_evaluation(model, test_dataloader, criterion, device, args)
                accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
                print(f"\n\n-------------------------------\nTest Evaluation\n-------------------------------\n")
                print_scores(accuracy, f1_score, precision, recall, test_loss)

                if args.do_logging:
                    print("\n\nFinal Evaluation")
                    print("Saving Results ...")
                    if train_loss < best_train_loss:
                        best_train_loss = train_loss
                        best_epoch = epoch + 1
                        print(f"\n>> New best model saved at epoch {best_epoch} with train_loss = {best_train_loss:.4f}")
                        
                        save_scores(args.seed, epoch, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
                        print("Results Saved\n\n")

                        if args.save_model:
                            save_model_path = get_save_model_path(args)
                            save_model(args, model, save_model_path)
                            print(f"Model saved to {save_model_path}")
            else:
                loss_list, actual_labels_list, predicted_labels_list, predicted_labels_b2n_list = run_b2n_evaluation(model, test_dataloader, criterion, device, args)
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
                print_scores_b2n(total_scores, new_scores, base_scores, harmonic_score, eval_loss_list, train_loss) 
    
                if args.do_logging:
                    print("\n\nFinal Evaluation")
                    print("Saving Results ...")

                    if train_loss < best_train_loss:
                        best_train_loss = train_loss
                        best_epoch = epoch + 1
                        print(f"\n>> New best model saved at epoch {best_epoch} with train_loss = {best_train_loss:.4f}")
                        
                        if args.is_half_labels:
                            save_b2n_scores(args.seed, epoch, total_scores, new_scores, base_scores, harmonic_score, eval_loss_list, args.json_file_path)
                        else:
                            save_scores(args.seed, epoch, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
                        print("Results Saved\n\n")

                        if args.save_model:
                            save_model_path = get_save_model_path(args)
                            save_model(args, model, save_model_path)
                            print(f"Model saved to {save_model_path}")
        