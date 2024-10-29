import os
import argparse
import logging
import csv

import numpy as np
from sklearn.metrics import auc

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from pytorch_metric_learning import losses, miners

from data_generator import *
from utils import *
from models.classifier_resnet import resnet18 as classifier_resnet
from models.embed_resnet import resnet18 as embed_resnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Dataset to be used in trials. One of [cifar10, cifar+10, cifar+50, tinyimg]')
parser.add_argument('--known_data_dir', type=str,
                    help='Known dataset directory.')
parser.add_argument('--unknown_data_dir', type=str,
                    help='Unknown dataset directory.')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Margin to use in triplet loss.')
parser.add_argument('--ku_percentage', type=float, default=0.5,
                    help='Percetange of dataset classes to be used as known unknowns.')
parser.add_argument('--in_channels', type=int, default=3,
                    help='Number of in channels for classifier and embedding network.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size.')
parser.add_argument('--classifier_epochs', type=int, default=15,
                    help='Number of epochs used to train classifier.')
parser.add_argument('--classifier_lr', type=float, default=0.001,
                    help='Learning rate for classifier.')
parser.add_argument('--embednet_epochs', type=int, default=60,
                    help='Number of epochs used to train embedding network.')
parser.add_argument('--embednet_lr', type=float, default=0.001,
                    help='Learning rate for embedding network.')
parser.add_argument('--num_trials', type=int, default=3,
                    help='Number of trials to perform.')
args = parser.parse_args()

# Initialize logging
results_dir = './results/' + args.dataset + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
    with open(os.path.join(results_dir+'results.csv'), 'w') as f:
        output = csv.writer(f)
        output.writerow(['Dataset', 'Beta', 'Known Unknowns Percentage', 'AUROC', 'FPRs', 'TPRs', 'CCR at 95% TPR', 'CCRs', 'TPRs'])

def train_classifier(model, device, train_loader, lr, criterion):
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9, weight_decay = 0.005)
    
    loss = 0
    correct = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        
        X = data.to(device)
        Y = labels.to(device)
        
        optimizer.zero_grad()
        
        Y_hat = model(X)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()

        loss += loss.item()
        correct += (torch.argmax(Y_hat, dim=1) == Y).sum()
        
    return loss, correct

def test_classifier(model, device, test_loader, criterion):
    
    model.eval()
    
    loss = 0
    correct = 0
    
    with torch.no_grad():
        
        for batch_idx, (data, labels) in enumerate(test_loader):

            X = data.to(device)
            Y = labels.to(device)

            Y_hat = model(X)
            loss = criterion(Y_hat, Y)

            loss += loss.item()
            correct += (torch.argmax(Y_hat, dim=1) == Y).sum()
        
    return loss, correct

def get_classifier_predictions(classifier, test_loader):
    
    classifier.eval()
    
    predictions=[]
    with torch.no_grad():
        
        for batch_idx, (data, _) in enumerate(test_loader):
            
            X = data.to(device)
            Y_hat = classifier(X)
            predictions.append(Y_hat.detach().cpu().numpy())
    
    predictions = np.argmax(np.concatenate(predictions), axis=1)
    
    return predictions

def train_embednet(model, device, data_loader, lr, margin=0.2, miner='semihard'):
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    if miner == 'semihard':
        miner_fn = miners.TripletMarginMiner(margin=margin, type_of_triplets='semihard')
    elif miner == 'hard':
        miner_fn = miners.TripletMarginMiner(margin=margin, type_of_triplets='hard')
        
    loss_fn = losses.TripletMarginLoss(margin=margin)
    
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        
        # Input embedddings from ResNet
        data = data.to(device)
        labels = labels.to(device)
        
        # Output embeddings from EmbedNet
        embeddings = model(data)
        
        # Triplet Loss
        triplets = miner_fn(embeddings, labels)
        loss = loss_fn(embeddings, labels, triplets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss

def test_embednet(embednet, embednet_train_loader, test_loader, in_classes, test_targets, predictions):
    
    # Get Output Embeddings of Trained Data
    train_embeddings, embedding_labels = get_embednet_embeddings(embednet, embednet_train_loader)

    # Sort embeddings by in and out labels
    in_labels = np.where(embedding_labels==1)[0]
    out_labels = np.where(embedding_labels==0)[0]
    
    # Find mean of in and out embeddings
    in_mean = np.mean(train_embeddings[in_labels], axis=0).reshape(-1, 1)
    out_mean = np.mean(train_embeddings[out_labels], axis=0).reshape(-1, 1)
    
    # Get Output Embeddings of Test Data
    test_embeddings, _ = get_embednet_embeddings(embednet, test_loader)
    
    # Get embedding distances to each mean
    distances = distance_to_means(in_mean, out_mean, test_embeddings)

    # Threshold by distance to out mean
    tpr, fpr, thresholds = threshold_by_out_distance(in_classes, distances, predictions, test_targets)
    
    # AUROC calculation
    auroc = auc(fpr, tpr)
    
    return auroc

def main():
    
    aurocs = []
    ccrs_95 = []
    
    tprs = []
    ccrs = []
    
    for trial in range(args.num_trials):
        
        print('\nTRIAL {}\n'.format(trial+1))
        
        if args.dataset == 'mnist':
            
            # Set known and unknown classes
            all_classes = np.arange(0,10)
            kk_classes = list(set(all_classes) - set(np.random.choice(all_classes, size=4, replace=False)))
            unknown_class_id = len(kk_classes)
            remaining_classes = list(set(all_classes) - set(kk_classes))
            num_ku_classes = args.ku_percentage * len(remaining_classes)
            
            if num_ku_classes < 1:
                num_ku_classes = 1
            else:
                num_ku_classes = int(np.floor(num_ku_classes))
            
            ku_classes = np.random.choice(remaining_classes, size=num_ku_classes, replace=False)
            uu_classes = np.array(list(set(remaining_classes) - set(ku_classes)))
            
            # Generate data
            kk_train_data, kk_train_targets = generate_kk_mnist_train_data(args.known_data_dir, kk_classes)
            ku_train_data, ku_train_targets = generate_ku_mnist_train_data(args.unknown_data_dir, ku_classes, unknown_class_id)

            kk_test_data, kk_test_targets = generate_kk_mnist_test_data(args.known_data_dir, kk_classes)
            ku_test_data, ku_test_targets = generate_unknown_mnist_test_data(args.unknown_data_dir, ku_classes, unknown_class_id)
            uu_test_data, uu_test_targets = generate_unknown_mnist_test_data(args.unknown_data_dir, uu_classes, unknown_class_id)
            
            kk_train_data = kk_train_data.unsqueeze(-1)
            ku_train_data = ku_train_data.unsqueeze(-1)
            kk_test_data = kk_test_data.unsqueeze(-1)
            ku_test_data = ku_test_data.unsqueeze(-1)
            uu_test_data = uu_test_data.unsqueeze(-1)
            
            kk_train_target = np.array(kk_train_targets)
            
        if args.dataset == 'svhn':
            
            # Set known and unknown classes
            all_classes = np.arange(0,10)
            kk_classes = list(set(all_classes) - set(np.random.choice(all_classes, size=4, replace=False)))
            unknown_class_id = len(kk_classes)
            remaining_classes = list(set(all_classes) - set(kk_classes))
            num_ku_classes = int(args.ku_percentage * len(remaining_classes))
            
            if num_ku_classes < 1:
                num_ku_classes = 1
            else:
                num_ku_classes = int(np.floor(num_ku_classes))
            
            ku_classes = np.random.choice(remaining_classes, size=num_ku_classes, replace=False)
            uu_classes = np.array(list(set(remaining_classes) - set(ku_classes)))
            
            # Generate data
            kk_train_data, kk_train_targets = generate_kk_svhn_train_data(args.known_data_dir, kk_classes)
            ku_train_data, ku_train_targets = generate_ku_svhn_train_data(args.unknown_data_dir, ku_classes, unknown_class_id)

            kk_test_data, kk_test_targets = generate_kk_svhn_test_data(args.known_data_dir, kk_classes)
            ku_test_data, ku_test_targets = generate_unknown_svhn_test_data(args.unknown_data_dir, ku_classes, unknown_class_id)
            uu_test_data, uu_test_targets = generate_unknown_svhn_test_data(args.unknown_data_dir, uu_classes, unknown_class_id)
            
            kk_train_data = kk_train_data.transpose(0,2,3,1)
            ku_train_data = ku_train_data.transpose(0,2,3,1)
            kk_test_data = kk_test_data.transpose(0,2,3,1)
            ku_test_data = ku_test_data.transpose(0,2,3,1)
            uu_test_data = uu_test_data.transpose(0,2,3,1)
        
        if args.dataset == 'cifar10':

            # Set known and unknown classes
            kk_classes = [2,3,4,5,6,7]
            unknown_class_id = len(kk_classes)
            remaining_classes = np.array(list(set(np.arange(0,10)) - set(kk_classes)))
            num_ku_classes = int(args.ku_percentage * len(remaining_classes))
            
            if num_ku_classes < 1:
                num_ku_classes = 1
            else:
                num_ku_classes = int(np.floor(num_ku_classes))
            
            ku_classes = np.random.choice(remaining_classes, size=num_ku_classes, replace=False)
            uu_classes = np.array(list(set(remaining_classes) - set(ku_classes)))

            # Generate data
            kk_train_data, kk_train_targets = generate_kk_cifar10_train_data(args.known_data_dir, kk_classes)
            ku_train_data, ku_train_targets = generate_ku_cifar10_train_data(args.unknown_data_dir, ku_classes, unknown_class_id)

            kk_test_data, kk_test_targets = generate_kk_cifar10_test_data(args.known_data_dir, kk_classes)
            ku_test_data, ku_test_targets = generate_unknown_cifar10_test_data(args.unknown_data_dir, ku_classes, unknown_class_id)
            uu_test_data, uu_test_targets = generate_unknown_cifar10_test_data(args.unknown_data_dir, uu_classes, unknown_class_id)

        elif args.dataset == 'cifar+10':

            # Set known and unknown classes
            kk_classes = [0,1,8,9]
            unknown_class_id = len(kk_classes)

            vehicles = [8,13,48,58,90,41,69,81,85,89]
            non_vehicles = np.array(list(set(np.arange(0,100)) - set(vehicles)))
            
            num_ku_classes = int(args.ku_percentage * 10)

            rand_classes = np.random.choice(non_vehicles, size=10, replace=False)
            ku_classes = np.random.choice(rand_classes, size=num_ku_classes, replace=False)
            uu_classes = np.array(list(set(rand_classes) - set(ku_classes)))

            # Generate data
            kk_train_data, kk_train_targets = generate_kk_cifar10_train_data(args.known_data_dir, kk_classes)
            ku_train_data, ku_train_targets = generate_ku_cifar100_train_data(args.unknown_data_dir, ku_classes, unknown_class_id)

            kk_test_data, kk_test_targets = generate_kk_cifar10_test_data(args.known_data_dir, kk_classes)
            ku_test_data, ku_test_targets = generate_unknown_cifar100_test_data(args.unknown_data_dir, ku_classes, unknown_class_id)
            uu_test_data, uu_test_targets = generate_unknown_cifar100_test_data(args.unknown_data_dir, uu_classes, unknown_class_id)

        elif args.dataset == 'cifar+50':

            # Set known and unknown classes
            kk_classes = [0,1,8,9]
            unknown_class_id = len(kk_classes)

            vehicles = [8,13,48,58,90,41,69,81,85,89]
            non_vehicles = np.array(list(set(np.arange(0,100)) - set(vehicles)))
            
            num_ku_classes = int(args.ku_percentage * 50)

            rand_classes = np.random.choice(non_vehicles, size=50, replace=False)
            ku_classes = np.random.choice(rand_classes, size=num_ku_classes, replace=False)
            uu_classes = np.array(list(set(rand_classes) - set(ku_classes)))

            # Generate data
            kk_train_data, kk_train_targets = generate_kk_cifar10_train_data(args.known_data_dir, kk_classes)
            ku_train_data, ku_train_targets = generate_ku_cifar100_train_data(args.unknown_data_dir, ku_classes, unknown_class_id)

            kk_test_data, kk_test_targets = generate_kk_cifar10_test_data(args.known_data_dir, kk_classes)
            ku_test_data, ku_test_targets = generate_unknown_cifar100_test_data(args.unknown_data_dir, ku_classes, unknown_class_id)
            uu_test_data, uu_test_targets = generate_unknown_cifar100_test_data(args.unknown_data_dir, uu_classes, unknown_class_id)

        elif args.dataset == 'tinyimg':
            class_ids = np.arange(0,200)
            kk_classes = np.random.choice(class_ids, size=20, replace=False)
            unknown_class_id = len(kk_classes)

            remaining_classes = np.array(list(set(class_ids) - set(kk_classes)))
            num_ku_classes = int(args.ku_percentage * len(remaining_classes))
            
            ku_classes = np.random.choice(remaining_classes, size=num_ku_classes, replace=False)
            uu_classes = np.array(list(set(remaining_classes) - set(ku_classes)))

            # Generate data
            kk_train_data, kk_train_targets = generate_kk_tinyimg_train_data(args.known_data_dir, kk_classes)
            ku_train_data, ku_train_targets = generate_ku_tinyimg_train_data(args.unknown_data_dir, ku_classes, unknown_class_id)

            kk_test_data, kk_test_targets = generate_kk_tinyimg_test_data(args.known_data_dir, kk_classes)
            ku_test_data, ku_test_targets = generate_unknown_tinyimg_test_data(args.unknown_data_dir, ku_classes, unknown_class_id)
            uu_test_data, uu_test_targets = generate_unknown_tinyimg_test_data(args.unknown_data_dir, uu_classes, unknown_class_id)
            
        # Create datasets
        embednet_targets = np.concatenate((np.ones(kk_train_targets.shape[0]), np.zeros(ku_train_targets.shape[0])))
        
        complete_test_data = np.concatenate((kk_test_data, ku_test_data, uu_test_data))
        complete_test_targets = np.concatenate((kk_test_targets, ku_test_targets, uu_test_targets))

        d_ku_test_data = np.concatenate((kk_test_data, ku_test_data))
        d_ku_test_targets = np.concatenate((kk_test_targets, ku_test_targets))

        d_uu_test_data = np.concatenate((kk_test_data, uu_test_data))
        d_uu_test_targets = np.concatenate((kk_test_targets, uu_test_targets))
        
        # DataLoaders
        classifier_train_loader = DataLoader(TensorDataset(torch.FloatTensor(np.array(kk_train_data)).permute(0,3,1,2), 
                                                           torch.LongTensor(kk_train_targets)),
                                                           batch_size=args.batch_size, shuffle=True)
        embednet_train_loader = DataLoader(TensorDataset(torch.cat((torch.FloatTensor(np.array(kk_train_data)), torch.FloatTensor(np.array(ku_train_data)))).permute(0,3,1,2),
                                                         torch.LongTensor(embednet_targets)),
                                                         batch_size=512, shuffle=True)
        classifier_test_loader = DataLoader(TensorDataset(torch.FloatTensor(np.array(kk_test_data)).permute(0, 3, 1, 2),
                                                  torch.LongTensor(kk_test_targets)), 
                                                  batch_size=args.batch_size, shuffle=False)
        #test_loader = DataLoader(TensorDataset(torch.cat((torch.FloatTensor(IN_test_data), torch.FloatTensor(OUT_test_data), torch.FloatTensor(UNKNOWN_test_data))).permute(0,3,1,2),
        #                                       torch.cat((torch.LongTensor(IN_test_targets), torch.LongTensor(OUT_test_targets), torch.LongTensor(UNKNOWN_test_targets)))),
        #                                       batch_size=128, shuffle=False)
        #test_loader = DataLoader(TensorDataset(torch.FloatTensor(complete_test_data).permute(0,3,1,2),
        #                                       torch.LongTensor(complete_test_targets)),
        #                         batch_size=128, shuffle=False)

        d_ku_test_loader = DataLoader(TensorDataset(torch.FloatTensor(np.array(d_ku_test_data)).permute(0,3,1,2),
                                                    torch.LongTensor(d_ku_test_targets)),
                                      batch_size=args.batch_size, shuffle=False)
        d_uu_test_loader = DataLoader(TensorDataset(torch.FloatTensor(np.array(d_uu_test_data)).permute(0,3,1,2),
                                                    torch.LongTensor(d_uu_test_targets)),
                                      batch_size=args.batch_size, shuffle=False)
        
        # Classifier
        classifier = classifier_resnet(args.in_channels, num_classes=len(kk_classes))
        classifier.to(device)

        # Classifier Training Params
        classifier_epochs = args.classifier_epochs
        classifier_lr = args.classifier_lr
        classifier_criterion = nn.NLLLoss()

        # Traing Loop
        print('TRAINING CLASSIFIER')
        best_accuracy = 0
        for epoch in range(classifier_epochs):

            train_loss, train_correct = train_classifier(classifier, device, classifier_train_loader, classifier_lr, classifier_criterion)
            test_loss, test_correct = test_classifier(classifier, device, classifier_test_loader, classifier_criterion)

            print('Epoch: {}/{} \n\t Train Loss: {:0.4f} | Train Accuracy: {:0.4f}'.format(epoch+1, classifier_epochs, train_loss, (train_correct/len(kk_train_data.data)) * 100))
            print('\t Test Loss: {:0.4f} | Test Accuracy {:0.4f}'.format(test_loss, (test_correct/len(kk_test_data.data)) * 100))

            if (test_correct/len(kk_test_data.data)) * 100 > best_accuracy:
                best_accuracy = (test_correct/len(kk_test_data.data)) * 100
                torch.save(classifier.state_dict(), './' + args.dataset + '_classifier.pth')
                
        # Load best classifier
        classifier.load_state_dict(torch.load('./' + args.dataset + '_classifier.pth'))

        # Get classifier predictions of all test data (KKC, KUC, and UUC)
        predictions = get_classifier_predictions(classifier, d_uu_test_loader)

        # EmbedNet
        embednet = embed_resnet(args.in_channels)
        embednet.to(device)

        # EmbedNet Training Params
        embednet_epochs = args.embednet_epochs
        embednet_lr = args.embednet_lr

        # Training Loop
        print('\nTRAINING EMBEDNET\n')
        best_auroc = 0
        for epoch in range(embednet_epochs):

            if epoch+1 == 50:
                embednet_lr /= 10

            # if epoch+1 <= 25:
            if epoch+1 <= args.embednet_epochs:
                train_loss = train_embednet(embednet, device, embednet_train_loader, embednet_lr, margin=args.beta, miner='semihard')
                test_auroc = test_embednet(embednet, embednet_train_loader, d_uu_test_loader, kk_classes, d_uu_test_targets, predictions)
                print('Epoch: {}/{} \n\t Train Loss: {:0.4f}'.format(epoch+1, embednet_epochs, train_loss))
                print('\t Test AUROC: {:0.4f}'.format(test_auroc))
            else:
                train_loss = train_embednet(embednet, device, embednet_train_loader, embednet_lr, margin=args.beta, miner='hard')
                test_auroc = test_embednet(embednet, embednet_train_loader, d_uu_test_loader, kk_classes, d_uu_test_targets, predictions)
                print('Epoch: {}/{} \n\t Train Loss: {:0.4f}'.format(epoch+1, embednet_epochs, train_loss))
                print('\t Test AUROC: {:0.4f}'.format(test_auroc))

            if test_auroc > best_auroc:
                best_auroc = test_auroc
                torch.save(embednet.state_dict(), './' + args.dataset + '_embednet.pth')
                
        # Load best embednet
        embednet.load_state_dict(torch.load('./' + args.dataset + '_embednet.pth'))
        
        # Get Output Embeddings of Trained Data
        train_embeddings, embedding_labels = get_embednet_embeddings(embednet, embednet_train_loader)

        # Sort embeddings by kk and ku labels
        kk_labels = np.where(embedding_labels==1)[0]
        ku_labels = np.where(embedding_labels==0)[0]

        # Find mean of kk and ku embeddings
        kk_mean = np.mean(train_embeddings[kk_labels], axis=0).reshape(-1, 1)
        ku_mean = np.mean(train_embeddings[ku_labels], axis=0).reshape(-1, 1)
        
        # Get Output Embeddings of Test Data
        test_embeddings, _ = get_embednet_embeddings(embednet, d_uu_test_loader)

        # Get embedding distances to each mean
        distances = distance_to_means(kk_mean, ku_mean, test_embeddings)

        # Threshold by distance to out mean
        tpr, fpr, thresholds = threshold_by_out_distance(kk_classes, distances, predictions, d_uu_test_targets)
        
        # AUROC calculation
        auroc = auc(fpr, tpr)
        
        # Find threshold corresponding to 95% TPR
        tpr_idx = np.where(np.array(tpr) >= 0.95)[0][-1]
        thresh = thresholds[tpr_idx]
        
        # Get confusion matrix of corresponding threshold
        confusion_matrix = get_conf_matrix(kk_classes, distances, predictions, thresh, d_uu_test_targets)

        # Correct classification rate (ccr) at 95% tpr
        ccr_95 = np.diag(confusion_matrix[:-1, :-1]).sum() / confusion_matrix[:-1, :-1].sum()
        
        # Append to results
        aurocs.append(auroc)
        ccrs_95.append(ccr_95)
        
        # Calculate CCR at varying tpr
        # ccr_values = []
        # tpr_values = []
        # for tpr_value in np.arange(0, 1.01, 0.01):
        #     tpr_idx = np.where(np.array(tpr) >= tpr_value)[0][-1]
        #     thresh = thresholds[tpr_idx]
        #     cm = get_conf_matrix(kk_classes, distances, predictions, thresh, d_uu_test_targets)
        #     ccr = np.diag(cm[:-1, :-1]).sum() / cm[:-1, :-1].sum()
        #     tpr_values.append(tpr_value)
        #     ccr_values.append(ccr)
        # ccrs.append(ccr_values)
        # tprs.append(tpr_values)
        
        # Logging
        with open(os.path.join(results_dir+'results.csv'), 'a', newline='') as f:
            output = csv.writer(f)
            output.writerow([args.dataset, args.beta, args.ku_percentage, auroc, 0, 0, ccr_95, 0, 0])
        
    # Reporting
    print('\nRESULTS')
    print('Unknown Unknowns AUROC:', np.mean(aurocs), '+/-', np.std(aurocs))
    print('Unknown Unknowns CCR at 95% TPR:', np.mean(ccrs_95), '+/-', np.std(ccrs_95))
    
if __name__ == '__main__':
    main()