import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_embednet_embeddings(model, data_loader):
    
    model.eval()
    
    embeddings = []
    labels = []
    with torch.no_grad():
        
        for batch_idx, (data, targets) in enumerate(data_loader):
            X, Y = data.to(device), targets.to(device)
            embedding = model(X)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(targets)
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    return embeddings, labels

def distance_to_means(in_mean, out_mean, test_embeddings):
    
    distances = np.empty((0,2))
    
    for embedding in test_embeddings:
        
        embedding = embedding.reshape(-1, 1)
        in_dist = np.linalg.norm(embedding - in_mean)**2
        out_dist = np.linalg.norm(embedding - out_mean)**2
        
        distances = np.concatenate((distances, np.array([[in_dist, out_dist]])))
    
    return distances

def threshold_by_out_distance(in_classes, distances, predictions, test_targets):
    
    min_distance = np.min(distances[:, 1])
    max_distance = np.max(distances[:, 1])
    
    thresh_step = (max_distance - min_distance) / 1000
    thresholds = np.arange(min_distance, max_distance, thresh_step)
    
    tpr, fpr = [], []
    for thresh in thresholds:
        
        conf_matrix = np.zeros((len(in_classes)+1, len(in_classes)+1), dtype=int)
        
        for i in range(test_targets.shape[0]):
            
            true = int(test_targets[i])
            pred = int(predictions[i])
            
            #in_dist = distances[i][0]
            out_dist = distances[i][1]
            
            if out_dist <= thresh:
                conf_matrix[true, -1] += 1
            else:
                conf_matrix[true, pred] += 1
                
        true_positives = np.sum(conf_matrix[:-1, :-1])
        true_negatives = conf_matrix[-1, -1]
        
        false_positives = np.sum(conf_matrix[-1, :-1])
        false_negatives = np.sum(conf_matrix[:-1, -1])
        
        tpr.append(true_positives / (true_positives + false_negatives))
        fpr.append(false_positives / (false_positives + true_negatives))
        
    if len(tpr) > 1000:
        tpr = tpr[:-1]
    if len(fpr) > 1000:
        fpr = fpr[:-1]
                
    return tpr, fpr, thresholds

def get_conf_matrix(in_classes, distances, predictions, threshold, test_targets):
    
    conf_matrix = np.zeros((len(in_classes)+1, len(in_classes)+1), dtype=int)
    
    for i in range(test_targets.shape[0]):
        
        true = int(test_targets[i])
        pred = int(predictions[i])
        out_dist = distances[i][1]
        
        if out_dist <= threshold:
            conf_matrix[true, -1] += 1
        else:
            conf_matrix[true, pred] += 1
    
    return conf_matrix

def get_in_out_predictions(distances, threshold):
    
    predictions = []
    for i in range(distances.shape[0]):
        
        out_dist = distances[i][1]
        if out_dist <= threshold:
            predictions.append(0)
        else:
            predictions.append(1)
        
    return np.array(predictions)