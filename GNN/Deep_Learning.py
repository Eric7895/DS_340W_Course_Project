# Try to import libraries
import numpy as np
import pandas as pd
import time
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv 
import torch.nn.functional as F
from sklearn.metrics import f1_score

class GCN(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        torch.manual_seed(42)
        # Define GCN layers
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, len(torch.unique(data.y)))

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = x.relu()

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = x.relu()

        # Apply dropout to reduce overfitting
        x = F.dropout(x, p=0.5, training=self.training)

        # Third GCN layer
        x = self.conv3(x, edge_index)

        return x

def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask, data, model):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Select only masked nodes for evaluation
        true_labels = data.y[mask].cpu().numpy()
        pred_labels = pred[mask].cpu().numpy()
        
        # Calculate accuracy
        correct = pred[mask].eq(data.y[mask])
        accuracy = int(correct.sum()) / int(mask.sum())
        
        # Calculate precision (macro-averaged)
        f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        
    return accuracy, f1

def main():
    # Read Data

    user = pd.read_csv('adjusted_users.csv', low_memory=False)
    edge = pd.read_csv('edge.csv')
    
    # Convert IDs in the user dataset to numeric codes
    user_id_mapping = pd.Categorical(user['id'])
    user['id'] = user_id_mapping.codes

    # Convert source and target IDs in the edge dataset to numeric codes
    edge = edge[
            edge['source_id'].isin(user_id_mapping.categories) &
            edge['target_id'].isin(user_id_mapping.categories)
            ]

    edge['source_id'] = pd.Categorical(
        edge['source_id'], categories=user_id_mapping.categories
    ).codes

    edge['target_id'] = pd.Categorical(
        edge['target_id'], categories=user_id_mapping.categories
    ).codes

    # Create edge tensor for the graph
    edge_index = torch.tensor(edge[['source_id', 'target_id']].values.T, dtype=torch.long)

    # Prepare node features
    numeric_columns = [
    "follower_count", "following_count", "tweet_count", "listed_count",
    "year_created", "description_True", "description_False",
    "entities_True", "entities_False", "location_True", "location_False",
    "protected_True", "protected_False", "url_True", "url_False",
    "verified_True", "verified_False", "withheld_True", "withheld_False",
    "bot_True", "bot_False"
    ]

    filtered_features = user[numeric_columns]

    # Ensure all columns are numeric and handle missing values
    filtered_features = filtered_features.infer_objects()  # Convert object columns to numeric
    filtered_features = filtered_features.fillna(0)       # Replace NaN with 0
    filtered_features = filtered_features.apply(pd.to_numeric, errors='coerce')  # Ensure numeric

    # Convert the features DataFrame to a PyTorch tensor
    features = torch.tensor(filtered_features.values, dtype=torch.float32)

    # Prepare labels
    # Map categorical labels (e.g., "human", "bot") to numeric values
    label_mapping = {"human": 0, "bot": 1}
    labels = torch.tensor(user['label'].map(label_mapping).values, dtype=torch.long)

    # Create the PyG data object
    # `x`: node features, `edge_index`: graph edges, `y`: node labels
    data = Data(x=features, edge_index=edge_index, y=labels)

    # Normalize features
    # Normalize features to ensure they are scaled appropriately for training
    data = NormalizeFeatures()(data)

    print('--------------------------------------------------------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    print(f"Total nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
    
    # Train/validation/test split
    perm = torch.randperm(data.num_nodes)
    train_idx = perm[:int(data.num_nodes * 0.8)]
    val_idx = perm[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)]
    test_idx = perm[int(data.num_nodes * 0.8):]

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True

    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[val_idx] = True

    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = True

    print(f"Train nodes: {data.train_mask.sum().item()}, Validation nodes: {data.val_mask.sum().item()}, Test nodes: {data.test_mask.sum().item()}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data, hidden_channels=16).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_precision = 0
    patience = 10
    patience_counter = 0

    # Training loop
    start_time = time.time()
    for epoch in range(1, 20001):  # Adjust the range as needed
        loss = train(data, model, optimizer, criterion)
        val_acc, val_f1 = evaluate(data.val_mask, data, model)
        test_acc, test_f1 = evaluate(data.test_mask, data, model)

        # Enable early stopping
        #if val_precision > best_val_precision:
        #    best_val_precision = val_precision
        #    patience_counter = 0
        #    torch.save(model.state_dict(), 'best_gcn_model.pth') # Save model for later use
        #else:
        #    patience_counter += 1
        
        #if patience_counter >= patience:
        #    print("Early stopping triggered.")
        #    elapsed = time.time() - start_time
        #    print(
        #        f"[{elapsed:.2f}s] Epoch: {epoch:03d}, "
        #        f"Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, "
        #        f"Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}"
        #    )

        #    break

        
        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{elapsed:.2f}s] Epoch: {epoch:03d}, "
                f"Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
                f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}"
            )

if __name__ == '__main__':
    main()
