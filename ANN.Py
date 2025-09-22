import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
data = pd.read_csv('data.csv', delimiter='|')

# Drop irrelevant columns
columns_to_drop = [
    'ts', 'uid', 'id.orig_h', 'id.resp_h', 'tunnel_parents',
    'detailed-label', 'proto', 'service', 'local_orig', 'local_resp', 'missed_bytes'
]
data = data.drop(columns=columns_to_drop)

# Replace '-' with 0 in numerical columns
numerical_cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
for col in numerical_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# One-hot encode categorical features
categorical_cols = ['conn_state', 'history']
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# Combine numerical and encoded categorical data
numerical_data = data[['id.orig_p', 'id.resp_p'] + numerical_cols]
processed_data = pd.concat([numerical_data, encoded_df], axis=1)

# Convert labels to binary (1 for Malicious, 0 for Benign)
processed_data['label'] = data['label'].apply(lambda x: 1 if x == 'Malicious' else 0)

# Separate features (X) and labels (y)
X = processed_data.drop('label', axis=1).values
y = processed_data['label'].values.reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def compute_loss(self, y, epsilon=1e-8):
        m = y.shape[0]
        loss = -np.mean(y * np.log(self.A2 + epsilon) + (1 - y) * np.log(1 - self.A2 + epsilon))
        return loss
    
    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        # Hidden layer gradients
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            A2 = self.forward(X)
            loss = self.compute_loss(y)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        A2 = self.forward(X)
        return (A2 >= 0.5).astype(int)
    
# Initialize network
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the model
nn.train(X_train, y_train, epochs=1000, learning_rate=0.001)

# Evaluate on test set
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')