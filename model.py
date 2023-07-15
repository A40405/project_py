# DecisionTree
import graphviz
import numpy as np
from io import StringIO 

class DecisionTree:
    def __init__(self, max_depth = 4, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}

    def fit(self, X, y):
        if len(X.shape) == 1: X = X.reshape(-1,1)
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)
    
    def predict(self, X):
        if len(X.shape) == 1: X = X.reshape(-1,1)
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature_idx']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])
        
    def _grow_tree(self, X, y, depth=0):
        def _leaf_value(y):
            counts = np.bincount(y)
            return np.argmax(counts)
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            leaf_value = _leaf_value(y)
            return {'leaf': True, 'value': leaf_value, 'id':depth}

        # Tìm thuộc tính tốt nhất và điểm chia cây con theo thuộc tính 
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Tìm cây con
        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = X[:, best_feat] >= best_thresh
        left_tree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right_tree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return {'leaf': False,
                'feature_idx': best_feat,
                'threshold': best_thresh,
                'left': left_tree,
                'right': right_tree,
                'id':depth}

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, i], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_col, split_thresh):
        # Tính entropy của cha
        parent_entropy = self._entropy(y)

        # Chia cột X_col thành 2 khoảng theo split_thresh
        left_idxs = X_col < split_thresh
        right_idxs = X_col >= split_thresh

        # Tính entropy con 
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        if n_l == 0 or n_r == 0:
            return 0
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        # Tính information gain
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        ig = parent_entropy - child_entropy
        return ig
    
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    def _to_graphviz(self):
        dot_data = StringIO()
        dot_data.write('digraph DecisionTree {\n')
        self._to_graphviz_rec(self.tree, dot_data)
        dot_data.write('}\n')
        return graphviz.Source(dot_data.getvalue())

    def _to_graphviz_rec(self, node, dot_data):
        if node['leaf']:
            value = f"Class : nhãn {node['value']}"
        else:
            value = f"feature={node['feature_idx']}, threshold={node['threshold']:.2f}"
        dot_data.write(f"{node['id']} [label=\"{value}\"];\n")
        if not node['leaf']:
            self._to_graphviz_rec(node['left'], dot_data)
            self._to_graphviz_rec(node['right'], dot_data)
            dot_data.write(f"{node['id']} -> {node['left']['id']} [label=\"True               \"];\n")
            dot_data.write(f"{node['id']} -> {node['right']['id']} [label=\" False\"];\n")
