import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

class TreeStruct():
    TREE_LEAF = -1
    DELETED_LEAF = -5

    def __init__(self, tree):
        self.children_left = tree.children_left
        self.children_right = tree.children_right
        self.value = tree.value
        self.update_leaves()

    def update_leaves(self):
        self.leaves = np.nonzero(self.children_left==TreeStruct.TREE_LEAF)[0]
        if self.leaves.shape[0] == 1:
            # this tree has been pruned to the root, we should delete it from the list of estimators
            return True
        self.leaf_siblings = np.array([self.find_sibling_node(leaf) for leaf in self.leaves])
        self.leaf_pos = np.zeros(self.children_left.shape[0],dtype=np.int32) - 1
        positions = np.arange(self.leaves.shape[0])
        self.leaf_pos[self.leaves] = positions
        return False

    def find_sibling_node(self, node):
        left = np.nonzero((self.children_left==node))[0]
        if left.shape[0] > 0:
            return self.children_right[left[0]]
        right = np.nonzero((self.children_right==node))[0]
        return self.children_left[right[0]]

    def is_leaf(self, node):
        return self.children_left[node] == TreeStruct.TREE_LEAF
    def is_pruned(self, node):
        return self.children_left[node] == TreeStruct.DELETED_LEAF

    def sibling_leaf_positions(self):
        return self.leaf_pos[self.leaf_siblings]

    def merge_leaves(self, leaf):
        if self.is_pruned(leaf):
            return False # already merged
        else:
            assert self.is_leaf(leaf)
            sib = self.leaf_siblings[self.leaf_pos[leaf]]
            if not self.is_leaf(sib):
                return False # can't merge leaf with branch
            self.children_left[[leaf, sib]] = TreeStruct.DELETED_LEAF
            self.children_right[[leaf, sib]] =  TreeStruct.DELETED_LEAF
            parent = np.nonzero(np.logical_or(self.children_left==leaf,self.children_right == leaf))[0][0]
            self.children_left[parent] = TreeStruct.TREE_LEAF
            self.children_right[parent] = TreeStruct.TREE_LEAF
            return True

class RefinedRandomForest():
    def __init__(self, rf, C = 1.0, prune_pct = 0.1, n_prunings = 1, criterion = 'sumnorm'):
        self.rf_ = rf
        self.C = C
        self.prune_pct = prune_pct
        self.n_prunings = n_prunings
        self.criterion = criterion
        self.trees_ = [TreeStruct(tree.tree_) for tree in rf.estimators_]
        self.leaves()

    def leaves(self):
        self.n_leaves_ = [tree.leaves.shape[0] for tree in self.trees_]
        self.M = np.sum(self.n_leaves_)
        self.offsets_ = np.zeros_like(self.n_leaves_)
        self.offsets_[1:] = np.cumsum(self.n_leaves_)[:-1]
        self.ind_trees_ = np.zeros(self.M,dtype=np.int32)
        self.ind_leaves_ = np.zeros(self.M,dtype=np.int32)
        for tree_ind, tree in enumerate(self.trees_):
            start = self.offsets_[tree_ind]
            end = self.offsets_[tree_ind+1] if tree_ind+1<len(self.trees_) else self.M
            self.ind_trees_[start:end] = tree_ind
            self.ind_leaves_[start:end] = tree.leaves

    def get_indicators(self, X):
        leaf = self.rf_.apply(X)
        sample_ind = np.arange(X.shape[0])
        row_ind = []
        col_ind = []
        for tree_ind, tree in enumerate(self.trees_):
            X_leaves = leaf[:,tree_ind]
            row_ind.append(sample_ind)
            col_ind.append(self.offsets_[tree_ind]+tree.leaf_pos[X_leaves])
        row_ind = np.concatenate(row_ind)
        col_ind = np.concatenate(col_ind)
        data = np.ones_like(row_ind)
        indicators = csr_matrix((data, (row_ind, col_ind)), shape=(X.shape[0],self.M))
        return indicators

    def prune_trees(self):
        ind_siblings = np.zeros_like(self.ind_leaves_)
        for tree_ind, tree in enumerate(self.trees_):
            offset = self.offsets_[tree_ind]
            sibl_ind = tree.sibling_leaf_positions()
            sibl_ind[sibl_ind>=0] += offset
            start = self.offsets_[tree_ind]
            end = self.offsets_[tree_ind+1] if tree_ind+1<len(self.trees_) else self.M
            ind_siblings[start:end] = sibl_ind
        coef = self.lr.coef_
        if type(self.rf_) == RandomForestRegressor:
            sibl_coef = coef[ind_siblings]
            sibl_coef[ind_siblings < 0] = np.inf
            if self.criterion == 'sumnorm':
                sum_coef = coef**2 + sibl_coef**2
            elif self.criterion == 'normdiff':
                sum_coef = (coef - sibl_coef)**2
            n_prunings = np.floor(coef.shape[0] * self.prune_pct).astype(int)
        else:
            sibl_coef = coef[:,ind_siblings]
            sibl_coef[:,ind_siblings < 0] = np.inf # so that we don't merge leaf with branch
            if self.criterion == 'sumnorm':
                sum_coef = np.sum(coef**2 + sibl_coef**2,axis=0)
            elif self.criterion == 'normdiff':
                sum_coef = np.sum((coef - sibl_coef)**2,axis=0)
            n_prunings = np.floor(coef.shape[1] * self.prune_pct).astype(int)
        
        ind = np.argsort(sum_coef)
        pruned = 0
        i = 0
        while pruned < n_prunings:
            tree_ind = self.ind_trees_[ind[i]]
            leaf_ind = self.ind_leaves_[ind[i]]
            res = self.trees_[tree_ind].merge_leaves(leaf_ind)
            if res:
                pruned += 1
            i += 1
        to_delete = []
        for tree_ind, tree in enumerate(self.trees_):
            if tree.update_leaves():
                to_delete.append(tree)
        for tree in to_delete:
            treeind = self.trees_.index(tree)
            del self.rf_.estimators_[treeind]
            self.trees_.remove(tree)
        self.leaves()

    def fit(self, X, y):
        n_pruned = 0
        while n_pruned <= self.n_prunings:
            indicators = self.get_indicators(X)
            #print('Model size: {} leaves'.format(indicators.shape[1]))
            #self.svr = SVR(C=self.C,fit_intercept=False,epsilon=0.)
            if type(self.rf_) == RandomForestRegressor:
                self.lr = LinearRegression(fit_intercept=False,
                                n_jobs=-1)
                print('###########  RandomForestRegressor ############')
            else:
                self.lr = LogisticRegression(C=self.C,
                                fit_intercept=False,
                                solver='lbfgs',
                                max_iter=100,
                                multi_class='multinomial', n_jobs=-1)
            # self.lr = LogisticRegression(C=self.C,
            #                 fit_intercept=False,
            #                 solver='lbfgs',
            #                 max_iter=100,
            #                 multi_class='multinomial', n_jobs=-1)
            self.lr.fit(indicators,y)
            if n_pruned < self.n_prunings:
                self.prune_trees()
            n_pruned += 1
        for tree_ind, tree in enumerate(self.trees_):
            offset = self.offsets_[tree_ind]
            if type(self.rf_) == RandomForestRegressor:
                print('tree.value: ', tree.value.shape)
                print('tree.leaves.shape: ', tree.leaves.shape)
                print('self.lr.coef_[offset:offset + tree.leaves.shape[0]]: ', self.lr.coef_[offset:offset + tree.leaves.shape[0]].shape)
                print('offset: ', offset)
                a = self.lr.coef_[offset:offset + tree.leaves.shape[0]]
                print('a:', a.shape)
                b = tree.value[tree.leaves,0].flatten()
                print('b:', b.shape)
                b = a
                tree.value[tree.leaves,0] = self.lr.coef_[offset:offset + tree.leaves.shape[0]].reshape((-1,1))
                print('tree.value[tree.leaves]', tree.value[tree.leaves].shape)
            else:
                tree.value[tree.leaves,0,:] = self.lr.coef_[:,offset:offset + tree.leaves.shape[0]].T

    def predict_proba(self, X):
        if type(self.rf_) == RandomForestRegressor:
            return self.lr.predict(self.get_indicators(X))
        else:
            return self.lr.predict_proba(self.get_indicators(X))