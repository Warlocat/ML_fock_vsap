from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, RobustScaler
# from sklearn import cluster, mixture
import numpy as np

# https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
def batched_std_update(std1, std2, mean1, mean2, n1, n2):  
    var_tot = (n1 / (n1 + n2))*std1**2 + (n2 / (n1 + n2))*std2**2 + (n1*n2 / ((n1 + n2)**2))*(mean1 - mean2)**2
    return var_tot**0.5

def batched_mean_update(mean1, mean2, n1, n2):
    return (n1 / (n1 + n2))*mean1 + (n2 / (n1 + n2))*mean2

def batched_update_standardscaler(scaler1, scaler2):
    """Combine two StandardScalars without needing to explicitly load their data into memory

    Args:
        scaler1 (StandardScalar): fitted scaler object with mean, std
        scaler2 (StandardScalar):  fitted scaler object with mean, std

    Returns:
        StandardScalar: updated scalar with new mean and std and new total data size.
    """    
    assert scaler1.n_features_in_ == scaler2.n_features_in_
    assert scaler1.with_mean == scaler2.with_mean
    assert scaler1.with_std == scaler2.with_std
    new_scaler = StandardScaler(with_mean = scaler1.with_mean, with_std = scaler1.with_std)
    new_scaler.n_features_in_ = scaler1.n_features_in_
    new_scaler.scale_ = batched_std_update(scaler1.scale_, scaler2.scale_, scaler1.mean_, scaler2.mean_, scaler1.n_samples_seen_, scaler2.n_samples_seen_)
    new_scaler.var_ = np.copy(new_scaler.scale_)**2
    new_scaler.mean_ = batched_mean_update(scaler1.mean_, scaler2.mean_, scaler1.n_samples_seen_, scaler2.n_samples_seen_)
    new_scaler.n_samples_seen_ = scaler1.n_samples_seen_ + scaler2.n_samples_seen_
    return new_scaler


class DataScaler:

    def __init__(self, k_values_ii = [1, 0.5, 0.1, 5e-2], 
                    k_values_ij = [1, 0.5, 0.1, 5e-2], 
                    applies_to_ii = [], applies_to_ij = [], mean_shift = False):
        self.mean_shift = mean_shift
        self.transformer_xii = StandardScaler(with_mean = self.mean_shift)
        self.transformer_xij = StandardScaler(with_mean = self.mean_shift)
        self.k_values_ii = k_values_ii
        self.k_values_ij = k_values_ij
        self.applies_to_ii = applies_to_ii
        self.applies_to_ij = applies_to_ij
    
    def fit(self, xii, xij):
        self.transformer_xii.fit(xii)
        self.transformer_xij.fit(xij)
        return self
    
    def transform(self, xii, xij, **kwargs):

        new_xii, new_xij = [], []
        new_xii.append(self.transformer_xii.transform(xii))
        new_xij.append(self.transformer_xij.transform(xij))

        for i in self.applies_to_ii:
            for k in self.k_values_ii:
                new_xii.append(np.sin(new_xii[0][:,[i]]/k))
                new_xii.append(np.cos(new_xii[0][:,[i]]/k))
        for i in self.applies_to_ij:
            for k in self.k_values_ij:
                new_xij.append(np.sin(new_xij[0][:,[i]]/k))
                new_xij.append(np.cos(new_xij[0][:,[i]]/k))

        return np.concatenate(new_xii, axis=1), np.concatenate(new_xij, axis=1)

    def batch_update(self, xii, xij):
        new_transformer_xii = StandardScaler(with_mean = self.mean_shift)
        new_transformer_xii.fit(xii)
        new_transformer_xij = StandardScaler(with_mean = self.mean_shift)
        new_transformer_xij.fit(xij)
        self.transformer_xii = batched_update_standardscaler(self.transformer_xii, new_transformer_xii)
        self.transformer_xij = batched_update_standardscaler(self.transformer_xij, new_transformer_xij)
        return self
    
    def new(self):
        """
        Create a brand-new DataScaler with the same init arguments,
        without copying any fitted state.
        """
        return DataScaler(k_values_ii = self.k_values_ii, k_values_ij = self.k_values_ij, 
                            applies_to_ii = self.applies_to_ii, applies_to_ij = self.applies_to_ij, mean_shift = self.mean_shift)
    
    def unify_shared_feature(self, ii_scaler_position, ij_scaler_position):
        """
        When ii and ij share the same feature, this safely unifies their statistics, but can only be done once
        """
        print('mlvk.model.preprocessing.DataScaler WARNING: modifying the statistics of transformer_ii and transformer_ij can only be done once!')
        self.transformer_xii.scale_[ii_scaler_position] = batched_std_update(self.transformer_xii.scale_[ii_scaler_position], self.transformer_xij.scale_[ij_scaler_position], self.transformer_xii.mean_[ii_scaler_position], self.transformer_xij.mean_[ij_scaler_position], self.transformer_xii.n_samples_seen_, self.transformer_xij.n_samples_seen_)
        self.transformer_xij.scale_[ij_scaler_position] = self.transformer_xii.scale_[ii_scaler_position].copy()
        return self




class FrobeniusScaler:

    def __init__(self, diag_only = False):
        self.diag_only = diag_only
        self.require_norm = True

    
    def fit(self, xii, xij):
        return self
    
    def transform(self, xii, xij):
        norm = np.linalg.norm(xii, axis = 0)
        if not self.diag_only:
            norm += np.linalg.norm(xij, axis = 0)*np.sqrt(2)
        return xii/norm, xij/norm

    def batch_update(self, xii, xij):
        pass

    def new(self):
        """
        Create a brand-new FrobeniusScaler with the same init arguments,
        without copying any fitted state.
        """
        return FrobeniusScaler(diag_only=self.diag_only)

class FrobeniusSineScaler:

    def __init__(self, k_values_ii = [1, 0.5, 0.1, 5e-2, 1e-2, 1e-3], 
                    k_values_ij = [1, 0.5, 0.1, 5e-2, 1e-2, 1e-3], 
                    applies_to_ii = [], applies_to_ij = [], normalize = False):

        self.k_values_ii = k_values_ii
        self.k_values_ij = k_values_ij
        self.applies_to_ii = applies_to_ii
        self.applies_to_ij = applies_to_ij
        self.normalize = normalize
        self.require_norm = True


    
    def fit(self, xii, xij):
        return self
    
    def transform(self, xii, xij, norms_ii, norms_ij):

        new_xii, new_xij = [], []
        new_xii.append(xii/norms_ii)
        new_xij.append(xij/norms_ij)

        for i in self.applies_to_ii:
            for k in self.k_values_ii:
                new_xii.append(np.sin(new_xii[0][:,[i]]/k))
        for i in self.applies_to_ij:
            for k in self.k_values_ij:
                new_xij.append(np.sin(new_xij[0][:,[i]]/k))

        return np.concatenate(new_xii, axis=1), np.concatenate(new_xij, axis=1)

    def batch_update(self, xii, xij):
        pass

    def new(self):
        """
        Create a brand-new FrobeniusScaler with the same init arguments,
        without copying any fitted state.
        """
        return FrobeniusSineScaler(diag_only=self.diag_only,  k_values_ii = self.k_values_ii, k_values_ij = self.k_values_ij, 
        applies_to_ii = self.applies_to_ii, applies_to_ij = self.applies_to_ij)
