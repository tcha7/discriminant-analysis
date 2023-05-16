import numpy as np
import scipy
import warnings
from sklearn.metrics import pairwise
from sklearn import preprocessing

class DCA:
    """To use DCA
    1) Initialize:
        dca = DCA(rho=0.01, rho_p=0.01, n_components = None)

    2) Fit to data:
        dca.fit(X,y)
        *** It is recommended not to use X with more than 10M rows and 1k columns.
        
    3) Transform the data:
        (Assuming that x initially has 100 features and we want to reduce it down to 10 dimensions)
        new_dimension = 10
        x_transformed = dca.transform(x, dim=new_dimension)
    
    However, to get even better performance, it is recommended that we do the non-linear feature mapping first. One way to do this is to use the Nystroem kernel approximation as follows.
    
    >> from sklearn import preprocessing, kernel_approximation
    >>
    >> nys = kernel_approximation.Nystroem(kernel='rbf', gamma=1, n_components=1000)
    >> x_tr_trans = nys.fit_transform(x_tr)
    >> scaler = preprocessing.MinMaxScaler()
    >> x_tr_scaled = scaler.fit_transform(x_tr_trans)
    >> 
    >> dca = DCA(rho=0.01, rho_p=0.01, n_components=1000)
    >> dca.fit(x_tr_scaled, y)
    >> x_tr_dca = dca.transform(x_tr_scaled)
    >> 
    >> x_test_trans = nys.transform(x_test)
    >> x_test_scaled = scaler.transform(x_test_trans)
    >> 
    >> x_test_dca = dca.transform(x_test_scaled)

    You can choose different kernels in the Nystroem approx. Sklearn implemented these kernels already: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
    
    Also, note that, it is recommended not to use `n_components` > 1000, and `x_tr` doesn't have more than 10M rows or 1000 columns.
    """

    def __init__(
        self, rho: float = None, rho_p: float = None, n_components: int = None
    ) -> None:
        self.n_components = n_components
        self.rho = rho
        self.rho_p = rho_p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, m = X.shape
        if ((n > 1e7) or (m > 1000)):
            warnings.warn("It is recommended not to use X with more than 10M rows and 1k columns. Otherwise, the algorithm may take up too much memory.")
        
        (self._Sw, self._Sb) = self._get_Smatrices(X, y)

        if self.rho == None:
            s0 = np.linalg.eigvalsh(self._Sw)
            self.rho = 0.02 * np.max(s0)
        if self.rho_p == None:
            self.rho_p = 0.1 * self.rho

        pSw = self._Sw + self.rho * np.eye(m)
        pSbar = self._Sb + self._Sw + (self.rho_p + self.rho) * np.eye(m)

        (s1, vr) = scipy.linalg.eigh(pSbar, pSw, overwrite_a=True, overwrite_b=True)
        s1 = s1[::-1]  # re-order from large to small
        Wdca = vr.T[::-1]
        self.eig_val = s1
        self.all_components = Wdca
        if self.n_components:
            self.components = Wdca[0 : self.n_components]
        else:
            self.components = Wdca

    def transform(self, X: np.ndarray, dim: int = None) -> np.ndarray:
        if dim == None:
            X_trans = np.inner(self.components, X)
        else:
            X_trans = np.inner(self.all_components[0:dim], X)
        return X_trans.T

    def inverse_transform(
        self,
        x_reduced: np.ndarray,
        projection_matrix: np.ndarray = None,
        dim: int = None,
    ) -> np.ndarray:
        if projection_matrix is None:
            if dim is None:
                W = self.components
            else:
                W = self.all_components[0:dim]
        else:
            W = projection_matrix
        # W = PxM where P<M
        foo = np.inner(W, W)
        bar = np.linalg.solve(foo.T, W)
        Xhat = np.inner(x_reduced, bar.T)
        return Xhat

    def _get_Smatrices(self, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        Sb = np.zeros((X.shape[1], X.shape[1]))

        S = np.inner(X.T, X.T)
        N = len(X)
        mu = np.mean(X, axis=0)
        classLabels = np.unique(y)
        for label in classLabels:
            classIdx = np.argwhere(y == label).T[0]
            Nl = len(classIdx)
            xL = X[classIdx]
            muL = np.mean(xL, axis=0)
            muLbar = muL - mu
            Sb = Sb + Nl * np.outer(muLbar, muLbar)

        Sbar = S - N * np.outer(mu, mu)
        Sw = Sbar - Sb
        self.mean_ = mu

        return (Sw, Sb)
    
    class KDCA:
        """To use KDCA
        1) Initialize:
            kdca = KDCA(rho=0.01, rho_p=0.01, kernel='rbf', gamma=1, degree=3, coef0=1)
        2) Fit to data:
            kdca.fit(X,y)
        3) Transform the data:
            (Assuming that x initially has 100 features and we want to reduce it down to 10 dimensions)
            new_dimension = 10
            x_transformed = kdca.transform(x, dim=new_dimension)
        """
    def __init__(
        self,
        rho: float = None,
        rho_p: float = None,
        n_components: int = None,
        kernel: str = "rbf",
        gamma: float = 1,
        degree: int = 3,
        coef0: float = 1,
    ) -> None:
        self.n_components = n_components
        self.rho = rho
        self.rho_p = rho_p
        self._kernel = kernel
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X = X
        (self._K, self._Kbar, self._Kbar2, self._Kw, self._Kb) = self._get_Kmatrices(
            X, y
        )

        if self.rho == None:
            s0 = np.linalg.eigvalsh(self._Kbar2)
            self.rho = 0.02 * np.max(s0)
        if self.rho_p == None:
            self.rho_p = 0.1 * self.rho

        pKb = self._Kb + self.rho_p * self._Kbar
        pKbar2 = self._Kbar2 + self.rho * self._Kbar
        (u, s, vT) = scipy.linalg.svd(pKbar2)
        s2 = np.diag(1.0 / np.sqrt(s))
        pKbar2_nhalf = np.inner(u, s2)

        pKb_cvs = np.inner(np.inner(pKbar2_nhalf.T, pKb), pKbar2_nhalf.T)
        (s3, vr) = scipy.linalg.eigh(pKb_cvs, overwrite_a=True)
        ## cols of u/ rows of vT are eigvect
        self.eig_val = s3[::-1]

        ## backward mapping
        alphaEvs = np.inner(vr.T, pKbar2_nhalf)
        Akdca = alphaEvs[::-1]

        self.all_components = Akdca
        if self.n_components:
            self.components = Akdca[0 : self.n_components]
        else:
            self.components = Akdca
        self._alphaBar = Akdca - np.outer(
            np.sum(Akdca, axis=1), np.ones(len(self._X))
        ) / len(self._X)

    def transform(self, x: np.ndarray, dim: int = None) -> np.ndarray:
        kx = self._get_kernel_matrix(
            self._X, x
        )  # kx = [k(x1) k(x2) ...] where x = [x1 x2 ...].T
        if dim is None:
            alphaBar = self._alphaBar[0 : self.n_components]
        else:
            alphaBar = self._alphaBar[0:dim]

        X_trans = np.inner(alphaBar, kx.T)

        return X_trans.T

    def _get_Kmatrices(
        self, X: np.ndarray, y: np.ndarray
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        K = self._get_kernel_matrix(X, X)
        N = len(X)
        Kw = np.zeros((N, N))
        classLabels = np.unique(y)
        for label in classLabels:
            classIdx = np.argwhere(y == label).T[0]
            Nl = len(classIdx)
            xL = X[classIdx]
            Kl = self._get_kernel_matrix(X, xL)
            Kmul = np.sum(Kl, axis=1) / Nl  # vector
            Kmul = np.outer(Kmul, np.ones(Nl))  # matrix
            Klbar = Kl - Kmul
            Kw = Kw + np.inner(Klbar, Klbar)

        # centering
        KwCenterer = preprocessing.KernelCenterer()
        KwCenterer.fit(Kw)
        Kw = KwCenterer.transform(Kw)
        KCenterer = preprocessing.KernelCenterer()
        KCenterer.fit(K)
        Kbar = KCenterer.transform(K)

        Kbar2 = np.inner(Kbar, Kbar.T)
        Kb = Kbar2 - Kw
        return (K, Kbar, Kbar2, Kw, Kb)

    def _get_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # K is len(X1)-by-len(X2) matrix
        if self._kernel == "rbf":
            K = pairwise.rbf_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "poly":
            K = pairwise.polynomial_kernel(
                X1, X2, degree=self._degree, gamma=self._gamma, coef0=self._coef0
            )
        elif self._kernel == "linear":
            K = pairwise.linear_kernel(X1, X2)
        elif self._kernel == "laplacian":
            K = pairwise.laplacian_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "chi2":
            K = pairwise.chi2_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "additive_chi2":
            K = pairwise.additive_chi2_kernel(X1, X2)
        elif self._kernel == "sigmoid":
            K = pairwise.sigmoid_kernel(X1, X2, gamma=self._gamma, coef0=self._coef0)
        else:
            print("[Error] Unknown kernel")
            K = None

        return K
    


class DiscriminantInformation:
    """To use DiscriminantInformation
    1) Initialize:
        di = DiscriminantInformation()

    2) get the discriminant information
        di_value = di.get_di(X,y)

    It is recommended to account for non-linearity with rbf kernel via nystroem approximation.
    Let's assume here that x is a single-feature n-array, e.g. x = X[:,0]
    1) First scale the feature either by min-max or by standardization.
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        # scaler = preprocessing.StandardScaler()
        x_sc = scaler.fit_transform(x)
    2) Do the kernel mapping with Nystroem approximation.
        nys = kernel_approximation.Nystroem(kernel='rbf', gamma=1, n_components=1000)
        x_trans = nys.fit_transform(x_sc.reshape(-1,1))
    3) get the discriminant information
        di = DiscriminantInformation()
        di_value = di.get_di(x_trans,y)
    """

    def __init__(self, rho: float = 0.01):
        self.rho = rho

    def get_di(self, X: np.ndarray, y: np.ndarray) -> float:
        n, m = X.shape
        (self._Sbar, self._Sb) = self._get_Smatrices(X, y)
        pSbar = self._Sbar + self.rho * np.eye(m)
        discriminant_mat = np.linalg.solve(pSbar, self._Sb)
        di = np.sum(np.diag(discriminant_mat))
        return di

    def _get_Smatrices(self, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        Sb = np.zeros((X.shape[1], X.shape[1]))

        S = np.inner(X.T, X.T)
        N = len(X)
        mu = np.mean(X, axis=0)
        classLabels = np.unique(y)
        for label in classLabels:
            classIdx = np.argwhere(y == label).T[0]
            Nl = len(classIdx)
            xL = X[classIdx]
            muL = np.mean(xL, axis=0)
            muLbar = muL - mu
            Sb = Sb + Nl * np.outer(muLbar, muLbar)

        Sbar = S - N * np.outer(mu, mu)
        # Sw = Sbar - Sb
        self.mean_ = mu

        return (Sbar, Sb)


class FisherDiscriminant:
    """To use class FisherDiscriminant:
    1) Initialize:
        fisher = FisherDisriminant()

    2) get the discriminant information
        fdr_value = fisher.get_fdr(x,y)
    """
    def __init__(self, rho: int = 0.01):
        self.rho = rho

    def get_fdr(self, x: np.ndarray, y: np.ndarray) -> float:
        (self._Sbar, self._Sb) = self._get_scatter(x, y)
        pSbar = self._Sbar + self.rho
        fdr = self._Sb / pSbar
        return fdr

    def _get_scatter(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        mu = np.mean(x)
        xbar = x - mu
        Sbar = np.inner(xbar, xbar)
        classLabels = np.unique(y)
        Sb = 0
        for label in classLabels:
            classIdx = np.where(y == label)[0]
            Nl = len(classIdx)
            xL = x[classIdx]
            muL = np.mean(xL)
            muLbar = muL - mu
            Sb = Sb + Nl * (muLbar**2)
        # Sw = Sbar - Sb
        return (Sbar, Sb)


class KernelDiscriminantInformation:
    """To use class KernelDiscriminantInformation:
    1) Initialize:
        kdi = KernelDiscriminantInformation()

    2) get the discriminant information
        kdi_value = kdi.get_kdi(X,y)
    
    ** X can be a (nxm) array or a single-feature n-array.
    However, it is recommended to scale X first. Otherwise, you will need to tune `gamma` (if you use rbf kernel).
    """
    def __init__(
        self,
        rho=0.01,
        kernel="rbf",
        gamma=1,
        degree=3,
        coef0=1,
    ):
        self.rho = rho
        self._kernel = kernel
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0

    def get_kdi(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1,1)

        (self._K, self._Kbar, self._Kbar2, self._Kw, self._Kb) = self._get_Kmatrices(
            X, y
        )
        pKbar2 = self._Kbar2 + self.rho * self._Kbar
        discriminant_mat = np.linalg.solve(pKbar2, self._Kb)
        kdi = np.sum(np.diag(discriminant_mat))
        return kdi

    def _get_Kmatrices(self, X, y):
        K = self._get_kernel_matrix(X, X)
        N = len(X)
        Kw = np.zeros((N, N))
        classLabels = np.unique(y)
        for label in classLabels:
            classIdx = np.where(y == label)[0]
            Nl = len(classIdx)
            xL = X[classIdx]
            Kl = self._get_kernel_matrix(X, xL)
            Kmul = np.sum(Kl, axis=1) / Nl  # vector
            Kmul = np.outer(Kmul, np.ones(Nl))  # matrix
            Klbar = Kl - Kmul
            Kw = Kw + np.inner(Klbar, Klbar)
        # centering
        KwCenterer = preprocessing.KernelCenterer()
        KwCenterer.fit(Kw)
        Kw = KwCenterer.transform(Kw)
        KCenterer = preprocessing.KernelCenterer()
        KCenterer.fit(K)
        Kbar = KCenterer.transform(K)
        Kbar2 = np.inner(Kbar, Kbar.T)
        Kb = Kbar2 - Kw
        return (K, Kbar, Kbar2, Kw, Kb)

    def _get_kernel_matrix(self, X1, X2):
        # K is len(X1)-by-len(X2) matrix
        if self._kernel == "rbf":
            K = pairwise.rbf_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "poly":
            K = pairwise.polynomial_kernel(
                X1, X2, degree=self._degree, gamma=self._gamma, coef0=self._coef0
            )
        elif self._kernel == "linear":
            K = pairwise.linear_kernel(X1, X2)
        elif self._kernel == "laplacian":
            K = pairwise.laplacian_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "chi2":
            K = pairwise.chi2_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "additive_chi2":
            K = pairwise.additive_chi2_kernel(X1, X2)
        elif self._kernel == "sigmoid":
            K = pairwise.sigmoid_kernel(X1, X2, gamma=self._gamma, coef0=self._coef0)
        else:
            print("[Error] Unknown kernel")
            K = None
        return K

class KPCA:
    """
    To use KPCA
    1) Initialize:
        kpca = KPCA(rho=0.01, rho_p=0.01, kernel='rbf', gamma=1, degree=3, coef0=1)

    2) Fit to data:
        kpca.fit(X)
        
    3) Transform the data:
        (Assuming that x initially has 100 features and we want to reduce it down to 10 dimensions)
        new_dimension = 10
        x_transformed = kpca.transform(x, dim=new_dimension)
    """
    def __init__(self, n_components=None, kernel="rbf", gamma=1, degree=3, coef0=1):
        self.n_components = n_components
        self._kernel = kernel
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0

    def fit(self, X):
        self._X = X
        (self._K, self._Kbar) = self._get_Kmatrices(X)
        (u, s, vT) = scipy.linalg.svd(self._Kbar, lapack_driver="gesvd")
        self.eigVal = s
        ## scale to meet the constraint
        sqrtSinv = np.diag(1.0 / np.sqrt(s))
        alphaEvs = np.inner(sqrtSinv, u)
        Akdca = alphaEvs

        self.allComponents = Akdca
        if self.n_components:
            self.components = Akdca[0 : self.n_components]
        else:
            self.components = Akdca
        self._alphaBar = Akdca - np.outer(
            np.sum(Akdca, axis=1), np.ones(len(self._X))
        ) / len(self._X)

    def transform(self, x, dim=None):
        kx = self._get_kernel_matrix(
            self._X, x
        )  # kx = [k(x1) k(x2) ...] where x = [x1 x2 ...].T
        if dim is None:
            alphaBar = self._alphaBar[0 : self.n_components]
        else:
            alphaBar = self._alphaBar[0:dim]
        X_trans = np.inner(alphaBar, kx.T)
        return X_trans.T

    def _get_Kmatrices(self, X):
        K = self._get_kernel_matrix(X, X)
        KCenterer = preprocessing.KernelCenterer()
        KCenterer.fit(K)
        Kbar = KCenterer.transform(K)
        return (K, Kbar)

    def _get_kernel_matrix(self, X1, X2):
        # K is len(X1)-by-len(X2) matrix
        if self._kernel == "rbf":
            K = pairwise.rbf_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "poly":
            K = pairwise.polynomial_kernel(
                X1, X2, degree=self._degree, gamma=self._gamma, coef0=self._coef0
            )
        elif self._kernel == "linear":
            K = pairwise.linear_kernel(X1, X2)
        elif self._kernel == "laplacian":
            K = pairwise.laplacian_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "chi2":
            K = pairwise.chi2_kernel(X1, X2, gamma=self._gamma)
        elif self._kernel == "additive_chi2":
            K = pairwise.additive_chi2_kernel(X1, X2)
        elif self._kernel == "sigmoid":
            K = pairwise.sigmoid_kernel(X1, X2, gamma=self._gamma, coef0=self._coef0)
        else:
            print("[Error] Unknown kernel")
            K = None

        return K


def trbf_mapping(X: np.ndarray, order: int, sigma: float = 1.0) -> np.ndarray:
    coef = np.exp(-1 * (np.diag(np.inner(X, X))) / (2 * sigma * sigma))
    phi = np.ones(len(X)).reshape(len(X), 1)
    for i in range(1, order + 1):
        monoMap = _monomial_mapping(X, i)
        monoMap = monoMap / (np.sqrt(scipy.special.factorial(i)) * (sigma**i))
        phi = np.hstack((phi, monoMap))
    coef = coef.reshape(len(coef), 1)
    phi = phi * coef
    return phi
    
def _monomial_mapping(X, p):
    # k(x,y) = (xTy)^p = phi(x)*phi(y)
    M = X.shape[1]
    N = X.shape[0]
    (A, B) = _expander(M, p)

    negIdx = np.where(X < 0)
    X = np.absolute(X)
    logX = np.log(X) + 0 * 1j
    infIdx = np.where(logX < -1000)
    logX[infIdx] = -1000
    logX[negIdx] = logX[negIdx] + 1j * np.pi
    foo = np.inner(logX, A)
    phi = np.exp(foo)
    phi = phi.real

    return phi


def _expander(n, d):
    if (n <= 0.0) or (d < 0.0):
        return None

    if (n > 1.0) and (d > 0.0):
        (A1, B1) = _expander(n - 1, d)
        (A2, B2) = _expander(n, d - 1)
        if len(A1.shape) == 1:
            a = np.zeros(1)
        else:
            M = A1.shape[0]
            a = np.zeros(M).reshape(M, 1)
        foo = np.hstack((A1, a))
        if len(A2.shape) == 1:
            b = A2[: n - 1]
            c = A2[n - 1]
        else:
            b = A2[:, : n - 1]
            c = A2[:, n - 1]
            c = c.reshape(len(c), 1)
        bar = np.hstack((b, 1.0 + c))
        A = np.vstack((foo, bar))
        foo = B1
        bar = B2 * d / (1.0 + c)
        B = np.vstack((foo, bar))
    else:
        B = np.array([1.0])
        if n == 1.0:
            A = np.array([d])
        else:
            A = np.zeros(n)
    return (A, B)


def polynomial_mapping(X: np.ndarray, order: int) -> np.ndarray:
    # k(x,y) = (xTy + 1)^p = phi(x)*phi(y)
    poly = preprocessing.PolynomialFeatures(
        degree=order, interaction_only=False, include_bias=True
    )
    phi = poly.fit_transform(X)
    return phi