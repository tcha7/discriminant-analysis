# discriminant_analysis
A library for codes related to discriminant analysis

This implements the algorithms based on these papers.
- Chanyaswad, Thee, J. Morris Chang, and Sun-Yuan Kung. "A compressive multi-kernel method for privacy-preserving machine learning." 2017 International Joint Conference on Neural Networks (IJCNN). IEEE, 2017.
- Kung, Sun-Yuan, et al. "Collaborative PCA/DCA learning methods for compressive privacy." ACM Transactions on Embedded Computing Systems (TECS) 16.3 (2017): 1-18.
- Chanyaswad, Thee, et al. "Discriminant-component eigenfaces for privacy-preserving face recognition." 2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2016.
- Chanyaswad, Thee, et al. "Differential mutual information forward search for multi-kernel discriminant-component selection with an application to privacy-preserving classification." 2017 IEEE 27th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2017.
- Kung, Sun-Yuan. "Discriminant component analysis for privacy protection and visualization of big data." Multimedia Tools and Applications 76 (2017): 3999-4034.


## Versioning
Version 1.1: 08/16/16
- Re-implement `DCA` to solve the eig-decomposition faster using `eigh` function.
- Old DCA renamed as DCA_v0.

Version 1.2: 10/14/16
- Correct the regularization term in `KDCA`.
- Old (incorrect) implementation renamed as KDCA_v0.

Version 2.0: 11/16/16
- Correct the `_get_Kmatrices` and `transform` in `KDCA`
- Old implementation renamed as KDCA_v1

Version 2.1: 11/17/16
- Re-implement the centering method to fix asymmetric gram matrix problem (and negative eigval of gram matrix problem)
- Re-implement the calculation of `Kw` for faster computation

Version 2.2: 11/18/16
- Use the symmetric mapping to cvs to solve for kdca instead in order to solve negative eigval of `Kbar2` and `Kbar` problem.
- Move argument input to the `init` function

Version 3.0: 2/2/17
- Add `KPCA`

Version 3.1: 3/30/17
- Fix svd did not converge with Laplacian kernel by using `gesvd` instead.

Version 4.0: 5/16/23
- Add typehint and some stylistic changes
- Add `KernelDiscriminantInformation`, `FisherDiscriminant`, and `DiscriminantInformation` for feature selection
