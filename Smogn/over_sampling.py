## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm

## load dependencies - internal
from smogn.box_plot_stats import box_plot_stats
from smogn.dist_metrics import euclidean_dist, heom_dist, overlap_dist

## generate synthetic observations
def over_sampling(
    
    ## arguments / inputs
    data,       ## training set
    index,      ## index of input data
    perc,       ## over / under sampling
    pert,       ## perturbation / noise percentage
    k           ## num of neighs for over-sampling
    ):
    """
    generates synthetic observations and is the primary function underlying the
    over-sampling technique utilized in the higher main function 'smogn()', the
    4 step procedure for generating synthetic observations is:
    
    1) pre-processing: temporarily removes features without variation, label 
    encodes nominal / categorical features, and subsets the training set into 
    two data sets by data type: numeric / continuous, and nominal / categorical
    
    2) distances: calculates the cartesian distances between all observations, 
    distance metric automatically determined by data type (euclidean distance 
    for numeric only data, heom distance for both numeric and nominal data, and 
    hamming distance for nominal only data) and determine k nearest neighbors
    
    3) over-sampling: selects between two techniques, either synthetic minority 
    over-sampling technique for regression 'smoter' or 'smoter-gn' which applies
    a similar interpolation method to 'smoter', but perterbs the interpolated 
    values
    
    'smoter' is selected when the distance between a given observation and a 
    selected nearest neighbor is within the maximum threshold (half the median 
    distance of k nearest neighbors) 'smoter-gn' is selected when a given 
    observation and a selected nearest neighbor exceeds that same threshold
    
    both 'smoter' and 'smoter-gn' only applies to numeric / continuous features, 
    for nominal / categorical features, synthetic values are generated at random 
    from sampling observed values found within the same feature
    
    4) post processing: restores original values for label encoded features, 
    reintroduces constant features previously removed, converts any interpolated
    negative values to zero in the case of non-negative features
    
    returns a pandas dataframe containing synthetic observations of the training
    set which are then returned to the higher main function 'smogn()'
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    """
    
    ## subset original dataframe by bump classification index
    data = data.iloc[index]
    
    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)
    
    x_synth = int(perc)
    ## total number of new synthetic observations to generate
    n_synth = int(n * (perc - x_synth))
    ## create null matrix to store new synthetic observations
    synth_matrix = np.ndarray(shape = ((x_synth * n + n_synth), d))
    print("x_synth value:", x_synth)
    print("n_synth value:", n_synth)
    print("Synth_matrix shape:", synth_matrix.shape)
    print("data length:", n)
    
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## find non-negative numeric features
    feat_non_neg = [] 
    num_dtypes = ["int64", "float64"]
    
    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and any(data.iloc[:, j] > 0):
            feat_non_neg.append(j)
    
    ## find features without variation (constant features)
    feat_const = data.columns[data.nunique() == 1]
    
    ## temporarily remove constant features
    if len(feat_const) > 0:
        
        ## create copy of orignal data and omit constant features
        data_orig = data.copy()
        data = data.drop(data.columns[feat_const], axis = 1)
        
        ## store list of features with variation
        feat_var = list(data.columns.values)
        
        ## reindex features with variation
        for i in range(d - len(feat_const)):
            data.rename(columns = {
                data.columns[i]: i
                }, inplace = True)
        
        ## store new dimension of feature space
        d = len(data.columns)
    
    ## create copy of data containing variation
    data_var = data.copy()
    
    ## create global feature list by column index
    feat_list = list(data.columns.values)
    
    ## create nominal feature list and
    ## label encode nominal / categorical features
    ## (strictly label encode, not one hot encode) 
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64"]
    
    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)
            data.iloc[:, j] = pd.Categorical(pd.factorize(
                data.iloc[:, j])[0])
    
    data = data.apply(pd.to_numeric)
    
    ## create numeric feature list
    feat_list_num = list(set(feat_list) - set(feat_list_nom))
    
    ## calculate ranges for numeric / continuous features
    ## (includes label encoded features)
    feat_ranges = list(np.repeat(1, d))
    
    if len(feat_list_nom) > 0:
        for j in feat_list_num:
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    else:
        for j in range(d):
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    
    ## subset feature ranges to include only numeric features
    ## (excludes label encoded features)
    feat_ranges_num = [feat_ranges[i] for i in feat_list_num]
    
    ## subset data by either numeric / continuous or nominal / categorical
    data_num = data.iloc[:, feat_list_num]
    data_nom = data.iloc[:, feat_list_nom]
    
    ## get number of features for each data type
    feat_count_num = len(feat_list_num)
    feat_count_nom = len(feat_list_nom)
    
    ## calculate distance between observations based on data types
    ## store results over null distance matrix of n x n
    dist_matrix = np.ndarray(shape = (n, n))
    print("We are starting the process")
    print(dist_matrix.shape)

    NpArray = np.ndarray(shape = (len(data_num), len(data_num), feat_count_num), dtype = np.float32)
    for i in range(len(data_num)):
        for j in range(len(data_num)):
            NpArray[i][j] = np.array(data_num.iloc[i, :]) # set each of the row elements to be the same repeated pandas dataframe element. 
    
    NpArray2 = np.ndarray(shape = (len(data_num), len(data_num), feat_count_num), dtype = np.float32)
    for i in range(len(data_num)):
        for j in range(len(data_num)):
            NpArray2[i][j] = NpArray[j][i]
    ## calculate distance between observations based on data types
    ## store results over null distance matrix of n x n
    print("We are starting the loop")
    dist_matrix = NpArray - NpArray2
    dist_matrix = np.apply_along_axis(lambda x: x ** 2 , 2, dist_matrix)
    dist_matrix = np.sum(dist_matrix, axis = 2, dtype = np.float32)
    dist_matrix = np.sqrt(dist_matrix, dtype = np.float32) ### each row contains distance between node i and node j. j being in the columns. 
                                                           ### within the dist_matrix, add a dimension where we contain the index along with distance. 
    del NpArray
    del NpArray2            
    
    print("Calculated the squared distances.")
    print("Process completed")
    print(dist_matrix)
    ## determine indicies of k nearest neighbors
    ## and convert knn index list to matrix
    knn_index = [None] * n
    
    for i in range(n):
        knn_index[i] = np.argsort(dist_matrix[i])[1:k + 1] # k total neighbours, resulting to a n x k matrix. 
                # adding in another dimension to get the neighbor index. 
                                                            
    knn_matrix = np.array(knn_index)   # (n, k) dimensional matrix
    
    
    ## calculate max distances to determine if gaussian noise is applied
    ## (half the median of the distances per observation)
    max_dist = [None] * n
    for i in range(n):
        max_dist[i] = box_plot_stats(dist_matrix[i])["stats"][2] / 2 # find the conditional distance for each observation with their k neighbours.
    
    ## randomly index data by the number of new synthetic observations
    r_index = np.random.choice(
        a = tuple(range(0, n)), 
        size = n_synth, 
        replace = False, 
        p = None
    )
    
    ## create null matrix to store new synthetic observations
    synth_matrix = np.ndarray(shape = ((x_synth * n + n_synth), d))
    
    synth_matrix[:n][:] = data.values
    for i in tqdm(range(len(r_index)), ascii = True, desc = "synth_matrix"):
        index = r_index[i] # the random index we're choosing. 
        
        knnIndicies = knn_matrix[index] # list of neighbor indicies as elements of knn_matrix. 
        SubDist = dist_matrix[index,knnIndicies] # the list of the distance between index and its closest neighbors. 
        safe_list = [] 
        
        for j in range(len(knnIndicies)): 
            if SubDist[j] < max_dist[index]: 
                safe_list.append(knnIndicies[j]) # add in the neighbor index.  
        safe_list = np.array(safe_list) 
            
                
        neigh = int(np.random.choice(   # choose a random index ranging up to k.
                    a = tuple(range(k)), 
                    size = 1))
                
        ## conduct synthetic minority over-sampling if safe. 
        neighPoint = knn_matrix[index, neigh] # choose that random neighbor point. 
        
        if neighPoint in safe_list:
            diffs = synth_matrix[neighPoint][0:(d - 1)] - synth_matrix[index][0:(d - 1)] # good
            synth_matrix[n + i][0:(d - 1)] = synth_matrix[index][0:(d - 1)] + rd.random() * diffs # good
                    
                    
            casePoint = np.array(synth_matrix[index][0:(d - 1)])
            xPoint = np.array(synth_matrix[neighPoint][0:(d - 1)]) 
            newPoint = np.array(synth_matrix[n + i][0:(d - 1)]) 
                    
            D1 = np.linalg.norm(newPoint-casePoint)  
            D2 = np.linalg.norm(newPoint-xPoint)                
            synth_matrix[n + i][(d - 1)] = (D2 * synth_matrix[index][(d - 1)] + D1 * synth_matrix[neighPoint][(d - 1)]) / (D1 + D2)
                ## conduct synthetic minority over-sampling technique
                ## for regression with the introduction of gaussian 
                ## noise (smoter-gn)
        else:
            index_gaus = n + i
            if max_dist[index] < pert: 
                pert = max_dist[index] 

            for x in range(d):
                if pd.isna(data.iloc[index, x]):
                    synth_matrix[index_gaus][x] = None
                else:
                    synth_matrix[index_gaus][x] = data.iloc[index, x] + float(np.random.normal(loc = 0,scale = np.std(data.iloc[:, x]), size = 1) * pert)
    
    
    ## convert synthetic matrix to dataframe
    data_new = pd.DataFrame(synth_matrix)
    
    ## synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:
        raise ValueError("oops! synthetic data contains missing values")
    
    
    
    ## replace label encoded values with original values
    ## reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var
        
        for j in range(len(feat_const)):
            data_new.insert(
                loc = int(feat_const[j]),
                column = feat_const[j], 
                value = np.repeat(
                    data_orig.iloc[0, feat_const[j]], 
                    len(synth_matrix))
            )
    
    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower = 0)
    
    ## return over-sampling results dataframe
    return data_new