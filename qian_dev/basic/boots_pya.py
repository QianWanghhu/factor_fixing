#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pyapprox as pya
from scipy.stats import uniform
from sklearn.model_selection import KFold

def least_squares(basis_matrix_function,samples,values):
    """
    Help function of least square regression used for PCE.
    Parameters:
    ===========
    basis_matrix_function: 
    samples: ndarray of inputs, 
             shape of (P, N) where P is the number of inputs and N is the size of samples
    values: ndarray of outputs
    Returns:
    ========
    coef: ndarray, solution of the regression.
    """
    basis_matrix = basis_matrix_function(samples)
    coef = np.linalg.lstsq(basis_matrix,values,rcond=None)[0]
    return coef

def fun(variable, samples, values, degree=2, nboot=500, I=None, ntrain_samples=None):
    """
    Function to train PCE and conduct boostrap.
    Parameters:
    ===========
    fpath: str, file path for inputs and outputs
    nboot: int, the number of bootstraps
    variables: pyapprox.variables.IndependentMultivariateRandomVariable
    samples: ndarray of inputs, 
             shape of (P, N) where P is the number of inputs and N is the size of samples
    values: ndarray of outputs
    ntrain_samples: int, the size of training set, 
                    default is None which then will adopt 3 times the number of poly terms

    Returns:
    ========
    errors_cv: list, Errors computed from the cross validation.
    errors_bt: list, errors calculated from bootstrap
    sensitivity_indices: list, total effects for all the bootstraps
    """
    poly = pya.get_polynomial_from_variable(variable)
    poly.set_indices(pya.compute_hyperbolic_indices(
        variable.num_vars(),degree))

    # nsamples = samples.shape[1]
    if ntrain_samples == None:
        ntrain_samples = poly.get_indices().shape[1]*3

    errors_cv, main_sensitivity, total_indices = [], [], []
    index_cover = []
    # Cross-validation
    kf = KFold(n_splits=10)
    if nboot > 1:
        for ii in range(nboot):
        
            index_cover.append(np.unique(I[ii]).size / ntrain_samples)

            train_samples = samples[:, I[ii]]
            train_values = values[I[ii]]
            x_cv = train_samples
            y_cv = train_values        
            for train_index, test_index in kf.split(x_cv.T):
                x_train, x_test = x_cv[:,train_index], x_cv[:,test_index]
                y_train, y_test = y_cv[train_index], y_cv[test_index]
                coef = least_squares(poly.basis_matrix,x_train,y_train)
                poly.set_coefficients(coef)
                approx_values = poly(x_test)
                cv = np.linalg.norm(approx_values-y_test)/np.linalg.norm(train_values)
                errors_cv.append(cv)


            coef = least_squares(poly.basis_matrix,train_samples,train_values)
            poly.set_coefficients(coef)
            main_effect, total_effect = pya.get_main_and_total_effect_indices_from_pce(poly.get_coefficients(),poly.get_indices())
            total_indices.append(total_effect[:])
            main_sensitivity.append(main_effect[:])
        return errors_cv, main_sensitivity, total_indices, index_cover
    else:
        I = np.arange(ntrain_samples)
        train_samples = samples[:, I]
        train_values = values[I]
        coef = least_squares(poly.basis_matrix,train_samples,train_values)
        poly.set_coefficients(coef)
        return poly


from pyapprox.approximate import approximate
from pyapprox.utilities import total_degree_space_dimension
from scipy import stats
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.variables import get_distribution_info
def identity_fun(x):
    return x

def get_poly_opts(variable, product_uniform):

    if product_uniform != 'exact':
        var_trans = AffineRandomVariableTransformation(
            variable)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        return poly_opts, var_trans
    
    from basic.read_data import file_settings
    from basic.utils import variables_prep
    input_path = file_settings()[1]
    filename = file_settings()[4]
    index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
    full_variable = variables_prep(filename, product_uniform=False)
    var_trans = AffineRandomVariableTransformation(
        variable)
    poly = PolynomialChaosExpansion()
    basis_opts = dict()
    identity_map_indices = []
    cnt = 0
    for ii in range(variable.nunique_vars):
        rv = variable.unique_variables[ii]
        name, scales, shapes = get_distribution_info(rv)
        if (type(rv.dist) != stats._continuous_distns.beta_gen):
            opts = {'rv_type': name, 'shapes': shapes,
                    'var_nums': variable.unique_variable_indices[ii]}
            basis_opts['basis%d' % ii] = opts
            continue

        identity_map_indices += variable.unique_variable_indices[ii]
        
        quad_rules = []
        inds = index_product[cnt]
        nquad_samples_1d = 50

        for jj in inds:
            a, b = full_variable.all_variables()[jj].interval(1)
            x, w = gauss_jacobi_pts_wts_1D(nquad_samples_1d, 0, 0)
            x = (x+1)/2 # map to [0, 1]
            x = (b-a)*x+a # map to [a,b]
            quad_rules.append((x, w))
        funs = [identity_fun]*len(inds)
        basis_opts['basis%d' % ii] = {'poly_type': 'product_indpnt_vars',
                                      'var_nums': [ii], 'funs': funs,
                                      'quad_rules': quad_rules}
        cnt += 1
        
    poly_opts = {'var_trans': var_trans}
    poly_opts['poly_types'] = basis_opts
    var_trans.set_identity_maps(identity_map_indices)
    return poly_opts, var_trans


from pyapprox.indexing import compute_hyperbolic_indices
from pyapprox.multivariate_polynomials import \
        define_poly_options_from_variable_transformation
def fun_new(variable, train_samples, train_values, product_uniform, nboot=10):
    #TODO Need to pass in variables that make up product so I can construct
    # quadrature rules

    poly_opts, var_trans = get_poly_opts(variable, product_uniform)

    if nboot == 1:
        indices = compute_hyperbolic_indices(variable.num_vars(), 2)
        options = {'basis_type': 'fixed', 'variable': variable,
                   'poly_opts': poly_opts,
                   'options': {'linear_solver_options': dict(),
                               'indices': indices, 'solver_type': 'lstsq'}}
        approx_res = approximate(
            train_samples, train_values, 'polynomial_chaos', options)
        return approx_res.approx
    
    nterms = total_degree_space_dimension(train_samples.shape[0], 2)

    # Find best PCE basis
    nfolds = min(nboot, train_samples.shape[1])
    # solver_options = {'cv': nfolds}
    # options = {'basis_type': 'expanding_basis', 'variable': variable,
    #            'verbosity': 0, 'options': {'max_num_init_terms': nterms,
    #            'linear_solver_options': solver_options}}
    # approx_res = approximate(train_samples, train_values, 'polynomial_chaos', options)

    # # Compute PCE on each fold using best PCE basis and least squares
    # nfolds = min(nboot, train_samples.shape[1])
    # linear_solver_options = [
    #     {'alpha':approx_res.reg_params[ii]}
    #     for ii in range(len(approx_res.reg_params))]
    # indices = [approx_res.approx.indices[:, np.where(np.absolute(c)>0)[0]]
    #            for c in approx_res.approx.coefficients.T]

    # for now just use quadratic basis
    indices = compute_hyperbolic_indices(variable.num_vars(), 2)
    options = {'basis_type': 'fixed', 'variable': variable,
               'poly_opts': poly_opts,
               'options': {'linear_solver_options': dict(),
                           'indices': indices, 'solver_type': 'lstsq'}}
    from pyapprox.approximate import cross_validate_approximation
    # this does not use fast leave many out cross validation for least squares
    # (which is used by approximate because that function does not return
    # all the approximations on each fold
    approx_list, residues_list, cv_score = cross_validate_approximation(
        train_samples, train_values, options, nfolds,
        'polynomial_chaos', random_folds='sklearn')
    pce_cv_total_effects = []
    pce_cv_main_effects = []
    for ii in range(nfolds):
        pce_sa_res_ii = pya.analyze_sensitivity_polynomial_chaos(
            approx_list[ii])
        pce_cv_main_effects.append(pce_sa_res_ii.main_effects)
        pce_cv_total_effects.append(pce_sa_res_ii.total_effects)
    return cv_score, pce_cv_main_effects, pce_cv_total_effects, approx_list
