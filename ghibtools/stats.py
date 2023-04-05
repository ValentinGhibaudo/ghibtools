import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import statsmodels.formula.api as smf

def mad(data, constant = 1.4826):
    median = np.median(data)
    return np.median(np.abs(data - median)) * constant

def normality(df, predictor, outcome):
    df = df.reset_index(drop=True)
    groups = list(set(df[predictor]))
    n_groups = len(groups)

    normalities = pg.normality(data = df , dv = outcome, group = predictor)['normal']
    
    if sum(normalities) == normalities.size:
        normal = True
    else:
        normal = False
    
    return normal

def sphericity(df, predictor, outcome, subject):
    spher, W , chi2, dof, pval = pg.sphericity(data = df, dv = outcome, within = predictor, subject = subject)
    return spher

def homoscedasticity(df, predictor, outcome):
    homoscedasticity = pg.homoscedasticity(data = df, dv = outcome, group = predictor)['equal_var'].values[0]
    return homoscedasticity

def parametric(df, predictor, outcome, subject = None):
    df = df.reset_index(drop=True)
    groups = list(set(df[predictor]))
    n_groups = len(groups)
    
    normal = normality(df, predictor, outcome)

    if subject is None:
        equal_var = homoscedasticity(df, predictor, outcome)
    else:
        equal_var = sphericity(df, predictor, outcome, subject)
    
    if normal and equal_var:
        parametricity = True
    else:
        parametricity = False
        
    return parametricity


def guidelines(df, predictor, outcome, design, parametricity):
        
    n_groups = len(list(set(df[predictor])))
    
    if parametricity:
        if n_groups <= 2:
            if design == 'between':
                tests = {'pre':'t-test_ind','post':None}
            elif design == 'within':
                tests = {'pre':'t-test_paired','post':None}
        else:
            if design == 'between':
                tests = {'pre':'anova','post':'pairwise_tukey'}
            elif design == 'within':
                tests = {'pre':'rm_anova','post':'pairwise_ttests_paired_paramTrue'}
    else:
        if n_groups <= 2:
            if design == 'between':
                tests = {'pre':'Mann-Whitney','post':None}
            elif design == 'within':
                tests = {'pre':'Wilcoxon','post':None}
        else:
            if design == 'between':
                tests = {'pre':'Kruskal','post':'pairwise_ttests_ind_paramFalse'}
            elif design == 'within':
                tests = {'pre':'friedman','post':'pairwise_ttests_paired_paramFalse'}
                
    return tests

def pg_compute_pre(df, predictor, outcome, test, subject=None, show = False):
    
    pval_labels = {'t-test_ind':'p-val','t-test_paired':'p-val','anova':'p-unc','rm_anova':'p-unc','Mann-Whitney':'p-val','Wilcoxon':'p-val', 'Kruskal':'p-unc', 'friedman':'p-unc'}
    esize_labels = {'t-test_ind':'cohen-d','t-test_paired':'cohen-d','anova':'np2','rm_anova':'np2','Mann-Whitney':'CLES','Wilcoxon':'CLES', 'Kruskal':None, 'friedman':None}
    
    if test == 't-test_ind':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=False)
        
    elif test == 't-test_paired':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        res = pg.ttest(pre, post, paired=True)
        
    elif test == 'anova':
        res = pg.anova(dv=outcome, between=predictor, data=df, detailed=False, effsize = 'np2')
    
    elif test == 'rm_anova':
        res = pg.rm_anova(dv=outcome, within=predictor, data=df, detailed=False, effsize = 'np2', subject = subject)
        
    elif test == 'Mann-Whitney':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.mwu(x, y)
        
    elif test == 'Wilcoxon':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        res = pg.wilcoxon(x, y)
        
    elif test == 'Kruskal':
        res = pg.kruskal(data=df, dv=outcome, between=predictor)
        
    elif test == 'friedman':
        res = pg.friedman(data=df, dv=outcome, within=predictor, subject=subject)
    
    pval = res[pval_labels[test]].values[0]
    es_label = esize_labels[test]
    if es_label is None:
        es = None
    else:
        es = res[es_label].values[0]
    
    es_interp = es_interpretation(es_label, es)
    results = {'p':pval, 'es':es, 'es_label':es_label, 'es_interp':es_interp}
      
    return results

def es_interpretation(es_label , es_value):

    if es_label == 'cohen-d' or es_label == 'CLES':
        if es_value < 0.2:
            interpretation = 'VS'
        elif es_value >= 0.2 and es_value < 0.5:
            interpretation = 'S'
        elif es_value >= 0.5 and es_value < 0.8:
            interpretation = 'M'
        elif es_value >= 0.8 and es_value < 1.3:
            interpretation = 'L'
        else:
            interpretation = 'VL'
    
    elif es_label == 'np2':
        if es_value < 0.01:
            interpretation = 'VS'
        elif es_value >= 0.01 and es_value < 0.06:
            interpretation = 'S'
        elif es_value >= 0.06 and es_value < 0.14:
            interpretation = 'M'
        else:
            interpretation = 'L'
            
    elif es_label is None:
        interpretation = None
                
    return interpretation

def get_stats_tests():
    
    ttest_ind = ['parametric', 'indep', 2, 't-test_ind' , 'NA']
    ttest_paired = ['parametric', 'paired', 2, 't-test_paired', 'NA']
    anova = ['parametric', 'indep', '3 ou +', 'anova', 'pairwise_tukey']
    rm_anova = ['parametric', 'paired', '3 ou +', 'rm_anova', 'pairwise_ttests_paired_paramTrue']
    mwu = ['non parametric', 'indep', 2, 'Mann-Whitney',  'NA']
    wilcox = ['non parametric', 'paired', 2, 'Wilcoxon', 'NA']
    kruskal = ['non parametric', 'indep', '3 ou +', 'Kruskal','pairwise_ttests_ind_paramFalse']
    friedman = ['non parametric', 'paired', '3 ou +', 'friedman', 'pairwise_ttests_paired_paramFalse']
    
    rows = [ttest_ind, ttest_paired, anova, rm_anova, mwu , wilcox, kruskal, friedman ]
    
    df=pd.DataFrame(rows , columns = ['parametricity','paired','samples','test','post_hoc'])
    df = df.set_index(['parametricity','paired','samples'])
    return df

def homemade_post_hoc(df, predictor, outcome, design = 'within', subject = None, parametric = True):
    pairs = pg.pairwise_tests(data=df, dv = outcome, within = predictor, subject = subject, parametric = False).loc[:,['A','B']]
    pvals = []
    for i, pair in pairs.iterrows():
        x = df[df[predictor] == pair[0]][outcome]
        y = df[df[predictor] == pair[1]][outcome]

        if design == 'within':
            if parametric:
                p = pg.ttest(x, y, paired= True)['p-val']
            else:
                p = pg.wilcoxon(x, y)['p-val']
        elif design == 'between':
            if parametric:
                p = pg.ttest(x, y, paired= False)['p-val']
            else:
                p = pg.mwu(x, y)['p-val']
        pvals.append(p.values[0])
        
    pairs['p-unc'] = pvals
    _, pvals_corr = pg.multicomp(pvals)
    pairs['p-corr'] = pvals_corr
    return pairs
        
def pg_compute_post_hoc(df, predictor, outcome, test, subject=None):

    if not subject is None:
        n_subjects = df[subject].unique().size
    else:
        n_subjects = df[predictor].value_counts()[0]
    
    if test == 'pairwise_tukey':
        res = pg.pairwise_tukey(data = df, dv=outcome, between=predictor)
        res['p-corr'] = pg.multicomp(res['p-tukey'])[1]

    elif test == 'pairwise_ttests_paired_paramTrue':
        res = pg.pairwise_tests(data = df, dv=outcome, within=predictor, subject=subject, parametric=True, padjust = 'holm')
        # res = homemade_post_hoc(df = df, outcome=outcome, predictor=predictor, design = 'within', subject=subject, parametric=True)
        
    elif test == 'pairwise_ttests_ind_paramFalse':
        if n_subjects >= 15:
            res = pg.pairwise_tests(data = df, dv=outcome, between=predictor, parametric=True, padjust = 'holm')
        else:
            res = permutation(df = df, outcome=outcome, predictor=predictor, design = 'between')

    elif test == 'pairwise_ttests_paired_paramFalse':
        if n_subjects >= 15:
            res = pg.pairwise_tests(data = df, dv=outcome, within=predictor, subject=subject, parametric=False, padjust = 'holm')
        else:
            res = permutation(df = df, outcome=outcome, predictor=predictor, design = 'within')
     
    return res

def auto_annotated_stats(df, predictor, outcome, test):
    
    x = predictor
    y = outcome

    order = list(set(df[predictor]))

    ax = sns.boxplot(data=df, x=x, y=y, order=order)
    pairs=[(order[0],order[1])]
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
    annotator.configure(test=test, text_format='star', loc='inside')
    annotator.apply_and_annotate()
    # plt.show()

def custom_annotated_two(df, predictor, outcome, order, pval, ax=None, plot_mode = 'box'):
        
    stars = pval_stars(pval)
    
    x = predictor
    y = outcome

    order = order
    formatted_pvalues = [f"{stars}"]
    if plot_mode == 'box':
        ax = sns.boxplot(data=df, x=x, y=y, order=order, ax=ax)
    elif plot_mode == 'violin':
        ax = sns.violinplot(data=df, x=x, y=y, order=order, bw = 0.08)
    pairs=[(order[0],order[1])]
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order, verbose = False)
    annotator.configure(test='test', text_format='star', loc='inside')
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    return ax

def custom_annotated_ngroups(df, predictor, outcome, post_hoc, order, ax=None, plot_mode = 'box'):
        
    pvalues = list(post_hoc['p-corr'])

    x = predictor
    y = outcome

    order = order
    pairs = [tuple(post_hoc.loc[i,['A','B']]) for i in range(post_hoc.shape[0])]
    formatted_pvalues = [f"{pval_stars(pval)}" for pval in pvalues]
    if plot_mode == 'box':
        ax = sns.boxplot(data=df, x=x, y=y, order=order, ax=ax)
    elif plot_mode == 'violin':
        ax = sns.violinplot(data=df, x=x, y=y, order=order, bw= 0.08)
    
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order, verbose = False)
    annotator.configure(test='test', text_format='star', loc='inside')
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    return ax

             
def pval_stars(pval):
    if pval < 0.05 and pval >= 0.01:
        stars = '*'
    elif pval < 0.01 and pval >= 0.001:
        stars = '**'
    elif pval < 0.001 and pval >= 0.0001:
        stars = '***'
    elif pval < 0.0001:
        stars = '****'
    elif pval >= 0.05:
        stars = 'ns'
    return stars


def transform_data(df, outcome):
    df_transfo = df.copy()
    df_transfo[outcome] = np.log(df[outcome])
    return df_transfo  




def auto_stats(df, predictor, outcome, ax=None, subject=None, design='within', mode = 'box', transform=False, verbose=True, order = None):
    
    """
    Automatically compute statistical tests chosen based on normality & homoscedasticity of data and plot it

    ------------
    Inputs =
    - df : tidy dataframe
    - predictor : str or list of str of the column name of the predictive variable (if list --> N-way anova)
    - outcome : column name of the outcome/target/dependent variable
    - ax : ax on which plotting the subplot, created if None (default = None)
    - subject : column name of the subject variable = the within factor variable
    - design : 'within' or 'between' for repeated or independent stats , respectively
    - mode : 'box' or 'violin' for mode of plotting
    - transform : log transform data if True and if data are non-normally distributed & heteroscedastic , to try to do a parametric test after transformation (default = False)
    - verbose : print idea of successfull or unsucessfull transformation of data, if transformed, acccording to non-parametric to parametric test feasable after transformation (default = True)
    - order : order of xlabels (= of groups) if the plot, default = None = default order
    
    Output = 
    - ax : subplot
    
    """

    if ax is None:
        fig, ax = plt.subplots()
    
    if isinstance(predictor, str):
        N = df[predictor].value_counts()[0]
        groups = list(df[predictor].unique())
        ngroups = len(groups)
        
        parametricity_pre_transfo = parametric(df, predictor, outcome, subject)
        
        if transform:
            if not parametricity_pre_transfo:
                df = transform_data(df, outcome)
                parametricity_post_transfo = parametric(df, predictor, outcome, subject)
                parametricity = parametricity_post_transfo
                if verbose:
                    if parametricity_post_transfo:
                        print('Successfull transformation')
                    else:
                        print('Un-successfull transformation')
            else:
                parametricity = parametricity_pre_transfo
        else:
            parametricity = parametricity_pre_transfo
        
        tests = guidelines(df, predictor, outcome, design, parametricity)
        
        pre_test = tests['pre']
        post_test = tests['post']
        results = pg_compute_pre(df, predictor, outcome, pre_test, subject)
        pval = round(results['p'], 4)
        
        if not results['es'] is None:
            es = round(results['es'], 3)
        else:
            es = results['es']
        es_label = results['es_label']
        es_inter = results['es_interp']
        
        if order is None:
            order = list(df[predictor].unique())
        else:
            order = order

        estimators = pd.concat([df.groupby(predictor).mean(numeric_only = True)[outcome].reset_index(), df.groupby(predictor).std(numeric_only = True)[outcome].reset_index()[outcome].rename('sd')], axis = 1).round(2).set_index(predictor)
        cis = [f'[{round(confidence_interval(x)[0],3)};{round(confidence_interval(x)[1],3)}]' for x in [df[df[predictor] == group][outcome] for group in groups]]
        ticks_estimators = [f"{cond} \n {estimators.loc[cond,outcome]} ({estimators.loc[cond,'sd']}) \n {ci} " for cond, ci in zip(order,cis)]

        if mode == 'box':
            if not post_test is None:
                post_hoc = pg_compute_post_hoc(df, predictor, outcome, post_test, subject)
                ax = custom_annotated_ngroups(df, predictor, outcome, post_hoc, order, ax=ax)
            else:
                ax = custom_annotated_two(df, predictor, outcome, order, pval, ax=ax)
            ax.set_xticks(range(ngroups))
            ax.set_xticklabels(ticks_estimators)
            
        elif mode == 'distribution':
            # ax = sns.histplot(df, x=outcome, hue = predictor, kde = True, ax=ax)
            ax = sns.kdeplot(data=df, x=outcome, hue = predictor, ax=ax, bw_adjust = 0.6)
        
        if design == 'between':
            if es_label is None:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n N = {N} values/group * {ngroups} groups \n {pre_test} : p-{pval}')
            else:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n N = {N} values/group * {ngroups} groups \n {pre_test} : p-{pval}, {es_label} : {es} ({es_inter})')
        elif design == 'within':
            n_subjects = df[subject].unique().size
            if es_label is None:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n N = {n_subjects} subjects * {ngroups} groups (*{int(N/n_subjects)} trial/group) \n {pre_test} : p-{pval}')
            else:
                ax.set_title(f'Effect of {predictor} on {outcome} : {pval_stars(pval)} \n  N = {n_subjects} subjects * {ngroups} groups (*{int(N/n_subjects)} trial/group) \n {pre_test} : p-{pval}, {es_label} : {es} ({es_inter})')

    
    elif isinstance(predictor, list):
        
        if design == 'within':
            test_type = 'rm_anova'
            test = pg.rm_anova(data=df, dv=outcome, within = predictor, subject = subject, effsize = 'np2').set_index('Source').round(3)
            pval = test.loc[f'{predictor[0]} * {predictor[1]}','p-GG-corr']
            pstars = pval_stars(pval)
            es_label = test.columns[-2]
            es = test.loc[f'{predictor[0]} * {predictor[1]}','np2']
            es_inter = es_interpretation(es_label=es_label, es_value=es)
            ppred_0 = test.loc[f'{predictor[0]}', 'p-GG-corr']
            ppred_1 = test.loc[f'{predictor[1]}', 'p-GG-corr']
            
        elif design == 'between':
            test_type = 'anova'
            test = pg.anova(data=df, dv=outcome, between = predictor).set_index('Source').round(3)
            pval = test.loc[f'{predictor[0]} * {predictor[1]}','p-unc']
            pstars = pval_stars(pval)
            es_label = test.columns[-1]
            es = test.loc[f'{predictor[0]} * {predictor[1]}','np2']
            es_inter = es_interpretation(es_label=es_label, es_value=es)
            ppred_0 = test.loc[f'{predictor[0]}', 'p-unc']
            ppred_1 = test.loc[f'{predictor[1]}', 'p-unc']
            
        if len(df[predictor[0]]) >= len(df[predictor[1]]):
            x = predictor[0]
            hue = predictor[1]
        else:
            x = predictor[1]
            hue = predictor[0]
        
        sns.pointplot(data = df , x = x, y = outcome, hue = hue, ax=ax)
        title = f'Effect of {predictor[0]} * {predictor[1]} on {outcome} : {pstars} \n {test_type} : pcorr-{pval}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}-{ppred_0} , p-{predictor[1]}-{ppred_1}'
        ax.set_title(title)
        
    return ax


def virer_outliers(df, predictor, outcome, deviations = 5):
    
    groups = list(df[predictor].unique())
    
    group1 = df[df[predictor] == groups[0]][outcome]
    group2 = df[df[predictor] == groups[1]][outcome]
    
    outliers_trop_hauts_g1 = group1[(group1 > group1.std() * deviations) ]
    outliers_trop_bas_g1 = group1[(group1 < group1.std() * -deviations) ]
    
    outliers_trop_hauts_g2 = group2[(group2 > group1.std() * deviations) ]
    outliers_trop_bas_g2 = group2[(group2 < group1.std() * -deviations) ]
    
    len_h_g1 = outliers_trop_hauts_g1.size
    len_b_g1 = outliers_trop_bas_g1.size
    len_h_g2 = outliers_trop_hauts_g2.size
    len_b_g2 = outliers_trop_bas_g2.size
    
    return len_b_g2


def outlier_exploration(df, predictor, labels, outcome, figsize = (16,8)):
                 
    g1 = df[df[predictor] == labels[0]][outcome]
    g2 = df[df[predictor] == labels[1]][outcome]

    fig, axs = plt.subplots(ncols = 2, figsize = figsize, constrained_layout = True)
    fig.suptitle('Outliers exploration', fontsize = 20)

    ax = axs[0]
    ax.scatter(g1 , g2)
    ax.set_title(f'Raw {labels[0]} vs {labels[1]} scatterplot')
    ax.set_ylabel(f'{outcome} in condition {labels[0]}')
    ax.set_xlabel(f'{outcome} in condition {labels[1]}')

    g1log = np.log(g1)
    g2log = np.log(g2)

    ax = axs[1]
    ax.scatter(g1log, g2log)
    ax.set_title(f'Log-log {labels[0]} vs {labels[1]} scatterplot')
    ax.set_ylabel(f'{outcome} in condition {labels[0]}')
    ax.set_xlabel(f'{outcome} in condition {labels[1]}')

    plt.show()
    
    
def qqplot(df, predictor, outcome, figsize = (10,15)):
    
    labels = list(df[predictor].unique())
    ngroups = len(labels) 
    
    groupe = {}
    
    for label in labels: 
        groupe[label] = {
                         'log' : np.log(df[df[predictor] == label][outcome]), 
                         'inverse' : 1 / (df[df[predictor] == label][outcome]),
                         'none' : df[df[predictor] == label][outcome]
                        }
     
    fig, axs = plt.subplots(nrows = 3, ncols = ngroups, figsize = figsize, constrained_layout = True)
    fig.suptitle(f'QQ-PLOT', fontsize = 20)
    
    for col, label in enumerate(labels): 
        for row, transform in enumerate(['none','log','inverse']):
            ax = axs[row, col]
            ax = pg.qqplot(groupe[label][transform], ax=ax)
            ax.set_title(f'Condition : {label} ; data are {transform} transformed')
        
    plt.show()

def permutation_test_homemade(x,y, design = 'within', n_resamples=999):
    def statistic(x, y):
        return np.mean(x) - np.mean(y)
    if design == 'within':
        permutation_type = 'samples'
    elif design == 'between':
        permutation_type = 'independent'
    res = stats.permutation_test(data=[x,y], statistic=statistic, permutation_type=permutation_type, n_resamples=n_resamples, batch=None, alternative='two-sided', axis=0, random_state=None)
    return res.pvalue

def permutation(df, predictor, outcome , design = 'within' , subject = None, n_resamples=999):
    pairs = list((itertools.combinations(df[predictor].unique(), 2)))
    pvals = []
    for pair in pairs:
        x = df[df[predictor] == pair[0]][outcome].values
        y = df[df[predictor] == pair[1]][outcome].values
        p = permutation_test_homemade(x=x,y=y, design=design, n_resamples=n_resamples)
        pvals.append(p)
    df_return = pd.DataFrame(pairs, columns = ['A','B'])
    df_return['p-unc'] = pvals
    rej , pcorrs = pg.multicomp(pvals, method = 'holm')
    df_return['p-corr'] = pcorrs
    return df_return

def reorder_df(df, colname, order):
    concat = []
    for cond in order:
        concat.append(df[df[colname] == cond])
    return pd.concat(concat)


def lmm(df, predictor, outcome, subject, order=None):

    if isinstance(predictor, str):
        formula = f'{outcome} ~ {predictor}' 
    elif isinstance(predictor, list):
        if len(predictor) == 2:
            formula = f'{outcome} ~ {predictor[0]}*{predictor[1]}' 
        elif len(predictor) == 3:
            formula = f'{outcome} ~ {predictor[0]}*{predictor[1]}*{predictor[2]}' 

    if not order is None:
        df = reorder_df(df, predictor, order)

    if isinstance(predictor, list) :
        order = list(df[predictor[0]].unique())
    else:
        order = list(df[predictor].unique())


    md = smf.mixedlm(formula, data=df, groups=df[subject])
    mdf = md.fit()
    print(mdf.summary())

    pvals = mdf.pvalues.to_frame(name = 'p')
    coefs = mdf.fe_params.to_frame(name = 'coef').round(3)
    dict_pval_stars = {idx.split('.')[1][:-1]:pval_stars(pvals.loc[idx,'p']) for idx in pvals.index if not idx in ['Intercept','Group Var']}
    dict_coefs = {idx.split('.')[1][:-1]:coefs.loc[idx,'coef'] for idx in coefs.index if not idx in ['Intercept','Group Var']}

    fig, ax = plt.subplots()
    if isinstance(predictor, str):
        sns.boxplot(data=df, x = predictor, y = outcome, ax=ax )
    elif isinstance(predictor, list):
        sns.pointplot(data=df, x = predictor[0], y = outcome, hue = predictor[1],ax=ax)
    ax.set_title(formula)
    ticks = []


    for i, cond in enumerate(order):
        print(cond)
        if i == 0:
            tick = cond
        else:
            tick = f"{cond} \n {dict_pval_stars[cond]} \n {dict_coefs[cond]}"
        ticks.append(tick)
    ax.set_xticks(range(df[predictor].unique().size))
    ax.set_xticklabels(ticks)
    plt.show()
    
    return mdf


def confidence_interval(x, confidence = 0.95, verbose = False):
    m = x.mean() 
    s = x.std() 
    dof = x.size-1 
    t_crit = np.abs(stats.t.ppf((1-confidence)/2,dof))
    ci = (m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x))) 
    if verbose:
        print(f'm : {round(m, 3)} , std : {round(s,3)} , ci : [{round(ci[0],3)};{round(ci[1],3)}]')
    return ci
