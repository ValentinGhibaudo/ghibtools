import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import statsmodels.formula.api as smf

def med_mad(data, constant = 1.4826):
    median = np.median(data)
    mad = np.median(np.abs(data - median)) * constant
    return median , mad

def normality(df, predictor, outcome):
    df = df.reset_index(drop=True)

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

def parametric(df, predictor, outcome, design, subject = None):
    df = df.reset_index(drop=True)
    n_groups = df[predictor].unique().size
    normal = normality(df, predictor, outcome)

    if design == 'between': 
        equal_var = homoscedasticity(df, predictor, outcome)
    else: 
        equal_var = sphericity(df, predictor, outcome, subject)
    
    if normal and equal_var:
        parametricity = True
    else:
        parametricity = False
        
    return parametricity


def guidelines(df, predictor, design, parametricity):
        
    n_groups = df[predictor].unique().size
    
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
        
def pval_stars(pval):
    if pval < 0.05 and pval >= 0.01:
        stars = '*'
    elif pval < 0.01 and pval >= 0.001:
        stars = '**'
    elif pval < 0.001 and pval >= 0.0001:
        stars = '***'
    elif pval < 0.0001:
        stars = '****'
    else:
        stars = 'ns'
    return stars


def transform_data(df, outcome):
    df_transfo = df.copy()
    df_transfo[outcome] = np.log(df[outcome])
    return df_transfo  

def readable_pval(pval):
    if pval < 0.01 and pval >= 0.001:
        return ' < 0.01'
    elif pval < 0.001 and pval >= 0.0001:
        return ' < 0.001'
    elif pval < 0.0001:
        return ' < 0.0001'
    else:
        return f' = {round(pval, 3)}'

def auto_stats(df, 
                predictor, 
                outcome, 
                ax=None, 
                subject=None, 
                design='within', 
                transform=False, 
                verbose=True, 
                order = None, 
                with_title = True,
                xtick_info = True,
                return_pval = False,
                outcome_clean_label = None,
                outcome_unit = None,
                strip = True,
                lines = True,
                title_info = 'short',
                multicomp_correction = True,
                fontsize = None,
                force_post_hoc = True
                ):
    
    """
    Automatically compute and plot statistical tests chosen based on normality & homoscedasticity (or sphericity) of data

    ------------
    Inputs =
    - df : tidy dataframe
    - predictor : str or list of str of the column name of the predictive variable (if list --> N-way anova)
    - outcome : column name of the outcome/target/dependent variable
    - ax : ax on which plotting the subplot, created if None (default = None)
    - subject : column name of the subject variable = the within factor variable
    - design : 'within' or 'between' for repeated or independent stats , respectively
    - transform : log transform data if True and if data are non-normally distributed & heteroscedastic , to try to do a parametric test after transformation (default = False)
    - verbose : print idea of successfull or unsuccessfull transformation of data, if transformed, acccording to non-parametric to parametric test feasable after transformation (default = True)
    - order : order of xlabels (= of groups) if the plot, default = None = default order
    - with_title : return ax with title if True(default = True)
    - xtick_info : return ax with descriptive statistics under xtick labels if True (default = True)
    - return_pval : return pval with ax if True (default = False)
    - outcome_clean_label : string clean label name of the outcome to verbose title and ytick
    - outcome_unit : unit of the outcome to verbose ytick
    - strip : Draw one dot per subject (default = True)
    - lines : Draw one line per subject (default = True)
    - title_info : could be 'short' or 'full' according to the verbosity of the title (default = 'short')
    - multicomp_correction : if two or more predictors, some detailed stats will be displayed and pvalue corrected if True (default = True)
    - fontsize : set fontsizes of titles (fontsize * 1), ylabel (fontsize * 0.9), xlabel (fontsize * 0.9), xticklabels (fontsize * 0.75), legend (fontsize * 0.6), if not None. Default is None = default matplotlib params
    - force_post_hoc : Force the display of post-hoc stats even if global test not significant
    Output = 
    - ax : subplot
    
    """

    title_fontsize = fontsize
    ylabel_fontsize = int(fontsize * 0.9) if not fontsize is None else fontsize
    xlabel_fontsize = ylabel_fontsize
    xticklabels_fontsize = int(fontsize * 0.75) if not fontsize is None else fontsize
    legend_fontsize = int(fontsize * 0.6) if not fontsize is None else fontsize
    fontsize_stars = 10 # fontsize of pvalue stars


    if ax is None:
        fig, ax = plt.subplots()
    
    if outcome_clean_label is None:
        outcome_clean_label = outcome

    if not order is None:
        if type(predictor) is str:
            df = reorder_df(df=df, colname=predictor, order=order).reset_index(drop = True)
        elif type(predictor) is list:
            df = reorder_df(df=df, colname=predictor[0], order=order).reset_index(drop = True)
    else:
        if type(predictor) is str:
            order = list(df[predictor].unique())
        elif type(predictor) is list:
            order = list(df[predictor[0]].unique())

    # print(df)

    if type(predictor) is str: # one way
        N = df[predictor].value_counts()[0]
        groups = list(df[predictor].unique())
        ngroups = len(groups)
        
        # print(df, predictor, outcome, subject)
        parametricity_pre_transfo = parametric(df, predictor, outcome, design, subject)
        
        if transform:
            if not parametricity_pre_transfo:
                df = transform_data(df, outcome)
                parametricity_post_transfo = parametric(df, predictor, outcome, design,  subject)
                parametricity = parametricity_post_transfo
                if verbose:
                    if parametricity_post_transfo:
                        print('Useful log transformation')
                    else:
                        print('Useless log transformation')
            else:
                parametricity = parametricity_pre_transfo
        else:
            parametricity = parametricity_pre_transfo
        
        tests = guidelines(df, predictor, design, parametricity)
        
        pre_test = tests['pre']
        post_test = tests['post']
        results = pg_compute_pre(df, predictor, outcome, pre_test, subject)
        pval = results['p']
        readable_p = readable_pval(pval)
        
        if not results['es'] is None:
            es = round(results['es'], 3)
        else:
            es = results['es']
        es_label = results['es_label']
        es_inter = results['es_interp']

        sd = df[outcome].std()
        min_val = df[outcome].min()
        max_val = df[outcome].max()

        ax = sns.boxplot(data = df, x = predictor, y = outcome, order = order, ax=ax, whis = 5) # construct basic ax without annotation

        if pval < 0.05 or force_post_hoc:
            if not post_test is None: # loop over pairwise combinations to plot annotations
                post_hoc = pg_compute_post_hoc(df, predictor, outcome, post_test, subject)
                tick_dict = {order[i]:i for i in range(len(order))}
                df_annot = post_hoc.copy()
                df_annot['star'] = None
                df_annot['xstart'] = None
                df_annot['xstop'] = None
                df_annot['dx'] = None
                df_annot['y'] = None

                for i, row in df_annot.iterrows():
                    df_annot.loc[i, 'star'] = pval_stars(row['p-corr'])
                    df_annot.loc[i, 'xstart'] = tick_dict[row['A']]
                    df_annot.loc[i, 'xstop'] = tick_dict[row['B']]
                    df_annot.loc[i,'y'] = max_val + i * (sd / 1.8 ) + sd / 3
                df_annot.loc[:,'dx'] = df_annot['xstop'] - df_annot['xstart']

                for i, row in df_annot.iterrows():
                    # ax.arrow(x = row['xstart'], y = row['y'], dx = row['dx'], dy = 0, length_includes_head = True) # horizontal bar    
                    # ax.arrow(x = row['xstart'], y = row['y'], dx = 0, dy =  - sd / 10) # small left vertical bar
                    # ax.arrow(x = row['xstop'], y = row['y'], dx = 0, dy = - sd / 10) # small right vertical bar

                    if row['p-corr'] < 0.05:
                        ls = 'solid'
                    else:
                        ls = 'dotted'

                    x = [row['xstart'], row['xstart'] + row['dx']]
                    y = [row['y']] * len(x)
                    ax.plot(x,y , color = 'k', ls = ls) # horizontal bar  

                    x = [row['xstart'], row['xstart']]
                    y = [row['y'], row['y'] - sd / 15]
                    ax.plot(x,y , color = 'k') # small left vertical bar

                    x = [row['xstop'], row['xstop']]
                    y = [row['y'], row['y'] - sd / 15]
                    ax.plot(x,y , color = 'k') # small right vertical bar
                    
                    ax.text(x = (row['xstart'] + row['xstop']) / 2 , y = row['y'] + sd / 10, s = row['star'], fontsize = fontsize_stars , horizontalalignment='center')
                y_max_arrow = df_annot['y'].max()
            else: # just read main test results to annotate
                y = max_val + sd
                star = pval_stars(results['p'])

                # ax.arrow(x = 0, y = y, dx = 1, dy = 0, length_includes_head = True) # horizontal bar    
                # ax.arrow(x = 0, y = y, dx = 0, dy = - sd / 10) # small left vertical bar
                # ax.arrow(x = 1, y = y, dx = 0, dy = - sd / 10) # small right vertical bar

                if results['p'] < 0.05:
                    ls = 'solid'
                else:
                    ls = 'dotted'

                x = [0, 1]
                y_plot = [y] * len(x)
                ax.plot(x,y_plot , color = 'k', ls = ls) # horizontal bar  

                x = [0, 0]
                y_plot = [y, y - sd / 10]
                ax.plot(x,y_plot , color = 'k') # small left vertical bar

                x = [1, 1]
                y_plot = [y,y- sd / 10]
                ax.plot(x,y_plot , color = 'k') # small right vertical bar

                
                ax.text(x = 0.5 , y = y + sd / 10, s = star, fontsize = fontsize_stars, horizontalalignment='center')
                y_max_arrow = y.copy()
            
            ax.set_ylim(min_val - sd, y_max_arrow + sd)
                
        if xtick_info:
            ax.set_xticks(range(ngroups))

            cis = [f'[{round(confidence_interval(x)[0],2)};{round(confidence_interval(x)[1],2)}]' for x in [df[df[predictor] == group][outcome] for group in groups]]

            if parametricity:
                estimators = pd.concat([df.groupby(predictor).mean(numeric_only = True)[outcome].reset_index(), df.groupby(predictor).std(numeric_only = True)[outcome].reset_index()[outcome].rename('sd')], axis = 1).round(2).set_index(predictor)
                ticks_estimators = [f"{cond} \n {estimators.loc[cond,outcome]} ({estimators.loc[cond,'sd']}) \n {ci} " for cond, ci in zip(order,cis)]
                
            else:
                ticks_estimators = []
                for cond, ci in zip(order, cis): 
                    med, mad = med_mad(df[df[predictor] == cond][outcome])
                    med, mad = round(med,2) , round(mad, 2)
                    ticks_estimator_cond = f"{cond} \n {med} ({mad}) \n {ci} "
                    ticks_estimators.append(ticks_estimator_cond)

            ax.set_xticklabels(ticks_estimators, fontsize = xticklabels_fontsize)

            
        if title_info == 'full':
            if design == 'between':
                if es_label is None:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n  {pre_test} : p{readable_p}', fontsize = title_fontsize)
                else:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n {pre_test} : p{readable_p}, {es_label} : {es} ({es_inter})', fontsize = title_fontsize)
            elif design == 'within':
                n_subjects = df[subject].unique().size
                if es_label is None:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n N = {n_subjects} subjects * {ngroups} groups (*{int(N/n_subjects)} trial/group) \n {pre_test} : p{readable_p}', fontsize = title_fontsize)
                else:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n  N = {n_subjects} subjects * {ngroups} groups (*{int(N/n_subjects)} trial/group) \n {pre_test} : p{readable_p}, {es_label} : {es} ({es_inter})', fontsize = title_fontsize)
        elif title_info == 'short':
            if design == 'between':
                if es_label is None:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n {pre_test} : p{readable_p}', fontsize = title_fontsize)
                else:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n {pre_test} : p{readable_p}, {es_label} : {es} ({es_inter})', fontsize = title_fontsize)
            elif design == 'within':
                n_subjects = df[subject].unique().size
                if es_label is None:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n N = {n_subjects} subjects \n {pre_test} : p{readable_p}', fontsize = title_fontsize)
                else:
                    ax.set_title(f'Effect of {predictor} on {outcome_clean_label} : {pval_stars(pval)} \n  N = {n_subjects} subjects \n {pre_test} : p{readable_p}, {es_label} : {es} ({es_inter})', fontsize = title_fontsize)

        if strip is True:
            sns.stripplot(x=predictor, y=outcome, data=df, order=order , color = 'k', alpha = 0.5, size = 3, jitter = 0.05, ax=ax)

        if lines is True and design == 'within':
            palette = ['k'] * n_subjects
            sns.lineplot(x=predictor, y=outcome, data=df, hue = subject, alpha = 0.2, legend = False, errorbar = None, palette = palette, ax=ax)


    elif type(predictor) is list: # n way anova
        
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
            x_predictor = predictor[0]
            hue_predictor = predictor[1]
            ppred_hue = ppred_1.copy()
        else:
            x_predictor = predictor[1]
            hue_predictor = predictor[0]
            ppred_hue = ppred_0.copy()

        readable_p = readable_pval(pval)
        sns.pointplot(data = df , 
                      x = x_predictor, 
                      y = outcome, 
                      hue = hue_predictor, 
                      ax=ax, 
                      order=order, 
                      errorbar= 'sd',
                      errwidth=1.5, 
                      capsize=0.05,
                      )
        title = f'Interaction {predictor[0]} * {predictor[1]} on {outcome_clean_label} : {pstars} \n {test_type} : pcorr {readable_p}, {es_label} : {es} ({es_inter}) \n p-{predictor[0]}{readable_pval(ppred_0)} , p-{predictor[1]}{readable_pval(ppred_1)}'
        ax.set_title(title, fontsize = title_fontsize)

        if multicomp_correction:
            multiple_comparison_correction = df[x_predictor].unique().size + df[hue_predictor].unique().size # n tests that will be done to get detailed stats intra dataset
        else:
            multiple_comparison_correction = 1

        xticklabels = []
        for i, level in enumerate(df[x_predictor].unique()):
            df_level = df[df[x_predictor] == level]
            parametricity = parametric(df_level, hue_predictor, outcome, design, subject)
            tests = guidelines(df_level, hue_predictor, design, parametricity)
            pre_test = tests['pre']
            res = pg_compute_pre(df = df_level, predictor = hue_predictor, outcome = outcome, subject=subject, test=pre_test)
            pval = res['p'] * multiple_comparison_correction
            star = pval_stars(pval)
            xticklabels.append(f'{level}\n{hue_predictor} : {star}')

        ax.set_xticklabels(xticklabels, fontsize = xticklabels_fontsize)

        legendlabels = []
        for i, level in enumerate(df[hue_predictor].unique()):
            df_level = df[df[hue_predictor] == level]
            parametricity = parametric(df_level, x_predictor, outcome, design, subject)
            tests = guidelines(df_level, x_predictor, design, parametricity)
            pre_test = tests['pre']
            res = pg_compute_pre(df = df_level, predictor = x_predictor, outcome = outcome, subject=subject, test=pre_test)
            pval = res['p'] * multiple_comparison_correction
            star = pval_stars(pval)
            legendlabels.append(f'{level} - {x_predictor} : {star}')

        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles, legendlabels, fontsize = legend_fontsize)


    ax.set_xlabel(ax.get_xlabel(), fontsize = xlabel_fontsize)
    
    if not with_title:
        ax.set_title(None)

    if outcome_unit is not None:
        ax.set_ylabel(f'{outcome_clean_label} [{outcome_unit}]', fontsize = ylabel_fontsize)
    else:
        ax.set_ylabel(outcome_clean_label, fontsize = ylabel_fontsize)

    if not return_pval:
        return ax
    else:
        return ax, pval


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
    N = x.size
    m = x.mean() 
    s = x.std() 
    dof = N-1 
    t_crit = np.abs(stats.t.ppf((1-confidence)/2,dof))
    ci = (m-s*t_crit/np.sqrt(N), m+s*t_crit/np.sqrt(N)) 
    if verbose:
        print(f'm : {round(m, 3)} , std : {round(s,3)} , ci : [{round(ci[0],3)};{round(ci[1],3)}]')
    return ci


def stats_quantitative(df, xlabel, ylabel, ax=None, corr_method = 'spearman'):
    if ax is None:
        fig, ax = plt.subplots()

    x = df[xlabel]
    y = df[ylabel]

    if corr_method == 'pearson':
        res_corr = stats.pearsonr(x, y)
        r = res_corr.statistic
    elif corr_method == 'spearman':
        res_corr = stats.spearmanr(x, y)
        r = res_corr.correlation
    pval_corr = res_corr.pvalue
    stars_corr = pval_stars(pval_corr)

    res_reg = stats.linregress(x, y)
    intercept = res_reg.intercept
    slope = res_reg.slope
    rsquare = res_reg.rvalue **2
    pval_reg = res_reg.pvalue
    stars_reg = pval_stars(pval_reg)
    
    ax.plot(x, intercept + slope*x, 'r', label=f'f(x) = {round(slope, 2)}x + {round(intercept, 2)}')
    ax.scatter(x = x, y=y, alpha = 0.8)

    ax.set_title(f'Correlation ({corr_method}) : {round(r, 3)}, p : {stars_corr}\nR² : {round(rsquare, 3)}, p : {stars_reg}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.legend()

    return ax

def get_descriptive_stats(df, predictor, outcome):
    groups = df[predictor].unique()
    descriptive_stats = pd.DataFrame(columns = groups, index = ['N','mean','sd','sem','median','mad','CI95'])
    for group in groups:
        df_group = df[df[predictor] == group]
        x = df_group[outcome].values
        descriptive_stats.loc['N',group] = x.size
        descriptive_stats.loc['mean',group] = np.mean(x)
        descriptive_stats.loc['sd',group] = np.std(x)
        descriptive_stats.loc['sem',group] = np.std(x) / np.sqrt(x.size)
        med, mad = med_mad(x)
        descriptive_stats.loc['median',group] = med
        descriptive_stats.loc['mad',group] = mad
        descriptive_stats.loc['CI95',group] = list(pd.Series(confidence_interval(x)).round(3))
    return descriptive_stats.T

def auto_stats_summary(df, predictor, outcome, design, subject=None):
    
    is_parametric = parametric(df = df, predictor = predictor, outcome = outcome,  design=design, subject = subject)
    tests = guidelines(df = df, predictor = predictor, design = design, parametricity = is_parametric)
    test = tests['pre']
    
    
    pval_labels = {'t-test_ind':'p-val','t-test_paired':'p-val','anova':'p-unc','rm_anova':'p-unc','Mann-Whitney':'p-val','Wilcoxon':'p-val', 'Kruskal':'p-unc', 'friedman':'p-unc'}
    esize_labels = {'t-test_ind':'cohen-d','t-test_paired':'cohen-d','anova':'np2','rm_anova':'np2','Mann-Whitney':'CLES','Wilcoxon':'CLES', 'Kruskal':None, 'friedman':None}
    
    if test == 't-test_ind':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        inferential_stats = pg.ttest(pre, post, paired=False)
        
    elif test == 't-test_paired':
        groups = list(set(df[predictor]))
        pre = df[df[predictor] == groups[0]][outcome]
        post = df[df[predictor] == groups[1]][outcome]
        inferential_stats = pg.ttest(pre, post, paired=True)
        
    elif test == 'anova':
        inferential_stats = pg.anova(dv=outcome, between=predictor, data=df, detailed=False, effsize = 'np2')
    
    elif test == 'rm_anova':
        inferential_stats = pg.rm_anova(dv=outcome, within=predictor, data=df, detailed=False, effsize = 'np2', subject = subject)
        
    elif test == 'Mann-Whitney':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        inferential_stats = pg.mwu(x, y)
        
    elif test == 'Wilcoxon':
        groups = list(set(df[predictor]))
        x = df[df[predictor] == groups[0]][outcome]
        y = df[df[predictor] == groups[1]][outcome]
        inferential_stats = pg.wilcoxon(x, y)
        
    elif test == 'Kruskal':
        inferential_stats = pg.kruskal(data=df, dv=outcome, between=predictor)
        
    elif test == 'friedman':
        inferential_stats = pg.friedman(data=df, dv=outcome, within=predictor, subject=subject)
    
    es_label = esize_labels[test]
    if es_label is None:
        es = None
    else:
        es = inferential_stats[es_label].values[0]
        
    es_interp = es_interpretation(es_label, es)
    # inferential_stats[f'ES : {es_label}'] = es
    inferential_stats['ES interpretation'] = es_interp
    inferential_stats = inferential_stats.rename(index = {inferential_stats.index[0]:test})
    
    descriptive_stats = get_descriptive_stats(df = df, predictor = predictor, outcome = outcome)
    
    return {'descriptive_stats':descriptive_stats, 'inferential_stats':inferential_stats}

def save_auto_stats_summary(stats_dict, path):
    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    for k,v in stats_dict.items():
        v.to_excel(writer, sheet_name = k)
    writer.close()

def stats_quali_quali(data, predictor, outcome, show = True, save = None):
    counts = data[predictor].value_counts(ascending = True)
    
    expected, observed, stats = pg.chi2_independence(data, x=predictor, y=outcome)
    observed = pd.crosstab(index =data[predictor] , columns = data[outcome])
    p = stats['pval'].mean()

    if show:
        fig, axs = plt.subplots(ncols = 2, figsize = (12,4), sharey = True)
        suptitle = f'Effect of {predictor} on {outcome} : p = {round(p, 5)}'
        fig.suptitle(suptitle, fontsize = 15, y = 1.05)
        fig.subplots_adjust(wspace = 0)
        
        ax = axs[0]
        observed.round(2).plot.bar(ax=ax, edgecolor = 'k')
        ax.set_title(f'Observed')
        xticklabels = []
        for level in observed.index:
            N = int(observed.sum(axis = 1).loc[level])
            xticklabel = f'{level}\nN={N}'
            xticklabels.append(xticklabel)
        ax.set_xticklabels(xticklabels, rotation = 0)
        ax.set_xlabel(f'{predictor}\nN={data[predictor].notna().sum()}')
    
        legendlabels = []
        for level in observed.columns:
            N = int(observed.sum(axis = 0).loc[level])
            legendlabel = f'{level}\nN={N}'
            legendlabels.append(legendlabel)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, legendlabels, title = outcome)
    
        
        ax = axs[1]
        expected.round(2).plot.bar(ax=ax, edgecolor = 'k', alpha = 0.6)
        ax.set_title('Expected')
        
        for ax, df in zip(axs, [observed, expected]):
            for bar in ax.containers:
                ax.bar_label(bar)

        if not save is None:
            fig.savefig(save, dpi = 500, bbox_inches = 'tight')
    
        plt.show()

    return p