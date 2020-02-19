import pandas as pd
import numpy as np
from numpy.random import normal, gamma, beta, binomial, dirichlet, multinomial
import bisect
import math


def readHierarchicalDf(fileloc):

    df = pd.read_csv(fileloc, header=None)
    dfCols = df.iloc[:2, ].T
    df = df.iloc[2:, ]
    levels = [list(set(dfCols.iloc[:, 0])), list(set(dfCols.iloc[:, 1]))]
    nFeat = dfCols.shape[0]

    rowLabels = [[-1 for x in range(nFeat)] for x in range(2)]
    for k in range(2):
        for i, level in enumerate(levels[k]):
            indices = dfCols[dfCols[k] == level].index.tolist()
            for ind in indices:
                rowLabels[k][ind] = i

    midx = pd.MultiIndex(levels=levels, codes=rowLabels)

    df = pd.DataFrame(df.values, columns=midx)
    return df


def createAugmentedProductsDf(df, dictCatAttributes):

    categoricalAttr = list(dictCatAttributes.keys())
    setAttributes = [list(dictCatAttributes[x]) for x in categoricalAttr]

    numericalAttr = list(set(df.columns) - set(categoricalAttr))
    numInd = sorted([list(df.columns).index(x) for x in numericalAttr])
    numerical = list(np.array(df.columns)[numInd])

    catInd = sorted([list(df.columns).index(x) for x in categoricalAttr])
    categorical = [item for sublist in setAttributes for item in sublist]  # flatten the categorical List
    # get the indices for the top-level categories
    topCategorical = [[x + len(numerical) for i in range(len(dictCatAttributes[categoricalAttr[x]]))]
                      for x in range(len(categoricalAttr))]
    topCategorical = list(range(len(numerical))) + [item for sublist in topCategorical for item in sublist]

    indices = pd.Series(numerical + categorical)
    topIndices = pd.Series(numerical + categoricalAttr)

    # performs one-hot encoding for the categorical features whilst leaving other features unchanged
    def calculateFeatures(row):

        catList = [list((pd.Series(setAttributes[i]) == row.iat[x]).astype(int)) for i, x in enumerate(catInd)]
        catList = [item for sublist in catList for item in sublist]  # flatten the categorical List
        val = pd.Series(np.array(list(row[numInd]) + catList), index=indices)

        return val

    augmentedDf = df.apply(calculateFeatures, axis=1)
    midx = pd.MultiIndex(levels=[list(topIndices), list(indices)],
                         codes=[topCategorical, list(range(len(topCategorical)))])
    augmentedDf = pd.DataFrame(augmentedDf.values, columns=midx)

    return augmentedDf


def simulateUtility(N, productShare, dictNumAttributes, dictCatAttributes):

    # simulate the utility scores across all attributes across all generations
    mcost_k_M = 100 * beta(a=4.5, b=2, size=N[0])
    mcost_t_M = 0.1 * beta(a=3.5, b=2, size=N[0])
    mcost_utility_M = ((-2e-3 * gamma(mcost_k_M, mcost_t_M, [len(dictNumAttributes['monthly_cost']), N[0]]))
                       * np.array(dictNumAttributes['monthly_cost'])[:, np.newaxis])
    mcost_utility_M -= mcost_utility_M.mean(axis=0)

    mcost_k_X = 100 * beta(a=4., b=2, size=N[1])
    mcost_t_X = 0.1 * beta(a=4.5, b=2, size=N[1])
    mcost_utility_X = ((-1.6e-3 * gamma(mcost_k_X, mcost_t_X, [len(dictNumAttributes['monthly_cost']), N[1]]))
                       * np.array(dictNumAttributes['monthly_cost'])[:, np.newaxis])
    mcost_utility_X -= mcost_utility_X.mean(axis=0)

    mcost_k_B = 100 * beta(a=3.5, b=2, size=N[2])
    mcost_t_B = 0.1 * beta(a=5, b=2, size=N[2])
    mcost_utility_B = ((-1.8e-3 * gamma(mcost_k_B, mcost_t_B, [len(dictNumAttributes['monthly_cost']), N[2]]))
                       * np.array(dictNumAttributes['monthly_cost'])[:, np.newaxis])
    mcost_utility_B -= mcost_utility_B.mean(axis=0)

    mcost_utility = np.hstack([mcost_utility_M, mcost_utility_X, mcost_utility_B]).T

    upcost_k_M = 100 * beta(a=4.5, b=2, size=N[0])
    upcost_t_M = 0.1 * beta(a=3.5, b=2, size=N[0])
    upcost_utility_M = ((-2e-5 * gamma(upcost_k_M, upcost_t_M, [len(dictNumAttributes['upfront_cost']), N[0]]))
                       * np.array(dictNumAttributes['upfront_cost'])[:, np.newaxis])
    upcost_utility_M -= upcost_utility_M.mean(axis=0)

    upcost_k_X = 100 * beta(a=4., b=2, size=N[1])
    upcost_t_X = 0.1 * beta(a=4.5, b=2, size=N[1])
    upcost_utility_X = ((-1e-5 * gamma(upcost_k_X, upcost_t_X, [len(dictNumAttributes['upfront_cost']), N[1]]))
                       * np.array(dictNumAttributes['upfront_cost'])[:, np.newaxis])
    upcost_utility_X -= upcost_utility_X.mean(axis=0)

    upcost_k_B = 100 * beta(a=3.5, b=2, size=N[2])
    upcost_t_B = 0.1 * beta(a=5, b=2, size=N[2])
    upcost_utility_B = ((-1.5e-5 * gamma(upcost_k_B, upcost_t_B, [len(dictNumAttributes['upfront_cost']), N[2]]))
                       * np.array(dictNumAttributes['upfront_cost'])[:, np.newaxis])
    upcost_utility_B -= upcost_utility_B.mean(axis=0)

    upcost_utility = np.hstack([upcost_utility_M, upcost_utility_X, upcost_utility_B]).T

    term_k_M = 100 * beta(a=4.5, b=2, size=N[0])
    term_t_M = 0.1 * beta(a=3.5, b=2, size=N[0])
    term_utility_M = ((-1e-2 * gamma(term_k_M, term_t_M, [len(dictNumAttributes['term']), N[0]]))
                       * np.array(dictNumAttributes['term'])[:, np.newaxis])
    term_utility_M -= term_utility_M.mean(axis=0)

    term_k_X = 100 * beta(a=4., b=2, size=N[1])
    term_t_X = 0.1 * beta(a=4.5, b=2, size=N[1])
    term_utility_X = ((-1.2e-2 * gamma(term_k_X, term_t_X, [len(dictNumAttributes['term']), N[1]]))
                       * np.array(dictNumAttributes['term'])[:, np.newaxis])
    term_utility_X -= term_utility_X.mean(axis=0)

    term_k_B = 100 * beta(a=3.5, b=2, size=N[2])
    term_t_B = 0.1 * beta(a=5, b=2, size=N[2])
    term_utility_B = ((-1.2e-2 * gamma(term_k_B, term_t_B, [len(dictNumAttributes['term']), N[2]]))
                       * np.array(dictNumAttributes['term'])[:, np.newaxis])
    term_utility_B -= term_utility_B.mean(axis=0)

    term_utility = np.hstack([term_utility_M, term_utility_X, term_utility_B]).T


    worth_k_M = 1000 * beta(a=4.5, b=2, size=N[0])
    worth_t_M = 0.01 * beta(a=3.5, b=2, size=N[0])
    worth_utility_M = ((1.5e-5 * gamma(worth_k_M, worth_t_M, [len(dictNumAttributes['vehicle_worth']), N[0]]))
                     * np.array(dictNumAttributes['vehicle_worth'])[:, np.newaxis])
    worth_utility_M -= worth_utility_M.mean(axis=0)

    worth_k_X = 1000 * beta(a=4, b=2, size=N[1])
    worth_t_X = 0.01 * beta(a=4.5, b=2, size=N[1])
    worth_utility_X = ((1.5e-5 * gamma(worth_k_X, worth_t_X, [len(dictNumAttributes['vehicle_worth']), N[1]]))
                       * np.array(dictNumAttributes['vehicle_worth'])[:, np.newaxis])
    worth_utility_X -= worth_utility_X.mean(axis=0)

    worth_k_B = 1000 * beta(a=3.5, b=2, size=N[2])
    worth_t_B = 0.01 * beta(a=4, b=2, size=N[2])
    worth_utility_B = ((1.5e-5 * gamma(worth_k_B, worth_t_B, [len(dictNumAttributes['vehicle_worth']), N[2]]))
                       * np.array(dictNumAttributes['vehicle_worth'])[:, np.newaxis])
    worth_utility_B -= worth_utility_B.mean(axis=0)

    worth_utility = np.hstack([worth_utility_M, worth_utility_X, worth_utility_B]).T

    range_k_M = 1000 * beta(a=4.5, b=2, size=N[0])
    range_t_M = 0.007 * beta(a=4, b=2, size=N[0])
    range_utility_M = ((2e-3 * gamma(range_k_M, range_t_M, [len(dictNumAttributes['range']), N[0]]))
                       * np.array(dictNumAttributes['range'])[:, np.newaxis])
    range_utility_M -= range_utility_M.mean(axis=0)

    range_k_X = 1000 * beta(a=4, b=2, size=N[1])
    range_t_X = 0.008 * beta(a=4, b=2, size=N[1])
    range_utility_X = ((2e-3 * gamma(range_k_X, range_t_X, [len(dictNumAttributes['range']), N[1]]))
                       * np.array(dictNumAttributes['range'])[:, np.newaxis])
    range_utility_X -= range_utility_X.mean(axis=0)

    range_k_B = 1000 * beta(a=3.5, b=2, size=N[2])
    range_t_B = 0.008 * beta(a=4, b=2, size=N[2])
    range_utility_B = ((2e-3 * gamma(range_k_B, range_t_B, [len(dictNumAttributes['range']), N[2]]))
                       * np.array(dictNumAttributes['range'])[:, np.newaxis])
    range_utility_B -= range_utility_B.mean(axis=0)

    range_utility = np.hstack([range_utility_M, range_utility_X, range_utility_B]).T

    charge_k_M = 1000 * beta(a=4.5, b=2, size=N[0])
    charge_t_M = 0.007 * beta(a=4, b=2, size=N[0])
    charge_utility_M = ((2e-3 * gamma(charge_k_M, charge_t_M, [len(dictNumAttributes['charge']), N[0]]))
                       * np.array(dictNumAttributes['charge'])[:, np.newaxis])
    charge_utility_M -= charge_utility_M.mean(axis=0)

    charge_k_X = 1000 * beta(a=4, b=2, size=N[1])
    charge_t_X = 0.008 * beta(a=4, b=2, size=N[1])
    charge_utility_X = ((2e-3 * gamma(charge_k_X, charge_t_X, [len(dictNumAttributes['charge']), N[1]]))
                       * np.array(dictNumAttributes['charge'])[:, np.newaxis])
    charge_utility_X -= charge_utility_X.mean(axis=0)

    charge_k_B = 1000 * beta(a=3.5, b=2, size=N[2])
    charge_t_B = 0.008 * beta(a=4, b=2, size=N[2])
    charge_utility_B = ((2e-3 * gamma(charge_k_B, charge_t_B, [len(dictNumAttributes['charge']), N[2]]))
                       * np.array(dictNumAttributes['charge'])[:, np.newaxis])
    charge_utility_B -= charge_utility_B.mean(axis=0)

    charge_utility = np.hstack([charge_utility_M, charge_utility_X, charge_utility_B]).T

    energy_sig_M = 0.4 * beta(a=10, b=2, size=N[0])
    energy_mu_M = normal(loc=0.9, scale=0.1, size=N[0])
    energy_inter = (1 * normal(energy_mu_M, energy_sig_M, [1, N[0]]))
    energy_utility_M = np.vstack([-1 * energy_inter, energy_inter]).T

    energy_sig_X = 0.4 * beta(a=10, b=2, size=N[1])
    energy_mu_X = normal(loc=0.9, scale=0.1, size=N[1])
    energy_inter = (1 * normal(energy_mu_X, energy_sig_X, [1, N[1]]))
    energy_utility_X = np.vstack([-1 * energy_inter, energy_inter]).T

    energy_sig_B = 0.4 * beta(a=10, b=2, size=N[2])
    energy_mu_B = normal(loc=1.1, scale=0.1, size=N[2])
    energy_inter = (1.2 * normal(energy_mu_B, energy_sig_B, [1, N[2]]))
    energy_utility_B = np.vstack([-1 * energy_inter, energy_inter]).T

    energy_utility = np.vstack([energy_utility_M, energy_utility_X, energy_utility_B])

    type_mix_M = binomial(n=1, p=0.4, size=N[0])
    type_sig_M = 2 * beta(a=10, b=2, size=N[0])
    type_mu_M = normal(loc=(3 * (type_mix_M - 0.5)), scale=0.1, size=N[0])
    type_inter = (0.5 * normal(type_mu_M, type_sig_M, [1, N[0]]))
    type_utility_M = np.vstack([type_inter, -1 * type_inter]).T

    type_mix_X = binomial(n=1, p=0.2, size=N[1])
    type_sig_X = 2 * beta(a=10, b=2, size=N[1])
    type_mu_X = normal(loc=(3 * (type_mix_X - 0.5)), scale=0.1, size=N[1])
    type_inter = (0.5 * normal(type_mu_X, type_sig_X, [1, N[1]]))
    type_utility_X = np.vstack([type_inter, -1 * type_inter]).T

    type_mix_B = binomial(n=1, p=0.4, size=N[2])
    type_sig_B = 2 * beta(a=10, b=2, size=N[2])
    type_mu_B = normal(loc=(3 * (type_mix_B - 0.5)), scale=0.1, size=N[2])
    type_inter = (0.5 * normal(type_mu_B, type_sig_B, [1, N[2]]))
    type_utility_B = np.vstack([type_inter, -1 * type_inter]).T

    type_utility = np.vstack([type_utility_M, type_utility_X, type_utility_B])
    type_utility = type_utility.clip(-5, 5)

    brand_scale = 10 * beta(a=10, b=2, size=sum(N))
    brand_alloc = dirichlet(10*np.array([0.06, 0.07, 0.07, 0.01, 0.09, 0.44, 0.16, 0.1]), size=sum(N))
    brand_utility = brand_alloc * brand_scale[:, np.newaxis]
    brand_utility -= brand_utility.mean(axis=1)[:, np.newaxis]

    utility = np.hstack([brand_utility, mcost_utility, upcost_utility, term_utility, worth_utility, range_utility,
                         charge_utility, energy_utility, type_utility])

    # simulate the current market share
    simCust = multinomial(1, list(productShare.values()), sum(N))
    simProduct = [list(productShare.keys())[x] for x in np.argmax(simCust, axis=1)]

    aug = pd.DataFrame(np.array([list(range(1, sum(N) + 1, 1)), ['Millenial'] * N[0] + ['Gen X'] * N[1]
                                 + ['Baby Boomer'] * N[2], simProduct]).T, columns = ['id', 'segment', 'current brand'])

    k = 3  # we start at three as first three columns reserved for id, segment and current brand
    topLevel = [0, 1, 2]
    topLevelLab = ['id', 'segment', 'current brand']
    bottomLevelLab = ['id', 'segment', 'current brand']
    dictAttributes = {**dictCatAttributes, **dictNumAttributes}
    for col in ['id', 'brand', 'model', 'monthly_cost', 'upfront_cost', 'term',
                'vehicle_worth', 'range', 'charge', 'energy', 'vehicle_type']:
        if col not in ['id', 'model']:
            topLevel = topLevel + [k] * len(list(dictAttributes[col]))
            topLevelLab = topLevelLab + [col]
            bottomLevelLab = bottomLevelLab + [lvl for lvl in list(dictAttributes[col])]
            k += 1

    midx = pd.MultiIndex(levels=[list(topLevelLab), list(bottomLevelLab)],
                         codes=[topLevel, list(range(len(bottomLevelLab)))])

    utilityDf = pd.DataFrame(pd.concat([aug, pd.DataFrame(utility)], axis=1).values, columns=midx)

    return utilityDf


def interpolateUtility(utilityDf, augmentProductsDf, dictNumAttributes, calculate, delta=None, combine=None):
    """
    Calculte the nRespondent x nProduct matrix of utilities
    :param utilityDf: Respondent's recorded utility across various attributes + current brand, segment
    :param augmentProductsDf: ProductsDf with one-hot encoding for categorical variables
    :param dictNumAttributes: showns numerical variabless with accompanying grid of utility responses
    :param calculate: boolean whether we should calculate utility or read from file
    :param delta: if delta defined, it shows product index for which utility must be calculated
    :param combine: a previous nRespondent x nProduct matrix of utilities over which we will overwrite the delta column
    :return: i-th respondent's utility on the j-th product
    """

    utilityDf = utilityDf.copy()
    numerical = list(dictNumAttributes.keys())

    if calculate:
        productID = augmentProductsDf[('id', 'id')]

        listNumVec = []
        listGridVec = []
        listUtilityMat = []
        for numAttr in numerical:
            listNumVec = listNumVec + [augmentProductsDf[(numAttr, numAttr)].astype(float)]
            listGridVec = listGridVec + [utilityDf[numAttr].columns.astype(int)]
            listUtilityMat = listUtilityMat + [utilityDf[numAttr].values.astype(float)]

        nProduct = len(productID)
        n = utilityDf.shape[0]

        if delta is None:
            delta = range(nProduct)
            combineMat = np.full((n, nProduct), -9.)
        else:
            combineMat = combine.copy()  # in cases where delta has values, we still need to modify relevant columns

        # for each product, find which grid interval each attribute lies in
        matrixIndMapping = np.full([nProduct, len(numerical)], -9)  # matrix which contains
        for i, numAttr in enumerate(numerical):
            for j in range(nProduct):
                matrixIndMapping[j, i] = max(bisect.bisect(listGridVec[i], listNumVec[i][j]) - 1, 0)

        attrTensor = np.full((len(numerical), n, nProduct), -9.)
        for i in range(n):
            for j in delta:

                indices = matrixIndMapping[j, :]
                for k in range(len(numerical)):
                    if math.isnan(listNumVec[k][j]):
                        attrTensor[k, i, j] = 0.
                    else:
                        X0 = listGridVec[k][indices[k]]
                        X1 = listGridVec[k][indices[k] + 1]
                        Y0 = listUtilityMat[k][i, indices[k]]
                        Y1 = listUtilityMat[k][i, indices[k] + 1]
                        attrTensor[k, i, j] = ((Y1 - Y0) / (X1 - X0) * (listNumVec[k][j] - X0)) + Y0

                combineMat[i, j] = sum(attrTensor[:, i, j])
    else:
        combineMat = np.genfromtxt("./Precomputed Data/utilityNumeric.csv", delimiter=',')

    return combineMat


def OfferMatching(augmentProductsDf, utilityDf, productsDf, utilityNumeric,
                  delta=None, personProductUtilityMatUse=None, baselineDf=None):

    utilityDf = utilityDf.copy()
    segment = pd.DataFrame(utilityDf.loc[:, 'segment'].values)

    if baselineDf is not None:
        baselineBrand = baselineDf.loc[:, 'Baseline Brand']
        baselineProduct = baselineDf.loc[:, 'Baseline Product']

    ids = utilityDf[("id", "id")].astype(int)

    productID = augmentProductsDf.loc[:, ('id', 'id')]
    brand = productsDf.loc[:, 'brand']
    nProduct = len(productID)
    n = utilityDf.shape[0]
    if delta is None:
        delta = range(nProduct)
        personProductUtilityMat = np.full((n, nProduct), -9.)
    else:
        personProductUtilityMat = personProductUtilityMatUse.copy()

    featuresCat = augmentProductsDf[['brand', 'energy', 'vehicle_type']].values.astype(int)
    brandAttribute = augmentProductsDf.loc[:, 'brand'].values.astype(int)
    utilityCat = utilityDf[['brand', 'energy', 'vehicle_type']].values.astype(float)

    for j in delta:
        if (max(brandAttribute[j, :]) == 1):
            for i in range(n):
                sumOther = utilityNumeric[i, j]
                sumProduct = np.dot(featuresCat[j, :], utilityCat[i, :])
                personProductUtilityMat[i, j] = sumProduct + sumOther
        else:
            for i in range(n):
                personProductUtilityMat[i, j] = 0.

    predictChoice = pd.DataFrame(productID.loc[np.argmax(personProductUtilityMat, axis=1)].values)
    predictBrand = pd.DataFrame(brand.iloc[np.argmax(personProductUtilityMat, axis=1)].values)

    if baselineDf is None:
        baselineBrand = predictBrand.copy()
        baselineProduct = predictChoice.copy()

    OfferMatchingDf = pd.concat([ids, segment, baselineBrand, predictBrand, baselineProduct, predictChoice,
                                 pd.DataFrame(personProductUtilityMat)], axis=1)

    lvl_list = ['id', 'segment', 'Brand', 'Product'] + sorted(set(list(brand.astype(str))),
                                                                         key=list(brand.astype(str)).index)
    lvl_list2 = [lvl_list.index(str(x)) for x in productsDf['brand']]

    midx = pd.MultiIndex(levels=[lvl_list, ['id', 'segment', 'Baseline', 'Live'] + list(productID)],
                         codes=[[0, 1, 2, 2, 3, 3] + lvl_list2,
                                 [0, 1, 2, 3, 2, 3] + [4 + x for x in range(len(list(productID)))]])
    OfferMatchingDf = pd.DataFrame(OfferMatchingDf.values, columns=midx)

    return OfferMatchingDf, personProductUtilityMat


def simulatorOutput(augmentProductsDf, offerMatching):
    def getColumn(row):
        if row.iat[0] > 0:
            return row.iat[1]
        else:
            return ""

    simResult = pd.DataFrame(augmentProductsDf[('id', 'id')])
    simResult.columns = ['Product ID']

    brandDf = pd.concat([augmentProductsDf['brand'].astype(int).max(axis=1),
                         pd.DataFrame(augmentProductsDf['brand'].astype(int).idxmax(axis=1)),
                         augmentProductsDf['brand'].astype(int)], axis=1)
    simResult['brand'] = brandDf.apply(getColumn, axis=1)

    simResult['term'] = augmentProductsDf['term'].astype(int)

    simResult['monthly_cost'] = pd.DataFrame(augmentProductsDf['monthly_cost']).astype(float)
    simResult['upfront_cost'] = augmentProductsDf['upfront_cost'].astype(float)
    simResult['Price ($/mth)'] = ((simResult['monthly_cost'] * simResult['term']) + simResult['upfront_cost']) \
                                 / simResult['term']

    # numRespondents shows the number of people predicted to purchase each product
    numRespondents = offerMatching[[('Product', 'Live'), ('Brand', 'Live')]].groupby(('Product', 'Live')).count()
    numRespondents['prod ID'] = numRespondents.index
    numRespondents = pd.DataFrame(numRespondents.values, columns=['Live', 'prod ID'])  # flatten multiind col for merge

    simResult = simResult.merge(numRespondents, left_on='Product ID', right_on='prod ID',
                                how='outer', suffixes=['_left', '_right'])
    simResult.columns = ['Product ID', 'brand', 'term', 'monthly_cost', 'Price ($/mth)',
                         'upfront_cost', '# of respondents', 'prod ID']
    simResult['# of respondents'] = simResult['# of respondents'].fillna(0)
    del simResult['prod ID']

    total_respondent = simResult['# of respondents'].sum()
    simResult['Perc of respondents'] = simResult['# of respondents'] / total_respondent

    brand_total = simResult.groupby('brand')['# of respondents'].sum().to_frame()
    bt = [brand_total.loc[p]['# of respondents'] for p in simResult['brand']]
    simResult['Perc of brand'] = simResult['# of respondents'] / bt
    simResult['Perc of brand'] = simResult['Perc of brand'].fillna(0)

    # splits the proportion predicted to buy each model further according to which segment they belong to
    segmentDf = pd.DataFrame(offerMatching[[('Product', 'Live'), ('Brand', 'Live')]] \
                             .groupby(('Product', 'Live')).count() / total_respondent)
    segmentDf['prod ID'] = segmentDf.index
    segmentDf.columns = ['Tot Prop', 'prod ID']
    segmentDf = pd.concat([segmentDf['prod ID'], segmentDf['Tot Prop']], axis=1)
    segmentDf.columns = ['prod ID', 'Tot Prop']
    segmentDf['Millenial'] = offerMatching[offerMatching[('segment', 'segment')] == 'Millenial'] \
                                 [[('Product', 'Live'), ('Brand', 'Live')]].groupby(
        ('Product', 'Live')).count() / total_respondent
    segmentDf['Gen X'] = offerMatching[offerMatching[('segment', 'segment')] == 'Gen X'] \
                                 [[('Product', 'Live'), ('Brand', 'Live')]].groupby(
        ('Product', 'Live')).count() / total_respondent
    segmentDf['Baby Boomer'] = offerMatching[offerMatching[('segment', 'segment')] == 'Baby Boomer'] \
                                 [[('Product', 'Live'), ('Brand', 'Live')]].groupby(
        ('Product', 'Live')).count() / total_respondent

    simResult = simResult.merge(segmentDf, left_on='Product ID', right_on='prod ID',
                                how='outer', suffixes=['_left', '_right'])
    simResult.columns = ['Product ID', 'brand', 'term', 'monthly_cost', 'upfront_cost', 'Price ($/mth)',
                         '# of respondents', 'Perc of respondents', 'Perc of brand', 'prod ID', 'Tot Prop',
                         'Millenial', 'Gen X', 'Baby Boomer']
    del simResult['prod ID']
    del simResult['Tot Prop']
    simResult['Millenial'] = simResult['Millenial'].fillna(0)
    simResult['Gen X'] = simResult['Gen X'].fillna(0)
    simResult['Baby Boomer'] = simResult['Baby Boomer'].fillna(0)

    return simResult


def InFlowsOutFlowsBase(simDf, ProductsDf, offerMatching):
    def columnCreate(row):
        return "(" + str(row[0]) + ", " + str(row[1]) + ", " \
               + str(row[2]) + ", " + str(row[3]) + ", " + str(row[4]) + ")"

    flowsDf = simDf[['brand', 'Product ID']]
    flowsDf.columns = ['brand', 'Origin ID']

    numbersDf = offerMatching[[('Product', 'Baseline'), ('Brand', 'Baseline')]].groupby(('Product', 'Baseline')).count()
    numbersDf['org ID'] = numbersDf.index
    numbersDf = pd.DataFrame(numbersDf.values, columns=['Baseline', 'org ID'])  # flatten multiindex columns for merge
    flowsDf = flowsDf.merge(numbersDf, left_on='Origin ID', right_on='org ID',
                            how='outer', suffixes=['_left', '_right'])
    flowsDf.columns = ['brand', 'Origin ID', 'Origin Number', 'org ID']
    flowsDf['Origin Number'] = flowsDf['Origin Number'].fillna(0)
    del flowsDf['org ID']

    numbersDf = offerMatching[[('Product', 'Live'), ('Brand', 'Live')]].groupby(('Product', 'Live')).count()
    numbersDf['org ID'] = numbersDf.index
    numbersDf = pd.DataFrame(numbersDf.values, columns=['Live', 'org ID'])  # flatten multiindex columns before merge
    flowsDf = flowsDf.merge(numbersDf, left_on='Origin ID', right_on='org ID',
                            how='outer', suffixes=['_left', '_right'])
    flowsDf.columns = ['brand', 'Origin ID', 'Origin Number', 'Destination Number', 'org ID']
    flowsDf['Destination Number'] = flowsDf['Destination Number'].fillna(0)
    del flowsDf['org ID']

    brand = ProductsDf['brand']
    term = ProductsDf['term']
    monthlyCost = ProductsDf['monthly_cost']
    upfrontCost = ProductsDf['upfront_cost']
    columnsDf = pd.concat([flowsDf['Origin ID'], brand, term, monthlyCost, upfrontCost], axis=1)
    columnsDf = columnsDf.apply(columnCreate, axis=1)

    splits = offerMatching[[('Product', 'Baseline'), ('Product', 'Live'), ('Brand', 'Baseline')]] \
        .groupby([('Product', 'Baseline'), ('Product', 'Live')]).count()
    splits.index = splits.index.set_levels(splits.index.levels[1].astype(int), level=1)
    splits.index = splits.index.set_levels(splits.index.levels[0].astype(int), level=0)
    nProduct = ProductsDf.shape[0]
    interVal = np.full((nProduct, nProduct), 0.)
    for i in range(splits.shape[0]):
        interVal[splits.index[i][0] - 1, splits.index[i][1] - 1] = \
            splits.at[(splits.index[i][0], splits.index[i][1]), ('Brand', 'Baseline')]

    numbers = flowsDf['Origin Number'].values
    stayers = np.full(nProduct, 0.)
    moveOut = np.full(nProduct, 0.)
    moveIn = np.full(nProduct, 0.)
    for i in range(nProduct):
        stayers[i] = interVal[i, i]
        if numbers[i] == 0:
            moveOut[i] = 0
        else:
            moveOut[i] = numbers[i] - stayers[i]
        moveIn[i] = sum(interVal[:, i]) - stayers[i]
    stayers = pd.DataFrame(stayers)
    stayers.columns = ["Stayers"]
    moveOut = pd.DataFrame(moveOut)
    moveOut.columns = ["Moves out"]
    moveIn = pd.DataFrame(moveIn)
    moveIn.columns = ["Moves in"]

    interVal = pd.DataFrame(interVal)
    interVal.columns = list(columnsDf.values)

    rawInflowsOutflows = pd.concat([flowsDf, stayers, moveOut, moveIn, interVal], axis=1)

    return rawInflowsOutflows


def fix_row_sum(df):
    df['Total'] = df.drop('Total', axis=0, errors='ignore').drop('Total', axis=1, errors='ignore').sum(axis=1)
    return df


def fix_col_sum(df):
    df.loc['Total'] = df.drop('Total', axis=0, errors='ignore').drop('Total', axis=1, errors='ignore').sum()
    return df


def Simulate(simDf, marketAssumptions, Total_market_volume=23.287):

    nSegment = 3
    nBrand = 8

    marketAssumptions = marketAssumptions.copy()
    re_contracting_df = marketAssumptions[0]
    churn_df = marketAssumptions[1]
    base_ARPU_df = marketAssumptions[2]
    base_V_ARPU_Rev_in = marketAssumptions[3]
    survey_result = marketAssumptions[4]  # columns are 'id', 'segment' and 'current brand'
    # columns = SIO_new, revenue_new, EBIT_new from Precomputed "live"
    simulate_new = marketAssumptions[5]  # this could be None if the Precomputed Data not available

    simResult = simDf[simDf['brand'] != ""]

    # splits the proportion predicted to buy each brand according to the segment they belong to
    sim_summary_by_seg = simResult.groupby('brand')[['Millenial', 'Gen X', 'Baby Boomer']].sum()
    sim_summary_by_seg['Total'] = sim_summary_by_seg.sum(axis=1)
    sim_summary_by_seg = sim_summary_by_seg.sort_values(by='Total', ascending=False)
    sim_summary_by_seg = fix_col_sum(sim_summary_by_seg)

    # identical to sim_summary_by_seg, except SBn is the recorded brand purchases of respondents (not predictions)
    SBn = survey_result.groupby(['segment', 'current brand']).count()['id'].fillna(0).reset_index().pivot(
        index='segment', columns='current brand', values='id')
    SBn = SBn[['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']]
    SBn.loc['Total'] = SBn.sum()
    SBn['Total'] = SBn.sum(axis=1)
    Segment_brand_n = SBn.fillna(0)

    x = Segment_brand_n.to_numpy()
    BD = Segment_brand_n.drop('Total')
    BD.iloc[0:nSegment, :] = x[0:nSegment] / x[nSegment]
    Breakdown_by_brand = BD
    # shows actual prop of each segment buying each brand (sum of brand row over segments = 1)
    base_SIO_df = Breakdown_by_brand[['Audi', 'Chevrolet', 'Jaguar', 'Kia',
                                      'Nissan', 'Tesla', 'Toyota', 'VW']].transpose()

    base_V_ARPU_Rev = base_V_ARPU_Rev_in
    base_V_ARPU_Rev['SIO (K)'] = Total_market_volume * base_V_ARPU_Rev['Share of SIOs'].astype(float)
    base_V_ARPU_Rev['APRU ($)'] = (base_SIO_df.values * base_ARPU_df.iloc[:, 1:]).sum(axis=1)
    base_V_ARPU_Rev['revenue ($K)'] = base_V_ARPU_Rev['SIO (K)'] * base_V_ARPU_Rev['APRU ($)']
    base_V_ARPU_Rev['EBIT ($K)'] = base_V_ARPU_Rev['revenue ($K)'] * base_V_ARPU_Rev['EBIT margin per SIO']
    base_V_ARPU_Rev['Variable cost percentage'] = 1 - base_V_ARPU_Rev['Fixed cost percentage']
    if simulate_new is not None:
        base_V_ARPU_Rev['Baseline EBIT'] = simulate_new['revenue_new'] * base_V_ARPU_Rev_in['EBIT margin per SIO']
        base_V_ARPU_Rev['Fixed cost base'] = (simulate_new['revenue_new'] - base_V_ARPU_Rev['Baseline EBIT']) * \
                                             base_V_ARPU_Rev['Fixed cost percentage']
        base_V_ARPU_Rev['Variable cost per SIO'] = ((simulate_new['revenue_new'] - base_V_ARPU_Rev['Baseline EBIT']) *
                                                    base_V_ARPU_Rev['Variable cost percentage']) / \
                                                   simulate_new['SIO_new']
        base_V_ARPU_Rev['Live EBIT margin'] = simulate_new['EBIT_new'] / simulate_new['revenue_new']

    churn_cohort_df = (fix_row_sum(base_SIO_df.values * re_contracting_df.iloc[:, 1:])['Total']
                       .to_frame()
                       .rename(index={'current brand': 'Brand'},
                               columns={'Total': 'Perc recontract'}))
    # wght avg of churn rate (wght = segment actual prop.)
    churn_cohort_df['Perc churn'] = fix_row_sum(base_SIO_df.values * churn_df.iloc[:, 1:])['Total']
    churn_cohort_df['Perc inert'] = 1 - churn_cohort_df['Perc churn'] - churn_cohort_df['Perc recontract']
    churn_cohort_df['# recontract'] = base_V_ARPU_Rev['SIO (K)'] * churn_cohort_df['Perc recontract']
    churn_cohort_df['# churn'] = base_V_ARPU_Rev['SIO (K)'] * churn_cohort_df['Perc churn']
    churn_cohort_df['# inert'] = base_V_ARPU_Rev['SIO (K)'] * churn_cohort_df['Perc inert']

    # econ_recontract = 5b, econ_churn_out = 5c, econ_churn_in = 5d, econ_inert = 5e
    econ_recontract = re_contracting_df.iloc[:, 1:] * base_SIO_df.values  # wght avg recontract rate (wght=seg act prop)
    x = econ_recontract.values
    y = base_V_ARPU_Rev['SIO (K)'].values
    for i in range(0, len(y), 1):
        x[i] = y[i] * x[i]
    econ_recontract.iloc[:, :] = x
    econ_recontract.index = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']

    econ_churn_out = churn_df.iloc[:, 1:] * base_SIO_df.values  # wght avg churn rate (wght = segment actual proportion)
    x = econ_churn_out.values
    y = base_V_ARPU_Rev['SIO (K)'].values
    for i in range(0, len(y), 1):
        x[i] = y[i] * x[i]
    econ_churn_out.iloc[:, :] = x
    econ_churn_out = fix_col_sum(econ_churn_out)
    econ_churn_out.index = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW', 'Total']

    # takes proportion predicted to buy model (by segment) multiplied by actual segment churn numbers = churn in
    x = econ_churn_out[econ_churn_out.index == 'Total'].values[:, 0:nSegment] \
        * sim_summary_by_seg.values[0:nBrand,0:nSegment] / sim_summary_by_seg.values[nBrand, 0:nSegment]
    econ_churn_in = pd.DataFrame(x,  # index=['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'],
                                 index=sim_summary_by_seg.index[:nBrand], columns=['Millenial', 'Gen X', 'Baby Boomer'])
    econ_inert = pd.DataFrame(1 * base_SIO_df.values)
    x = econ_inert.values
    y = churn_cohort_df['# inert'].values
    for i in range(0, len(y), 1):
        x[i] = y[i] * x[i]
    econ_inert = pd.DataFrame(x, index=['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'],
                              columns=['Millenial', 'Gen X', 'Baby Boomer'])

    SIO_dyn_start = econ_recontract + econ_churn_out.drop('Total', axis=0) + econ_inert
    SIO_dyn_end = econ_recontract + econ_churn_in + econ_inert
    ARPU_dyn_start = base_ARPU_df.copy()

    # workers predicted to buy each brand (partitioned by segment) and multiplied by revenue per month
    A = simResult[['brand', 'Millenial', 'Gen X', 'Baby Boomer']]
    A.loc[:, 'Millenial'] = simResult['Price ($/mth)'] * A['Millenial']
    A.loc[:, 'Gen X'] = simResult['Price ($/mth)'] * A['Gen X']
    A.loc[:, 'Baby Boomer'] = simResult['Price ($/mth)'] * A['Baby Boomer']
    A = A.groupby(by='brand').sum()

    # workers predicted to buy each brand (partitioned by segment)
    B = simResult[['brand', 'Millenial', 'Gen X', 'Baby Boomer']]
    B = B.groupby(by='brand').sum()

    ARPU_Seg = (A / B).transpose()[['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']]
    ARPU_Seg = ARPU_Seg.fillna(ARPU_Seg.mean())  # this will break if entire column is NaN
    part1 = pd.DataFrame(econ_inert.values * ARPU_dyn_start.iloc[:, 1:])
    part1.index = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    ARPU_dyn_end = (part1 + (econ_recontract + econ_churn_in) * ARPU_Seg.transpose()) / SIO_dyn_end

    revenue_dyn_start = SIO_dyn_start.values * ARPU_dyn_start.iloc[:, 1:]
    revenue_dyn_start.index = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    revenue_dyn_start = fix_row_sum(revenue_dyn_start)
    revenue_dyn_end = SIO_dyn_end * ARPU_dyn_end
    revenue_dyn_end = fix_row_sum(revenue_dyn_end)  # .sort_values('Total', ascending=False)

    ## prepare result to output
    live = churn_df['brand'].to_frame()  ## pick up the indexed
    live['SIO_base'] = base_V_ARPU_Rev['SIO (K)']
    live['SIO_new'] = fix_row_sum(SIO_dyn_end)['Total'].values
    live['SIO_change'] = live['SIO_new'] - live['SIO_base']
    live['SIO_Percchange'] = live['SIO_change'] / live['SIO_base']

    live['ARPU_base'] = fix_row_sum(revenue_dyn_start)['Total'].values / fix_row_sum(SIO_dyn_start)['Total'].values
    live['ARPU_new'] = fix_row_sum(revenue_dyn_end)['Total'].values / fix_row_sum(SIO_dyn_end)['Total'].values
    live['ARPU_change'] = live['ARPU_new'] - live['ARPU_base']
    live['ARPU_Percchange'] = live['ARPU_change'] / live['ARPU_base']

    live['revenue_base'] = live['SIO_base'] * live['ARPU_base']
    live['revenue_new'] = live['SIO_new'] * live['ARPU_new']
    live['revenue_change'] = live['revenue_new'] - live['revenue_base']
    live['revenue_Percchange'] = live['revenue_change'] / live['revenue_base']

    live['EBIT_base'] = base_V_ARPU_Rev['EBIT ($K)']
    if simulate_new is None:
        live['EBIT_new'] = live['revenue_new'] * base_V_ARPU_Rev['EBIT margin per SIO']
    else:
        live['EBIT_new'] = live['revenue_new'] - base_V_ARPU_Rev['Fixed cost base']\
                           - (live['SIO_new'] * base_V_ARPU_Rev['Variable cost per SIO'])
    live['EBIT_change'] = live['EBIT_new'] - live['EBIT_base']
    live['EBIT_Percchange'] = live['EBIT_change'] / live['EBIT_base']

    numDynamic = pd.concat([churn_cohort_df['# recontract'], churn_cohort_df['# churn']], axis=1)
    numDynamic.index = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    return live, numDynamic, econ_churn_out, churn_cohort_df


def InFlowsOutFlows(simDf, ProductsDf, rawInflowsOutflows, numDynamic, offerMatching, churnOutSIO):
    """
    :param simDf: dataframe (nProduct x {9 + nSegment} cols) Shows number of respondents predicted to purchase each
        product, divided by segment ['Product ID', 'brand', 'term', 'monthly_cost', 'upfront_cost', 'Price ($/mth)',
        '# of respondents', 'Perc of respondents', 'Perc of brand']
    :param ProductsDf: datafrane (nProduct x 12 cols) Shows attributes of Product
        ['id', 'brand', 'model', 'monthly_cost', 'upfront_cost', 'term', 'vehicle_worth', 'range', 'charge', 'energy',
        'vehicle_type', 'colours']
    :param rawInflowsOutflows: dataframe (nProduct x {7 + nProduct}) . Give movement statistics. Rows/Col = From/To
        ['brand', 'Origin ID', 'Origin Number', 'Destination Number', 'Stayers', 'Moves out', 'Moves in']
    :param numDynamic: dataframe (nBrand x 2) ['# recontract', '# churn']
    :param offerMatching: dataframe (nRespondent x {6 + nProduct}) Calculates utility scores for each product-respondent
        ['id', 'segment', 'Brand-Baseline', 'Brand-Live', 'Product-Baseline', 'Product-Live']
    :param churnOutSIO: datframe ((nBrand + 1) x nSegment) [Final row = 'Total']
    :return: dataframe (nBrand x {3 + nBrand}) ['Stayers', 'Leavers', 'Joiners']
    """
    brands = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    segments = ['Millenial', 'Gen X', 'Baby Boomer']

    SIOplan = pd.DataFrame(simDf.loc[:, 'Product ID'])
    SIOplan.columns = ['Offer']
    SIOplan['brand'] = ProductsDf.loc[:, 'brand']
    SIOplan['term'] = ProductsDf.loc[:, 'term']
    SIOplan['monthly_cost'] = ProductsDf.loc[:, 'monthly_cost']
    SIOplan['upfront_cost'] = ProductsDf.loc[:, 'upfront_cost']
    SIOplan['Baseline Perc share'] = rawInflowsOutflows['Origin Number'] / rawInflowsOutflows['Origin Number'].sum()
    SIOplan['Scenario Perc share'] = rawInflowsOutflows['Destination Number'] / rawInflowsOutflows[
        'Destination Number'].sum()
    SIOplan['Price'] = simDf.loc[:, 'Price ($/mth)']

    brandSIObaseline = SIOplan[['brand', 'Baseline Perc share']].groupby('brand').sum()
    brandSIOscenario = SIOplan[['brand', 'Scenario Perc share']].groupby('brand').sum()
    totChurn = numDynamic['# churn'].sum()

    # calculate the (number recontracting to product) plus (number churning to product) under baseline scenario
    def calcSIObaseline(row):
        if isinstance(row[1], str):
            return ((numDynamic.at[row[1], '# recontract'] * row[5])
                    / brandSIObaseline.at[row[1], 'Baseline Perc share']) + (totChurn * row[5])
        elif math.isnan(row[1]):
            return 0.
        else:
            return ((numDynamic.at[row[1], '# recontract'] * row[5])
                    / brandSIObaseline.at[row[1], 'Baseline Perc share']) + (totChurn * row[5])

    # calculate the (number recontracting to product) plus (number churning to product) under new scenario
    def calcSIOscenario(row):
        if isinstance(row[1], str):
            return ((numDynamic.at[row[1], '# recontract'] * row[6])
                    / brandSIOscenario.at[row[1], 'Scenario Perc share']) + (totChurn * row[6])
        elif math.isnan(row[1]):
            return 0.
        else:
            return ((numDynamic.at[row[1], '# recontract'] * row[6])
                    / brandSIOscenario.at[row[1], 'Scenario Perc share']) + (totChurn * row[6])

    def maxZero(row):
        return max(-1 * row[9], 0)

    SIOplan['Baseline SIOs (K)'] = SIOplan.apply(calcSIObaseline, axis=1)
    SIOplan['Scenario SIOs (K)'] = SIOplan.apply(calcSIOscenario, axis=1)
    SIOplan['Delta SIOs (K)'] = SIOplan['Scenario SIOs (K)'] - SIOplan['Baseline SIOs (K)']
    SIOplan['Stayers SIOs (K)'] = SIOplan[['Baseline SIOs (K)', 'Scenario SIOs (K)']].min(axis=1)
    SIOplan['Leavers SIOs (K)'] = SIOplan.apply(maxZero, axis=1)
    SIOplan['Joiners SIOs (K)'] = (-1 * SIOplan).apply(maxZero, axis=1)

    col1 = SIOplan[['brand', 'Baseline SIOs (K)']].groupby('brand').sum()
    #del col1.index.name
    col1.index.name = None  # compatible with Pandas v1.0
    col2 = SIOplan[['brand', 'Scenario SIOs (K)']].groupby('brand').sum()
    #del col2.index.name
    col2.index.name = None
    col4 = SIOplan[['brand', 'Stayers SIOs (K)']].groupby('brand').sum()
    #del col4.index.name
    col4.index.name = None
    col5 = SIOplan[['brand', 'Leavers SIOs (K)']].groupby('brand').sum()
    #del col5.index.name
    col5.index.name = None
    col6 = SIOplan[['brand', 'Joiners SIOs (K)']].groupby('brand').sum()
    #del col6.index.name
    col6.index.name = None

    churnStat = pd.concat([col1, col2], axis=1)
    churnStat['Delta'] = col2['Scenario SIOs (K)'] - col1['Baseline SIOs (K)']
    churnStat = churnStat.reindex(brands)
    churnStat['Stayers'] = col4
    churnStat['Leavers'] = col5
    churnStat['Joiners'] = col6

    denom = SIOplan[['brand', 'Baseline Perc share', 'Scenario Perc share']].groupby('brand').sum()
    #del denom.index.name
    denom.index.name = None

    interDf = SIOplan[['brand', 'monthly_cost', 'Baseline Perc share', 'Scenario Perc share']].copy()
    interDf['product1'] = SIOplan['Price'] * SIOplan['Baseline Perc share']
    interDf['product2'] = SIOplan['Price'] * SIOplan['Scenario Perc share']
    num1 = interDf[['brand', 'product1']].groupby('brand').sum()
    #del num1.index.name
    num1.index.name = None
    num1 = interDf[['brand', 'product1']].groupby('brand').sum()
    #del num1.index.name
    num1.index.name = None
    num2 = interDf[['brand', 'product2']].groupby('brand').sum()
    #del num2.index.name
    num2.index.name = None

    churnStat['Before MMC'] = num1['product1'] / denom['Baseline Perc share']
    churnStat['After MMC'] = num2['product2'] / denom['Scenario Perc share']

    interDf = offerMatching[
        [('id', 'id'), ('segment', 'segment'), ('Brand', 'Baseline'), ('Brand', 'Live')]]
    interDf.columns = ['ID', 'Segment', 'Baseline Brand', 'Live Brand']
    nSegment = interDf['Segment'].nunique()
    nProviders = interDf['Baseline Brand'].nunique()
    interDf = interDf.groupby(['Segment', 'Baseline Brand', 'Live Brand']).count()

    valueCanvas = np.zeros((nProviders, nProviders * nSegment))
    levelsMIDX = [segments, brands]
    listInd = [np.tile(np.array([x]), nProviders) for x in range(nSegment)]
    codes1 = [item for sublist in listInd for item in sublist]
    codes2 = np.tile(range(nProviders), nSegment)
    codesMIDX = [codes1, codes2]
    midx = pd.MultiIndex(levels=levelsMIDX, codes=codesMIDX)
    SIObySegmentProvider = pd.DataFrame(valueCanvas, columns=midx, index=brands)
    for i in range(interDf.shape[0]):
        colX = interDf.index[i]
        SIObySegmentProvider.at[colX[1], (colX[0], colX[2])] = interDf.iat[i, 0]

    segmentTot = pd.DataFrame([SIObySegmentProvider.loc[:, segments[i]].sum().sum() for i in range(nSegment)],
                              index=segments)
    segmentTot.columns = ['SegmentTot']
    PercBySegmentProvider = pd.DataFrame(valueCanvas, columns=midx, index=brands)
    for i in range(interDf.shape[0]):
        colX = interDf.index[i]
        PercBySegmentProvider.at[colX[1], (colX[0], colX[2])] = interDf.iat[i, 0] / segmentTot.at[colX[0], 'SegmentTot']

    PercBySegmentProvider = PercBySegmentProvider.copy()
    MarketNumBySegmentProvider = pd.DataFrame(valueCanvas, columns=midx, index=brands)
    for i in range(interDf.shape[0]):
        colX = interDf.index[i]
        MarketNumBySegmentProvider.at[colX[1], (colX[0], colX[2])] = (interDf.iat[i, 0]
                                                                      * churnOutSIO.at['Total', colX[0]])\
                                                                     / segmentTot.at[colX[0], 'SegmentTot']

    # Aggregate the Brand by Brand Transitions across the three segments
    MarketNumByProvider = MarketNumBySegmentProvider['Millenial']
    MarketNumByProvider = MarketNumByProvider + MarketNumBySegmentProvider['Gen X']
    MarketNumByProvider = MarketNumByProvider + MarketNumBySegmentProvider['Baby Boomer']

    diagVal = np.diag(np.array(MarketNumByProvider.values))
    MarketNumByProvider['RowTotal'] = MarketNumByProvider.sum(axis=1)
    MarketNumByProvider['ColTotal'] = MarketNumByProvider.sum(axis=0)
    MarketNumByProvider['Stayers'] = diagVal
    MarketNumByProvider['Leavers'] = MarketNumByProvider['RowTotal'] - MarketNumByProvider['Stayers']
    MarketNumByProvider['Joiners'] = MarketNumByProvider['ColTotal'] - MarketNumByProvider['Stayers']
    del MarketNumByProvider['RowTotal']
    del MarketNumByProvider['ColTotal']

    for i in range(nSegment):
        diagVal = np.diag(np.array(SIObySegmentProvider[segments[i]].values))
        SIObySegmentProvider[(segments[i], 'RowTotal')] = SIObySegmentProvider[segments[i]].sum(axis=1)
        SIObySegmentProvider[(segments[i], 'ColTotal')] = SIObySegmentProvider[segments[i]].sum(axis=0)
        SIObySegmentProvider[(segments[i], 'Stayers')] = diagVal
        SIObySegmentProvider[(segments[i], 'Leavers')] = SIObySegmentProvider[(segments[i], 'RowTotal')] - \
                                                         SIObySegmentProvider[(segments[i], 'Stayers')]
        SIObySegmentProvider[(segments[i], 'Joiners')] = SIObySegmentProvider[(segments[i], 'ColTotal')] - \
                                                         SIObySegmentProvider[(segments[i], 'Stayers')]
        del SIObySegmentProvider[(segments[i], 'RowTotal')]
        del SIObySegmentProvider[(segments[i], 'ColTotal')]

    return SIOplan, churnStat, SIObySegmentProvider, PercBySegmentProvider, \
           MarketNumBySegmentProvider, MarketNumByProvider


def performPrecomputation(N, writeUtility, folderStr, destFolder):

    ProductsDf = pd.read_csv(folderStr + "leaseOffers.csv")

    productShare = {'Audi': 0.028, 'Chevrolet': 0.070, 'Jaguar': 0.010, 'Kia': 0.019, 'Nissan': 0.049,
                    'Tesla': 0.722, 'Toyota': 0.084, 'VW': 0.018}

    setMonthly = pd.Series([120, 200, 250, 400, 500, 750, 1000, 1250])
    setUpfront = pd.Series([2000, 4000, 6000, 8000, 10000])
    setTerm = pd.Series([24, 36, 48])
    setWorth = pd.Series([20000, 30000, 40000, 80000, 100000])
    setRange = pd.Series([20, 40, 80, 180, 300])
    setCharge = pd.Series([3, 10, 50, 100, 160])
    dictNumAttributes = {'monthly_cost': setMonthly, 'upfront_cost': setUpfront, 'term': setTerm,
                         'vehicle_worth': setWorth, 'range': setRange, 'charge': setCharge}

    setBrands = pd.Series(['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'])
    setEnergy = pd.Series(['Electric Vehicle', 'Plug-in Hybrid'])
    setType = pd.Series(['Sedan', 'SUV'])
    dictCatAttributes = {'brand': setBrands, 'energy': setEnergy, 'vehicle_type': setType}

    re_contracting_df = pd.read_csv(folderStr + "recontractRateEV.csv")
    re_contracting_df = pd.concat([re_contracting_df['brand'],
                                   re_contracting_df[['Millenial', 'Gen X', 'Baby Boomer']].astype(float)], axis=1)
    churn_df = pd.read_csv(folderStr + "churnRateEV.csv")
    churn_df = pd.concat([churn_df['brand'],
                          churn_df[['Millenial', 'Gen X', 'Baby Boomer']].astype(float)], axis=1)
    base_ARPU_df = pd.read_csv(folderStr + "baseARPU_EV.csv")
    base_ARPU_df = pd.concat([base_ARPU_df['brand'],
                              base_ARPU_df[['Millenial', 'Gen X', 'Baby Boomer']].astype(float)], axis=1)
    base_V_ARPU_Rev_in = pd.read_csv(folderStr + "baseVolumeEBIT_EV.csv")
    base_V_ARPU_Rev_in = pd.concat(
        [base_V_ARPU_Rev_in['brand'], base_V_ARPU_Rev_in[['Share of SIOs', 'EBIT margin per SIO',
                                                          'Fixed cost percentage']].astype(float)], axis=1)

    augmentProductsDf = createAugmentedProductsDf(ProductsDf, dictCatAttributes)

    if writeUtility:
        utilityDf = simulateUtility(N, productShare, dictNumAttributes, dictCatAttributes)
        utilityDf.to_csv(folderStr + 'UtilityScoresEV.csv', index=False)
    else:
        utilityDf = readHierarchicalDf(folderStr + "UtilityScoresEV.csv")

    marketAssumptions = [re_contracting_df.copy(), churn_df.copy(), base_ARPU_df.copy(), base_V_ARPU_Rev_in.copy(),
                         pd.DataFrame(utilityDf[['id', 'segment', 'current brand']].values,
                                      columns=['id', 'segment', 'current brand']), None]

    utilityNumeric = interpolateUtility(utilityDf, augmentProductsDf, dictNumAttributes, True, delta=None, combine=None)
    pd.DataFrame(utilityNumeric).to_csv(destFolder + 'utilityNumeric.csv', index=False, header=False)

    offerMatching, personProdMatch = OfferMatching(augmentProductsDf, utilityDf, ProductsDf, utilityNumeric,
                                                   None, None, None)
    offerMatching.to_csv(destFolder + 'offerMatching.csv', index=False)
    baselinePred = offerMatching[[("id", "id"), ("Brand", "Baseline"), ("Product", "Baseline")]]
    baselinePred.columns = ['id', 'Baseline Brand', 'Baseline Product']
    baselinePred.to_csv(destFolder + 'baselinePrediction.csv', index=False)
    pd.DataFrame(personProdMatch).to_csv(destFolder + 'personProdMatch.csv', index=False, header=False)

    simResult = simulatorOutput(augmentProductsDf, offerMatching)
    live, numDynamic, churnOutSIO, churn_cohort_df = Simulate(simResult, marketAssumptions)
    new_simulate = pd.DataFrame(live[['SIO_new', 'revenue_new', 'EBIT_new']])
    new_simulate.to_csv(destFolder + 'simulate_new.csv', index=False, header=False)

    print("Computation Completed and Saved to %s folder." %destFolder)

if __name__ == '__main__':

    folderStr = './Data/'
    destFolder = './Precomputed Data/'
    N = [200, 200, 160]
    writeUtility = True

    setBrands = pd.Series(['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'])
    setEnergy = pd.Series(['Electric Vehicle', 'Plug-in Hybrid'])
    setType = pd.Series(['Sedan', 'SUV'])
    dictCatAttributes = {'brand': setBrands, 'energy': setEnergy, 'vehicle_type': setType}

    performPrecomputation(N, writeUtility, folderStr, destFolder)

    ProductsDf = pd.read_csv(folderStr + "leaseOffers.csv")
    augmentProductsDf = createAugmentedProductsDf(ProductsDf, dictCatAttributes)

    offerMatching = readHierarchicalDf(destFolder + 'offerMatching.csv')
    simResult = simulatorOutput(augmentProductsDf, offerMatching)

    rawInflowsOutflows = InFlowsOutFlowsBase(simResult, ProductsDf, offerMatching)

    print(rawInflowsOutflows)