import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
from analyze import *
from textwrap import dedent as d
import plotly.graph_objs as go
import plotly.figure_factory as ff

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

folderStr = "./Data/"
destFolder = "./Precomputed Data/"

color_dict = {'Audi': 'mediumblue', 'Chevrolet': 'gold', 'Jaguar': 'teal', 'Kia': 'deepskyblue',
              'Nissan': 'red', 'Tesla': 'purple', 'Toyota': 'orange', 'VW': 'deeppink'}
ProductsDf = pd.read_csv(folderStr + "leaseOffers.csv")
ProductsDfEdit = ProductsDf[["id", "brand", "model", "monthly_cost", "term", "upfront_cost"]]
ProductsDfEdit.columns = ["id", "brand", "model", "monthly", "term", "upfront"]
ProductsDf['colours'] = ProductsDf['brand'].map(color_dict)

utilityDf = readHierarchicalDf(folderStr + "UtilityScoresEV.csv")
survey_result = pd.DataFrame(utilityDf[['id', 'segment', 'current brand']].values,
                                      columns=['id', 'segment', 'current brand'])
personProductUtilityMat = np.genfromtxt(destFolder + "personProdMatch.csv", delimiter=",")
combineMat = np.genfromtxt(destFolder + "utilityNumeric.csv", delimiter=',')
baselineDf = pd.read_csv(destFolder + "baselinePrediction.csv")

setMonthly = pd.Series([120, 200, 250, 400, 500, 750, 1000])
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
base_V_ARPU_Rev_in = pd.concat([base_V_ARPU_Rev_in['brand'], base_V_ARPU_Rev_in[['Share of SIOs', 'EBIT margin per SIO',
                                                      'Fixed cost percentage']].astype(float)], axis=1)
revenue_new = pd.read_csv(destFolder + 'simulate_new.csv', header=None)
revenue_new.columns = ['SIO_new', 'revenue_new', 'EBIT_new']
marketAssumptions = [re_contracting_df.copy(), churn_df.copy(), base_ARPU_df.copy(),
                    base_V_ARPU_Rev_in.copy(), survey_result.copy(), revenue_new.copy()]


def PDtoDict(row):
    return eval("dict(" + ProductsDfEdit.columns[0] + " = " + str(row.iat[0]) + ", "
                + ProductsDfEdit.columns[1] + " = \'" + str(row.iat[1]) + "\', "
                + ProductsDfEdit.columns[2] + " = \'" + str(row.iat[2]) + "\', "
                + ProductsDfEdit.columns[3] + " = " + str(row.iat[3]) + ", "
                + ProductsDfEdit.columns[4] + " = " + str(row.iat[4]) + ", "
                + ProductsDfEdit.columns[5] + " = " + str(row.iat[5]) + ")")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def sweepCalculate(sweepSpecify, attribute, sweepType="Lockstep"):

    k = 0
    mnthlyCond = (attribute == 'Monthly') or (attribute == 'Monthly-Upfront')
    upfrntCond = (attribute == 'Upfront') or (attribute == 'Monthly-Upfront')
    segments = ['Millenial', 'Gen X', 'Baby Boomer']
    attLower = attribute.lower()

    if sweepType not in ["Lockstep", "Attribute Grid", "univariate"]:
        raise ValueError("Invalid sweepType. Choose 'Lockstep', 'Attribute Grid' or 'univariate'.")

    sweepSpecify = sweepSpecify.astype(float)
    if (sweepSpecify[['Monthly Step', 'Upfront Step']] == 0).max().max():
        raise ValueError("Monthly/Upfront steps should be nonzero.")
    elif ((sweepSpecify['Monthly End'] - sweepSpecify['Monthly Begin']) % sweepSpecify['Monthly Step']).max() > 0:
        raise ValueError("Monthly steps are not a multiple of Monthly range. Change sweepSpecify.")
    elif ((sweepSpecify['Upfront End'] - sweepSpecify['Upfront Begin']) % sweepSpecify['Upfront Step']).max() > 0:
        raise ValueError("Upfront steps are not a multiple of Upfront range. Change sweepSpecify.")
    elif sweepType == "Lockstep":
        upfrntVar = ((sweepSpecify['Upfront End'] - sweepSpecify['Upfront Begin']) / sweepSpecify['Upfront Step']).std()
        mnthlyVar = ((sweepSpecify['Monthly End'] - sweepSpecify['Monthly Begin']) / sweepSpecify['Monthly Step']).std()
        if (upfrntVar > 0) or (mnthlyVar > 0):
            raise ValueError(
                "Number of Monthly/Upfront grid points arent identical across Products. Can't perform Lockstep sweep.")

    if mnthlyCond:
        mnthlyGridN = len(np.arange(sweepSpecify.loc[k, 'Monthly Begin'],
                                    sweepSpecify.loc[k, 'Monthly End'] + sweepSpecify.loc[k, 'Monthly Step'],
                                    sweepSpecify.loc[k, 'Monthly Step']))
    if upfrntCond:
        upfrntGridN = len(np.arange(sweepSpecify.loc[k, 'Upfront Begin'],
                                    sweepSpecify.loc[k, 'Upfront End'] + sweepSpecify.loc[k, 'Upfront Step'],
                                    sweepSpecify.loc[k, 'Upfront Step']))
    if mnthlyCond:
        mnthlyGridX = np.zeros((sweepSpecify.shape[0], mnthlyGridN))
    if upfrntCond:
        upfrntGridX = np.zeros((sweepSpecify.shape[0], upfrntGridN))

    for product in sweepSpecify['Product ID'].index:
        if mnthlyCond:
            mnthlyGridX[product, ] = np.arange(sweepSpecify.loc[product, 'Monthly Begin'],
                                               sweepSpecify.loc[product, 'Monthly End']
                                               + sweepSpecify.loc[product, 'Monthly Step'],
                                               sweepSpecify.loc[product, 'Monthly Step'])
        if upfrntCond:
            upfrntGridX[product, ] = np.arange(sweepSpecify.loc[product, 'Upfront Begin'],
                                               sweepSpecify.loc[product, 'Upfront End']
                                               + sweepSpecify.loc[product, 'Upfront Step'],
                                               sweepSpecify.loc[product, 'Upfront Step'])

    useProductsDf = ProductsDf.copy()
    useProductsDf['use_index'] = ProductsDf.index
    indices = pd.merge(useProductsDf, sweepSpecify, left_on='id', right_on='Product ID', how="inner")['use_index']

    if (sweepType == "Attribute Grid"):
        evalMatSIO = np.zeros((upfrntGridN, mnthlyGridN))
        evalMatARPU = np.zeros((upfrntGridN, mnthlyGridN))
        evalMatRevenue = np.zeros((upfrntGridN, mnthlyGridN))
        evalMatEBIT = np.zeros((upfrntGridN, mnthlyGridN))

        for i in range(upfrntGridN):
            for j in range(mnthlyGridN):

                useProductsDf = ProductsDf.copy()
                for m, id in enumerate(indices):
                    useProductsDf.at[id, 'monthly_cost'] = mnthlyGridX[m, j]
                    useProductsDf.at[id, 'upfront_cost'] = upfrntGridX[m, i]

                augmentProductsDf = createAugmentedProductsDf(useProductsDf, dictCatAttributes)
                utilityLease = interpolateUtility(utilityDf, augmentProductsDf, dictNumAttributes, True, list(indices),
                                                  combineMat)
                offerMatching, personProdMatch = OfferMatching(augmentProductsDf, utilityDf, useProductsDf,
                                                               utilityLease, list(indices), personProductUtilityMat,
                                                               baselineDf)
                simResult = simulatorOutput(augmentProductsDf, offerMatching)
                live, numDynamic, churnOutSIO, churn_cohort_df = Simulate(simResult, marketAssumptions)

                brand = useProductsDf.at[id, 'brand']
                if brand in ["Audi", "VW"]:
                    indOutput = live[live['brand'].isin(["Audi", "VW"])].index.values
                    evalMatSIO[i, j] = live.loc[indOutput, 'SIO_new'].sum()
                    evalMatARPU[i, j] = live.loc[indOutput, 'ARPU_new'].sum()
                    evalMatRevenue[i, j] = live.loc[indOutput, 'revenue_new'].sum()
                    evalMatEBIT[i, j] = live.loc[indOutput, 'EBIT_new'].sum()

                    indAudi = live[live['brand'] == "Audi"].index.values
                    indVW = live[live['brand'] == "VW"].index.values
                    wghtAudi = live.loc[indAudi, 'SIO_new'].sum() / live.loc[indOutput, 'SIO_new'].sum()
                    wghtVW = live.loc[indVW, 'SIO_new'].sum() / live.loc[indOutput, 'SIO_new'].sum()
                    evalMatARPU[i, j] = (wghtAudi * live.loc[indAudi, 'ARPU_new'].sum()) \
                                        + (wghtVW * live.loc[indVW, 'ARPU_new'].sum())
                else:
                    indOutput = live[live['brand'] == brand].index.values
                    evalMatSIO[i, j] = live.loc[indOutput, 'SIO_new'].sum()
                    evalMatARPU[i, j] = live.loc[indOutput, 'ARPU_new'].sum()
                    evalMatRevenue[i, j] = live.loc[indOutput, 'revenue_new'].sum()
                    evalMatEBIT[i, j] = live.loc[indOutput, 'EBIT_new'].sum()

                #print([[i, j], [evalMatSIO[i, j], evalMatARPU[i, j], evalMatRevenue[i, j], evalMatEBIT[i, j]]])


        iBaseMnthly = np.where(mnthlyGridX == sweepSpecify.loc[k, 'Monthly Baseline'])[1][0]
        iBaseUpfrnt = np.where(upfrntGridX == sweepSpecify.loc[k, 'Upfront Baseline'])[1][0]
        SIObase = evalMatSIO[iBaseUpfrnt, iBaseMnthly]
        ARPUbase = evalMatARPU[iBaseUpfrnt, iBaseMnthly]
        RevenueBase = evalMatRevenue[iBaseUpfrnt, iBaseMnthly]
        EBITbase = evalMatEBIT[iBaseUpfrnt, iBaseMnthly]

        resultDf = pd.concat([pd.DataFrame(evalMatSIO), pd.DataFrame(evalMatARPU),
                              pd.DataFrame(evalMatRevenue), pd.DataFrame(evalMatEBIT)], axis=1)
        resultDf.index = upfrntGridX[0, :].astype(int)
        resultDf.columns = (["SIO_" + str(int(x)) for x in mnthlyGridX[0, :]]
                            + ["ARPU_" + str(int(x)) for x in mnthlyGridX[0, :]]
                            + ["Revenue_" + str(int(x)) for x in mnthlyGridX[0, :]]
                            + ["EBIT_" + str(int(x)) for x in mnthlyGridX[0, :]])

        baseVal = [SIObase, ARPUbase, RevenueBase, EBITbase]

        return resultDf, baseVal

    elif (sweepType == "Lockstep"):

        if mnthlyCond:
            useGridX = mnthlyGridX
        elif upfrntCond:
            useGridX = upfrntGridX

        baseInds = np.array([np.where(useGridX[k, :] == sweepSpecify.loc[k, attribute + ' Baseline'])[0][0]
                             for k in range(sweepSpecify.shape[0])])

        ################################################################################################################

        useProductsDf = ProductsDf.copy()
        for i, id in enumerate(indices):
            useProductsDf.at[id, attLower + '_cost'] = useGridX[i, baseInds[i]]

        augmentProductsDf = createAugmentedProductsDf(useProductsDf, dictCatAttributes)
        utilityLease = interpolateUtility(utilityDf, augmentProductsDf, dictNumAttributes, True, list(indices),
                                          combineMat)
        offerMatching, personProdMatch = OfferMatching(augmentProductsDf, utilityDf, useProductsDf,
                                                       utilityLease, list(indices), personProductUtilityMat,
                                                       baselineDf)

        baselinePred = offerMatching[[("id", "id"), ("Brand", "Live"), ("Product", "Live")]]
        baselinePred.columns = ['id', 'Baseline Brand', 'Baseline Product']

        ################################################################################################################

        dfSIO = pd.DataFrame(columns=['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'] +
                                      [attLower + '_%s' % (s + 1) for s in range(len(indices))])
        dfARPU = pd.DataFrame(columns=['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'] +
                                       [attLower + '_%s' % (s + 1) for s in range(len(indices))])
        dfRevenue = pd.DataFrame(columns=['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'] +
                                          [attLower + '_%s' % (s + 1) for s in range(len(indices))])
        dfEBIT = pd.DataFrame(columns=['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'] +
                                       [attLower + '_%s' % (s + 1) for s in range(len(indices))])
        dfFlow = pd.DataFrame(columns=['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'] +
                                       [attLower + '_%s' % (s + 1) for s in range(len(indices))] + ['Segment'])

        countI = 0
        for j in range(useGridX.shape[1]):
            useProductsDf = ProductsDf.copy()
            for i, id in enumerate(indices):
                useProductsDf.at[id, attLower + '_cost'] = useGridX[i, j]

            augmentProductsDf = createAugmentedProductsDf(useProductsDf, dictCatAttributes)
            utilityLease = interpolateUtility(utilityDf, augmentProductsDf, dictNumAttributes, True, list(indices),
                                              combineMat)
            offerMatching, personProdMatch = OfferMatching(augmentProductsDf, utilityDf, useProductsDf,
                                                           utilityLease, list(indices), personProductUtilityMat,
                                                           baselinePred)  # update baseline pred to specified baseline

            simResult = simulatorOutput(augmentProductsDf, offerMatching)
            rawInflowsOutflows = InFlowsOutFlowsBase(simResult, useProductsDf, offerMatching)
            live, numDynamic, churnOutSIO, churn_cohort_df = Simulate(simResult, marketAssumptions)
            SIOplan, churnStat, SIObySegmentProvider, PercBySegmentProvider, MarketNumBySegmentProvider, \
            MarketNumByProvider = \
                InFlowsOutFlows(simResult, useProductsDf, rawInflowsOutflows, numDynamic, offerMatching, churnOutSIO)
            dfSIO.loc[j] = list(live['SIO_new'].values) + [useGridX[x, j] for x in range(len(baseInds))]
            dfARPU.loc[j] = list(live['ARPU_new'].values) + [useGridX[x, j] for x in range(len(baseInds))]
            dfRevenue.loc[j] = list(live['revenue_new'].values) + [useGridX[x, j] for x in range(len(baseInds))]
            dfEBIT.loc[j] = list(live['EBIT_new'].values) + [useGridX[x, j] for x in range(len(baseInds))]

            brand = useProductsDf[useProductsDf['id'] == sweepSpecify['Product ID'][0]]['brand'].values[0]
            for segment in segments:
                providerSIOin = PercBySegmentProvider[segment][brand]
                providerSIOout = PercBySegmentProvider[segment].loc[brand]
                providerSIObase = providerSIOout.sum()
                other = providerSIOout - providerSIOin
                other.loc[brand] = providerSIOin.sum()
                other = [x/providerSIObase for x in other]
                dfFlow.loc[countI] = list(other) + [useGridX[x, j] for x in range(len(baseInds))] + [segment]
                countI += 1

        # Note this last row inserted uses the base values whilst above inserted rows use new values
        dfSIO.loc[useGridX.shape[1]] = list(live['SIO_base'].values) \
                                       + [useGridX[x, baseInds[x]] for x in range(len(baseInds))]
        dfARPU.loc[useGridX.shape[1]] = list(live['ARPU_base'].values) \
                                        + [useGridX[x, baseInds[x]] for x in range(len(baseInds))]
        dfRevenue.loc[useGridX.shape[1]] = list(live['revenue_base'].values) \
                                           + [useGridX[x, baseInds[x]] for x in range(len(baseInds))]
        dfEBIT.loc[useGridX.shape[1]] = list(live['EBIT_base'].values) \
                                        + [useGridX[x, baseInds[x]] for x in range(len(baseInds))]

        resultDf = pd.concat([dfSIO, dfARPU, dfRevenue, dfEBIT], axis=1)
        resultDf.columns = (["SIO_" + x for x in dfSIO.columns] + ["ARPU_" + x for x in dfARPU.columns] +
                            ["Revenue_" + x for x in dfRevenue.columns] + ["EBIT_" + x for x in dfEBIT.columns])

        return resultDf, dfFlow


def plotFinancials(resultsDf, dfFlowLoad, provider, attribute, val, viewType, aggView=False, baseInds=None):

    attLower = attribute.lower()
    clrs = ['mediumblue', 'gold', 'teal', 'deepskyblue', 'red', 'purple', 'orange', 'deeppink']
    aggBrands = ['Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    segments = ['Millenial', 'Gen X', 'Baby Boomer']
    dfFlow = dfFlowLoad.copy()

    SIO_cols = [col for col in resultsDf.columns if 'SIO_' in col]
    dfSIO = resultsDf[SIO_cols].copy()
    dfSIO.columns = [x[len('SIO_'):] for x in SIO_cols]
    ARPU_cols = [col for col in resultsDf.columns if 'ARPU_' in col]
    dfARPU = resultsDf[ARPU_cols].copy()
    dfARPU.columns = [x[len('ARPU_'):] for x in ARPU_cols]
    Revenue_cols = [col for col in resultsDf.columns if 'Revenue_' in col]
    dfRevenue = resultsDf[Revenue_cols].copy()
    dfRevenue.columns = [x[len('Revenue_'):] for x in Revenue_cols]
    EBIT_cols = [col for col in resultsDf.columns if 'EBIT_' in col]
    dfEBIT = resultsDf[EBIT_cols].copy()
    dfEBIT.columns = [x[len('EBIT_'):] for x in EBIT_cols]

    # the last row of df shows the 'base' (starting) values. Every other row shows 'new' (terminal) values
    dfSIObase = dfSIO.loc[dfSIO.shape[0] - 1]
    dfARPUbase = dfARPU.loc[dfARPU.shape[0] - 1]
    dfRevenueBase = dfRevenue.loc[dfRevenue.shape[0] - 1]
    dfEBITbase = dfEBIT.loc[dfEBIT.shape[0] - 1]

    dfSIO.drop(dfSIO.tail(1).index, inplace=True)
    dfARPU.drop(dfARPU.tail(1).index, inplace=True)
    dfRevenue.drop(dfRevenue.tail(1).index, inplace=True)
    dfEBIT.drop(dfEBIT.tail(1).index, inplace=True)

    if aggView:
        dfSIOold = dfSIO.copy()
        dfSIOold['denomVWgroup'] = dfSIOold['Audi'] + dfSIOold['VW']
        dfSIOold['wghtAudi'] = dfSIOold['Audi'] / dfSIOold['denomVWgroup']
        dfSIOold['wghtVW'] = dfSIOold['VW'] / dfSIOold['denomVWgroup']

        attCols = [col for col in dfEBIT.columns if attLower in col]
        dfSIOcopy = dfSIO[attCols].copy()
        dfSIOcopy['Chevrolet'] = dfSIO['Chevrolet'].copy()
        dfSIOcopy['Jaguar'] = dfSIO['Jaguar'].copy()
        dfSIOcopy['Kia'] = dfSIO['Kia'].copy()
        dfSIOcopy['Nissan'] = dfSIO['Nissan'].copy()
        dfSIOcopy['Tesla'] = dfSIO['Tesla'].copy()
        dfSIOcopy['Toyota'] = dfSIO['Toyota'].copy()
        dfSIOcopy['VW'] = dfSIO['Audi'] + dfSIO['VW']
        dfSIO = dfSIOcopy.copy()

        dfARPUcopy = dfARPU[attCols].copy()
        dfARPUcopy['Chevrolet'] = dfARPU['Chevrolet'].copy()
        dfARPUcopy['Jaguar'] = dfARPU['Jaguar'].copy()
        dfARPUcopy['Kia'] = dfARPU['Kia'].copy()
        dfARPUcopy['Nissan'] = dfARPU['Nissan'].copy()
        dfARPUcopy['Tesla'] = dfARPU['Tesla'].copy()
        dfARPUcopy['Toyota'] = dfARPU['Toyota'].copy()
        dfARPUcopy['VW'] = (dfSIOold['wghtAudi'] * dfARPU['Audi']) + \
                            (dfSIOold['wghtVW'] * dfARPU['VW'])
        dfARPU = dfARPUcopy.copy()

        dfRevenueCopy = dfRevenue[attCols].copy()
        dfRevenueCopy['Chevrolet'] = dfRevenue['Chevrolet'].copy()
        dfRevenueCopy['Jaguar'] = dfRevenue['Jaguar'].copy()
        dfRevenueCopy['Kia'] = dfRevenue['Kia'].copy()
        dfRevenueCopy['Nissan'] = dfRevenue['Nissan'].copy()
        dfRevenueCopy['Tesla'] = dfRevenue['Tesla'].copy()
        dfRevenueCopy['Toyota'] = dfRevenue['Toyota'].copy()
        dfRevenueCopy['VW'] = dfRevenue['Audi'] + dfRevenue['VW']
        dfRevenue = dfRevenueCopy.copy()

        dfEBITcopy = dfEBIT[attCols].copy()
        dfEBITcopy['Chevrolet'] = dfEBIT['Chevrolet'].copy()
        dfEBITcopy['Jaguar'] = dfEBIT['Jaguar'].copy()
        dfEBITcopy['Kia'] = dfEBIT['Kia'].copy()
        dfEBITcopy['Nissan'] = dfEBIT['Nissan'].copy()
        dfEBITcopy['Tesla'] = dfEBIT['Tesla'].copy()
        dfEBITcopy['Toyota'] = dfEBIT['Toyota'].copy()
        dfEBITcopy['VW'] = dfEBIT['Audi'] + dfEBIT['VW']
        dfEBIT = dfEBITcopy.copy()

    if (viewType == "Consolidated + Initial Baseline") or (viewType == "Individual + Initial Baseline"):

        if viewType == "Consolidated + Initial Baseline":
            dfSIObase['VW'] = dfSIObase['Audi'] + dfSIObase['VW']
            dfSIObase = pd.Series(dfSIObase[attCols + aggBrands])
            dfSIObase.index = attCols + aggBrands

            dfRevenueBase['VW'] = dfRevenueBase['Audi'] + dfRevenueBase['VW']
            dfRevenueBase = pd.Series(dfRevenueBase[attCols + aggBrands])
            dfRevenueBase.index = attCols + aggBrands

            dfEBITbase['VW'] = dfEBITbase['Audi'] + dfEBITbase['VW']
            dfEBITbase = pd.Series(dfEBITbase[attCols + aggBrands])
            dfEBITbase.index = attCols + aggBrands

            dfARPUbase['VW'] = dfRevenueBase['VW'] / dfSIObase['VW']
            dfARPUbase = pd.Series(dfARPUbase[attCols + aggBrands])
            dfARPUbase.index = attCols + aggBrands

        dfSIO = dfSIO - dfSIObase
        dfARPU = dfARPU - dfARPUbase
        dfRevenue = dfRevenue - dfRevenueBase
        dfEBIT = dfEBIT - dfEBITbase
    else:
        dfSIO = dfSIO - dfSIO.loc[baseInds[0]]
        dfARPU = dfARPU - dfARPU.loc[baseInds[0]]
        dfRevenue = dfRevenue - dfRevenue.loc[baseInds[0]]
        dfEBIT = dfEBIT - dfEBIT.loc[baseInds[0]]

    dfSIO = dfSIO.round(3)
    dfARPU = dfARPU.round(3)
    dfRevenue = dfRevenue.round(3)
    dfEBIT = dfEBIT.round(3)

    if (attLower + '_1') not in dfFlow.columns:
        return None, None, None, None, None, None, None, None
    elif len(dfFlow[attLower + '_1'].unique()) <= 1:
        return None, None, None, None, None, None, None, None

    if attribute == 'Monthly':
        xText = ["-" + "$" + str(round(abs(x))) + 'p/m' if x < 0 else "+" + "$" + str(round(x)) + 'p/m'
                                for x in dfARPU[attLower + '_1']]
    elif attribute == 'Upfront':
        xText = ["-" + "$" + str(round(abs(x))) if x < 0 else "+" + "$" + str(round(x))
                                for x in dfARPU[attLower + '_1']]

    if aggView:
        ChevroletBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Chevrolet'], name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['VW'], name='VW', marker_color=clrs[7])
        dataUse = [ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]
    else:
        AudiBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Audi'], name='Audi', marker_color=clrs[0])
        ChevroletBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Chevrolet'], name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfSIO[attLower + '_1'], y=dfSIO['VW'], name='VW', marker_color=clrs[7])
        dataUse = [AudiBars, ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]

    SIOfigure = {
        'data': dataUse,
        'layout': go.Layout(
            hovermode="closest",
            xaxis={'title': attribute + " Change", 'titlefont': {'color': 'black', 'size': 14},
                   'tickmode': 'array',
                   'tickvals': dfARPU[attLower + '_1'],
                   'ticktext': xText,
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "SIOs Change (Thousands)", 'titlefont': {'color': 'black', 'size': 14, },
                   'ticksuffix': 'K',
                   'tickfont': {'color': 'black'}}
        )
    }

    if aggView:
        ChevroletBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Chevrolet'], name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['VW'], name='VW', marker_color=clrs[7])
        dataUse = [ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]
    else:
        AudiBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Audi'], name='Audi', marker_color=clrs[0])
        ChevroletBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Chevrolet'], name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfARPU[attLower + '_1'], y=dfARPU['VW'], name='VW', marker_color=clrs[7])
        dataUse = [AudiBars, ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]

    ARPUfigure = {
        'data': dataUse,
        'layout': go.Layout(
            hovermode="closest",
            xaxis={'title': attribute + " Change", 'titlefont': {'color': 'black', 'size': 14},
                   'tickmode': 'array',
                   'tickvals': dfARPU[attLower + '_1'],
                   'ticktext': xText,
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "ARPU Change", 'titlefont': {'color': 'black', 'size': 14, },
                   'tickprefix': '$',
                   'tickfont': {'color': 'black'}}
        )
    }

    if aggView:
        ChevroletBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Chevrolet'], name='Chevrolet',
                               marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['VW'], name='VW', marker_color=clrs[7])
        dataUse = [ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]
    else:
        AudiBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Audi'], name='Audi', marker_color=clrs[0])
        ChevroletBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Chevrolet'], name='Chevrolet',
                               marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfRevenue[attLower + '_1'], y=dfRevenue['VW'], name='VW', marker_color=clrs[7])
        dataUse = [AudiBars, ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]

    RevenueFigure = {
        'data': dataUse,
        'layout': go.Layout(
            hovermode="closest",
            xaxis={'title': attribute + " Change", 'titlefont': {'color': 'black', 'size': 14},
                   'tickmode': 'array',
                   'tickvals': dfRevenue[attLower + '_1'],
                   'ticktext': xText,
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "Revenue Change (per month)", 'titlefont': {'color': 'black', 'size': 14, },
                   'tickprefix': '$',
                   'ticksuffix': 'K',
                   'tickfont': {'color': 'black'}}
        )
    }

    if aggView:
        ChevroletBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Chevrolet'], name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['VW'], name='VW', marker_color=clrs[7])
        dataUse = [ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]
    else:
        AudiBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Audi'], name='Audi', marker_color=clrs[0])
        ChevroletBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Chevrolet'], name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Jaguar'], name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Kia'], name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Nissan'], name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Tesla'], name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['Toyota'], name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfEBIT['VW'], name='VW', marker_color=clrs[7])
        dataUse = [AudiBars, ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]

    EBITfigure = {
        'data': dataUse,
        'layout': go.Layout(
            hovermode="closest",
            xaxis={'title': attribute + " Change", 'titlefont': {'color': 'black', 'size': 14},
                   'tickmode': 'array',
                   'tickvals': dfEBIT[attLower + '_1'],
                   'ticktext': xText,
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "EBIT Change (per month)", 'titlefont': {'color': 'black', 'size': 14, },
                   'tickprefix': '$',
                   'ticksuffix': 'K',
                   'tickfont': {'color': 'black'}}
        )
    }

    segVec = [None for x in range(len(segments))]
    for i, segment in enumerate(segments):
        dfUse = dfFlow[dfFlow['Segment'] == segment].iloc[:, :8]
        dfUse[provider] = dfUse[provider] - 1
        attCols = [col for col in dfUse.columns if attLower in col]
        if aggView:
            dfUseCopy = dfUse[attCols].copy()
            dfUseCopy['Chevrolet'] = dfUse['Chevrolet'].copy()
            dfUseCopy['Jaguar'] = dfUse['Jaguar'].copy()
            dfUseCopy['Kia'] = dfUse['Kia'].copy()
            dfUseCopy['Nissan'] = dfUse['Nissan'].copy()
            dfUseCopy['Tesla'] = dfUse['Tesla'].copy()
            dfUseCopy['Toyota'] = dfUse['Toyota'].copy()
            dfUseCopy['VW'] = dfUse['Audi'] + dfUse['VW']
            dfUseCopy = dfUseCopy.round(4)
            dfUse = dfUseCopy.copy()
            ChevroletBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Chevrolet'], name='Chevrolet',
                                   marker_color=clrs[1])
            JaguarBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Jaguar'], name='Jaguar', marker_color=clrs[2])
            KiaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Kia'], name='Kia', marker_color=clrs[3])
            NissanBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Nissan'], name='Nissan', marker_color=clrs[4])
            TeslaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Tesla'], name='Tesla', marker_color=clrs[5])
            ToyotaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Toyota'], name='Toyota', marker_color=clrs[6])
            VWBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['VW'], name='VW', marker_color=clrs[7])
            dataUse = [ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]
        else:
            AudiBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Audi'], name='Audi', marker_color=clrs[0])
            ChevroletBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Chevrolet'], name='Chevrolet',
                                   marker_color=clrs[1])
            JaguarBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Jaguar'], name='Jaguar', marker_color=clrs[2])
            KiaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Kia'], name='Kia', marker_color=clrs[3])
            NissanBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Nissan'], name='Nissan', marker_color=clrs[4])
            TeslaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Tesla'], name='Tesla', marker_color=clrs[5])
            ToyotaBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['Toyota'], name='Toyota', marker_color=clrs[6])
            VWBars = go.Bar(x=dfEBIT[attLower + '_1'], y=dfUse['VW'], name='VW', marker_color=clrs[7])
            dataUse = [AudiBars, ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]


        segmentFigure = {
            'data': dataUse,
            'layout': go.Layout(
                hovermode="closest",
                title=segment,
                xaxis={'title': attribute + " Change", 'titlefont': {'color': 'black', 'size': 14},
                       'tickmode': 'array',
                       'tickvals': dfEBIT[attLower + '_1'],
                       'ticktext': xText,
                       'tickfont': {'size': 9, 'color': 'black'}},
                yaxis={'title': "Percentage Shift in Customers", 'titlefont': {'color': 'black', 'size': 14, },
                       'tickformat': ',.2%',
                       'tickfont': {'color': 'black'}}
            )
        }
        segVec[i] = segmentFigure

    if val is not None:
        if attLower == "monthly":
            val = int(val[1:len(val)-3])
        elif attLower == "upfront":
            val = int(val[1:])
        uniqXval = np.array(dfFlow[attLower + '_1'].unique())
        #print(uniqXval)
        indCalc = np.where(uniqXval == val)[0]

        # in case where grid has changed, need to update calculations before updating plots
        if len(indCalc) == 0:
            raise PreventUpdate

        index = indCalc[0]
    else:
        index = 0
    yVal = dfFlow[attLower + '_1'].unique()[index]
    xVal = [x for x in range(len(segments))]
    if aggView:
        attCols = [col for col in dfFlow.columns if attLower in col]
        dfFlowCopy = dfFlow[attCols].copy()
        dfFlowCopy['Chevrolet'] = dfFlow['Chevrolet'].copy()
        dfFlowCopy['Jaguar'] = dfFlow['Jaguar'].copy()
        dfFlowCopy['Kia'] = dfFlow['Kia'].copy()
        dfFlowCopy['Nissan'] = dfFlow['Nissan'].copy()
        dfFlowCopy['Tesla'] = dfFlow['Tesla'].copy()
        dfFlowCopy['Toyota'] = dfFlow['Toyota'].copy()
        dfFlowCopy['VW'] = dfFlow['Audi'] + dfFlow['VW']
        dfFlowCopy = dfFlowCopy.round(6)
        if provider == 'Audi':
            dfFlowCopy['VW'] = dfFlowCopy['VW'] - 1
        else:
            dfFlowCopy[provider] = dfFlowCopy[provider] - 1
        ChevroletBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Chevrolet'],
                               name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Jaguar'],
                            name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Kia'],
                         name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Nissan'],
                            name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Tesla'],
                           name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Toyota'],
                            name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['VW'],
                        name='VW', marker_color=clrs[7])
        dataUse = [ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]
    else:
        dfFlowCopy = dfFlow.copy()
        dfFlowCopy = dfFlowCopy.round(6)
        dfFlowCopy[provider] = dfFlowCopy[provider] - 1
        AudiBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Audi'],
                          name='Audi', marker_color=clrs[0])
        ChevroletBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Chevrolet'],
                               name='Chevrolet', marker_color=clrs[1])
        JaguarBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Jaguar'],
                            name='Jaguar', marker_color=clrs[2])
        KiaBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Kia'],
                         name='Kia', marker_color=clrs[3])
        NissanBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Nissan'],
                            name='Nissan', marker_color=clrs[4])
        TeslaBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Tesla'],
                           name='Tesla', marker_color=clrs[5])
        ToyotaBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['Toyota'],
                            name='Toyota', marker_color=clrs[6])
        VWBars = go.Bar(x=xVal, y=dfFlowCopy[dfFlowCopy[attLower + '_1'] == yVal]['VW'],
                        name='VW', marker_color=clrs[7])
        dataUse = [AudiBars, ChevroletBars, JaguarBars, KiaBars, NissanBars, TeslaBars, ToyotaBars, VWBars]

    segmentFigure = {
        'data': dataUse,
        'layout': go.Layout(
            hovermode="closest",
            xaxis={'title': attribute + " Change", 'titlefont': {'color': 'black', 'size': 14},
                   'tickmode': 'array',
                   'tickvals': xVal,
                   'ticktext': segments,
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "Percentage Shift in Customers", 'titlefont': {'color': 'black', 'size': 14, },
                   'tickformat': ',.2%',
                   'tickfont': {'color': 'black'}}
        )
    }

    return SIOfigure, ARPUfigure, RevenueFigure, EBITfigure, segVec[0], segVec[1], segVec[2], segmentFigure


app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id='intermediate-value-2', style={'display': 'none'}),
    html.Div(id='intermediate-value-3', style={'display': 'none'}),
    html.Div(id='intermediate-value-4', style={'display': 'none'}),
    html.Div(id='intermediate-value-5', style={'display': 'none'}),
    html.Div(id='intermediate-value-6', style={'display': 'none'}),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Product Attractiveness', value='tab-1'),
        dcc.Tab(label='Attribute Sweep', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                dcc.Markdown(d("""
                **Electic Vehicle Leases**\n
                Edit the portfolio of EV leases available in San Francisco for consumers and run the analysis.
                """)),
                dash_table.DataTable(
                    id='products-input',
                    columns=(
                        [{'id': p, 'name': p} for p in ProductsDfEdit.columns]
                    ),
                    data=[x for x in ProductsDfEdit.apply(PDtoDict, axis=1)],
                    editable=True
                ),
                html.Div([
                    html.H6(children="Brand Attraction", style={'textAlign': 'center'}),
                ], style={'padding': '850px 0px 0px'}),
                dcc.Dropdown(
                    id='brandBeauty',
                    options=[{'label': 'Major Brands', 'value': 'Major Brands'},
                             {'label': 'Minor Brands', 'value': 'Minor Brands'}],
                    value='Major Brands'
                ),
                html.Div([
                    html.H6(children="Brand Attrition", style={'textAlign': 'center'}),
                ], style={'padding': '800px 0px 0px'}),
                dcc.Dropdown(
                    id='brandAttrition',
                    options=[{'label': 'Audi', 'value': 'Audi'},
                             {'label': 'Chevrolet', 'value': 'Chevrolet'},
                             {'label': 'Jaguar', 'value': 'Jaguar'},
                             {'label': 'Kia', 'value': 'Kia'},
                             {'label': 'Nissan', 'value': 'Nissan'},
                             {'label': 'Tesla', 'value': 'Tesla'},
                             {'label': 'Toyota', 'value': 'Toyota'},
                             {'label': 'VW', 'value': 'VW'}],
                    value='Tesla'
                ),
                html.Div([
                    html.H6(children="---------------------------------", style={'textAlign': 'center'}),
                ], style={'padding': '700px 0px 0px'}),
            ],
            style={'width': '25%', 'float': 'left', 'display': 'inline-block',
                    'borderRight': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'padding': '10px 20px 20px'
                    }
            ),
            html.H3(children="Predicted Market Share by Product", style={'textAlign': 'center'}),
            html.Div([
                dcc.Graph(
                    id='product-histogram',
                    style={
                        'height': 600
                    }
                )
            ],
                style={'width': '70%', 'float': 'right', 'display': 'inline-block'}),
            html.H3(children="Strength of Attraction to Chosen Product", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='heatmap',
                        style={
                            'height': 600,
                            'padding': '20px 20px 20px'
                        }
                    )
                ],
                style={'width': '55%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='miniScatter',
                        style={
                            'height': 600,
                            'padding': '20px 20px 20px'
                        }
                    )
                ],
                style={'width': '43%', 'float': 'left', 'display': 'inline-block'}),
                html.H3(children="Strength of Attraction to Chosen Brand", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Graph(
                        id='brandFig',
                        style={
                            'height': 800,
                            'padding': '20px 20px 20px'
                        }
                    )
                ]),
                html.H3(children="Brand Scavengers", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Graph(
                        id='brandWar',
                        style={
                            'height': 800,
                            'padding': '20px 20px 20px'
                        }
                    )
                ])
            ],
            style={'width': '70%', 'float': 'right', 'display': 'inline-block'}),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.H3(children="Analysis Type"),
                html.Div([
                    dcc.RadioItems(
                        id='analysis-type',
                        options=[{'label': i, 'value': i} for i in ['Lockstep', 'Attribute Grid']],
                        value='Lockstep',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                style={'width': '25%', 'float': 'left', 'display': 'inline-block'}
                ),
                html.Div([
                    html.Button(id='submit-button', children='Submit')
                ],
                    style={'width': '50%', 'float': 'left', 'display': 'inline-block'}
                )
            ],
            style={
                   'padding': '20px 0px 50px'
                   }
            ),
            html.Div([
                html.Div([
                    dcc.RadioItems(id='analysisOptions2')
                ], style={'width': '25%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.RadioItems(id='analysisOptions')
                ], style={'width': '75%', 'float': 'left', 'display': 'inline-block'})
            ]),

            html.Hr(),

            html.H3(children="Parameter Sweep", style={'textAlign': 'left'}),
            html.Div(children="""
            Each row in the table below performs a price and/or a data sweep over the product identified 
            by "ID". We may then examine impact on financial measures and Inflows/Outflows."""),
            dash_table.DataTable(
              id='adding-rows-table', editable=True, row_deletable=True
            ),
            html.Button('Add Row', id='editing-rows-button', n_clicks=0),
            html.H3('Key Financial Measures', style={'textAlign': 'center'}),
            html.Div(id='contingent')
        ])


@app.callback(
    Output('analysisOptions', 'options'),
    [Input('analysis-type', 'value')]
)
def analysisOptions(typeAnalysis):
    if typeAnalysis == "Lockstep":
        return [{'label': i, 'value': i} for i in ['Individual + Defined Baseline', 'Consolidated + Defined Baseline',
                                                   'Individual + Initial Baseline', 'Consolidated + Initial Baseline']]
    elif typeAnalysis == "Attribute Grid":
        return [{'label': i, 'value': i} for i in ['Nominal', 'Percentage']]


@app.callback(
    Output('analysisOptions', 'value'),
    [Input('analysisOptions', 'options')]
)
def analysisOptionsValue(options):
    return options[0]['value']


@app.callback(
    Output('analysisOptions2', 'options'),
    [Input('analysis-type', 'value')]
)
def analysisOptions2(typeAnalysis):
    if typeAnalysis == "Lockstep":
        return [{'label': i + ' cost', 'value': i} for i in ['Monthly', 'Upfront']]
    elif typeAnalysis == "Attribute Grid":
        return [{'label': i, 'value': i} for i in ['Monthly-Upfront']]


@app.callback(
    Output('analysisOptions2', 'value'),
    [Input('analysisOptions2', 'options')]
)
def analysisOptionsValue2(options):
    return options[0]['value']


@app.callback(
    [Output('adding-rows-table', 'data'),
     Output('adding-rows-table', 'columns')],
    [Input('editing-rows-button', 'n_clicks'),
     Input('analysisOptions2', 'value')],
    [State('adding-rows-table', 'data'),
     State('adding-rows-table', 'columns')])
def add_row(n_clicks, attribute, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    elif n_clicks == 0:
        if attribute == 'Monthly-Upfront':
            columns = [{'name': 'Product ID', 'id': 'ID', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly Begin', 'id': 'MonthlyBegin', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly End', 'id': 'MonthlyEnd', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly Step', 'id': 'MonthlyStep', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly Baseline', 'id': 'MonthlyBase', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront Begin', 'id': 'UpfrontBegin', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront End', 'id': 'UpfrontEnd', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront Step', 'id': 'UpfrontStep', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront Baseline', 'id': 'UpfrontBase', 'deletable': False, 'renamable': False}]
            rows = [{'ID': 7, 'MonthlyBegin': 155, 'MonthlyEnd': 205, 'MonthlyStep': 10, 'MonthlyBase': 195,
                     'UpfrontBegin': 4540, 'UpfrontEnd': 5040, 'UpfrontStep': 100, 'UpfrontBase': 4940},
                    {'ID': 8, 'MonthlyBegin': 255, 'MonthlyEnd': 305, 'MonthlyStep': 10, 'MonthlyBase': 295,
                     'UpfrontBegin': 5340, 'UpfrontEnd': 5840, 'UpfrontStep': 100, 'UpfrontBase': 5740}]
        elif attribute == 'Monthly':
            columns = [{'name': 'Product ID', 'id': 'ID', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly Begin', 'id': 'MonthlyBegin', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly End', 'id': 'MonthlyEnd', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly Step', 'id': 'MonthlyStep', 'deletable': False, 'renamable': False},
                       {'name': 'Monthly Baseline', 'id': 'MonthlyBase', 'deletable': False, 'renamable': False}]
            rows = [{'ID': 9, 'MonthlyBegin': 399, 'MonthlyEnd': 499, 'MonthlyStep': 20, 'MonthlyBase': 399},
                    {'ID': 10, 'MonthlyBegin': 900, 'MonthlyEnd': 1000, 'MonthlyStep': 20, 'MonthlyBase': 900},
                    {'ID': 11, 'MonthlyBegin': 999, 'MonthlyEnd': 1199, 'MonthlyStep': 40, 'MonthlyBase': 999}]
        elif attribute == 'Upfront':
            columns = [{'name': 'Product ID', 'id': 'ID', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront Begin', 'id': 'UpfrontBegin', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront End', 'id': 'UpfrontEnd', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront Step', 'id': 'UpfrontStep', 'deletable': False, 'renamable': False},
                       {'name': 'Upfront Baseline', 'id': 'UpfrontBase', 'deletable': False, 'renamable': False}]
            rows = [{'ID': 7, 'UpfrontBegin': 3940, 'UpfrontEnd': 4740, 'UpfrontStep': 200, 'UpfrontBase': 4740},
                    {'ID': 8, 'UpfrontBegin': 4940, 'UpfrontEnd': 5740, 'UpfrontStep': 200, 'UpfrontBase': 5740}]

    return rows, columns


@app.callback(
    Output('contingent', 'children'),
    [Input('analysis-type', 'value')]
)
def contingentGraph(typeAnalysis):
    if typeAnalysis == 'Lockstep':
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='SIOgraph',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='RevenueGraph',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'})
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='ARPUgraph',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='EBITgraph',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'})
            ]),
            html.H3('Brand Attrition with Product Change', style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='flowA',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '32%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='flowB',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '32%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='flowC',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '32%', 'float': 'left', 'display': 'inline-block'}),
            ]),

            html.H3('Brand Attrition with Product Change (by Segment)', style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Dropdown(id='attributeVal')
                ], style={'width': '19%', 'display': 'inline-block', 'padding': '20px 750px 20px'}),
                dcc.Graph(
                    id='flowAll',
                    style={
                        'height': 600
                    }
                )
            ])
        ])
    else:
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='SIOgrid',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='RevenueGrid',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'})
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='ARPUgrid',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='EBITgrid',
                        style={
                            'height': 600
                        }
                    )
                ],
                    style={'width': '49%', 'float': 'left', 'display': 'inline-block'})
            ])
        ])


@app.callback(
    Output('attributeVal', 'options'),
    [Input('submit-button', 'n_clicks')],
    [State('adding-rows-table', 'data'),
     State('adding-rows-table', 'columns')]
)
def contingentDropdown(nclicks, rows, columns):
    if rows is None:
        raise PreventUpdate

    df = pd.DataFrame([[int(row.get(c['id'], None)) for c in columns] for row in rows],
                      columns=[c['name'] for c in columns])
    menuVal = np.arange(df.iat[0, 1], df.iat[0, 2] + df.iat[0, 3], df.iat[0, 3])
    if df.columns[1] == "Monthly Begin":
        optionsList = [{'label': "$" + str(x) + "p/m", 'value': "$" + str(x) + "p/m"} for x in menuVal]
    else:
        optionsList = [{'label': "$" + str(x), 'value': "$" + str(x)} for x in menuVal]

    return optionsList


@app.callback(
    Output('attributeVal', 'value'),
    [Input('attributeVal', 'options')])
def contingentDropdown2(optionsList):
    if optionsList is None:
        raise PreventUpdate

    return optionsList[0]['value']


@app.callback(
    Output('intermediate-value', 'children'),
    [Input('products-input', 'data'),
     Input('products-input', 'columns')])
def calculateOffer(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    useProductsDf = ProductsDf.copy()
    use2 = useProductsDf.copy()
    deltaInd = np.argwhere((df['monthly'].values != use2['monthly_cost'].values) |
                           (df['term'].values != use2['term'].values) |
                           (df['upfront'].values != use2['upfront_cost'].values))
    deltaInd = [item for sublist in deltaInd for item in sublist]
    for ind in deltaInd:
        useProductsDf.at[ind, 'monthly_cost'] = df.at[ind, 'monthly']
        useProductsDf.at[ind, 'term'] = df.at[ind, 'term']
        useProductsDf.at[ind, 'upfront_cost'] = df.at[ind, 'upfront']
    augmentProductsDf = createAugmentedProductsDf(useProductsDf, dictCatAttributes)
    utilityLease = interpolateUtility(utilityDf, augmentProductsDf, dictNumAttributes, True, deltaInd, combineMat)
    #utilityPriceData = interpolateUtility(augmentUtilityDf, augmentProductsDf, True, deltaInd, 'Baseline', combineMat)
    offerMatching, personProdMatch = OfferMatching(augmentProductsDf, utilityDf, useProductsDf, utilityLease,
                                                   deltaInd, personProductUtilityMat, baselineDf)
    colList = ['Prod' + str(x + 1) for x in range(offerMatching.shape[1] - 6)]
    offerMatching.columns = ['ID', 'Segment', 'BaseBrand', 'LiveBrand', 'BaseProduct', 'LiveProduct'] + colList
    return offerMatching.to_json(date_format='iso', orient='split')


@app.callback(
    Output('intermediate-value-2', 'children'),
    [Input('intermediate-value', 'children')])
def calculateProbs(jsonified_data):
    offerMatching = pd.read_json(jsonified_data, orient='split')
    personProductUtilityMatUse = offerMatching.iloc[:, 6:].values
    mat = np.exp(personProductUtilityMatUse[:, ~np.all(personProductUtilityMatUse == 0, axis=0)])
    scaleVec = mat.sum(axis=1)
    mat = mat / scaleVec[:, None]
    probVersion = offerMatching.copy()
    probVersion.iloc[:, 6:] = mat
    return probVersion.to_json(date_format='iso', orient='split')


@app.callback(
    Output('product-histogram', 'figure'),
    [Input('intermediate-value', 'children')])
def display_histogram(jsonified_data):
    nP = ProductsDf.shape[0]
    offerMatching  = pd.read_json(jsonified_data, orient='split')
    useProductsDf = ProductsDf.copy()
    df1 = pd.value_counts(offerMatching['LiveProduct']).to_frame().reset_index().sort_values('index')
    df1.columns = ['prod ID', 'freq']
    df1.index = df1['prod ID']
    df2 = pd.DataFrame({'prod ID': range(1, nP + 1, 1), 'freq': 0})
    df2.index = df2['prod ID']
    df2.freq = df1.freq
    df2 = df2.fillna(0)
    df2.columns = ['prod ID', 'freq']
    potX = ['e-tron', 'Bolt', 'I-Pace', 'Niro EV', 'Niro Plg', 'Optima', 'Leaf S', 'Leaf S+', 'Model 3', 'Model S',
            'Model X', 'Prius', 'e-Golf']
    trace = go.Bar(x=potX, y=df2['freq'], name="Predicted Market Share by Product",
                   marker={'color': useProductsDf['colours']})
    return {
        'data': [trace],
        'layout': go.Layout(
            hovermode="closest",
            xaxis={'title': "Products", 'titlefont': {'color': 'black', 'size': 14},
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "Number Predicted to Buy Product", 'titlefont': {'color': 'black', 'size': 14, },
                   'tickfont': {'color': 'black'}}
        )
    }


@app.callback(
    Output('heatmap', 'figure'),
    [Input('intermediate-value', 'children')])
def display_heatmap(jsonified_data):

    nP = ProductsDf.shape[0]
    providerInd = np.array([0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7])

    labelsSheet = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    colors = ['mediumblue', 'gold', 'teal', 'deepskyblue', 'red', 'purple', 'orange', 'deeppink']

    potX = ['e-tron', 'Bolt', 'I-Pace', 'Niro EV', 'Niro Plg', 'Optima', 'Leaf S', 'Leaf S+', 'Model 3', 'Model S',
            'Model X', 'Prius', 'e-Golf']

    offerMatching = pd.read_json(jsonified_data, orient='split')
    personProductUtilityMatUse = offerMatching.iloc[:, 6:].values
    mat = np.exp(personProductUtilityMatUse[:, ~np.all(personProductUtilityMatUse == 0, axis=0)])
    scaleVec = mat.sum(axis=1)
    mat = mat / scaleVec[:, None]
    indChoice = mat.argmax(axis=1)

    desirability = np.full((nP, nP), -9.)
    binary = np.full((nP, nP), 0.)
    for i in range(nP):
        if sum(indChoice == i) > 0:
            matProb = mat[indChoice == i, :].T
            for j in range(nP):
                if providerInd[i] == providerInd[j]:
                    desirability[i, j] = np.percentile(matProb[np.array(range(nP)) == j, :], 50)
                else:
                    desirability[i, j] = np.percentile(matProb[np.array(range(nP)) == j, :], 80)
                if desirability[i, j] > 0.15:
                    binary[i, j] = desirability[i, j]

    heat = go.Heatmap(z=list(np.round(binary, 4)), x=potX, y=potX, colorscale='Viridis')
    return {
        'data': [heat],
        'layout': go.Layout(
            title="Opportunities for Winning (and Losing) Market Share",
            hovermode="closest",
            xaxis={'title': "Competing Market Offers", 'titlefont': {'color': 'black', 'size': 14},
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "Predicted Market Offer chosen by Respondent", 'titlefont': {'color': 'black', 'size': 14},
                   'tickfont': {'size': 9, 'color': 'black'}},

            shapes=[{'type': 'line', 'x0': 0.5, 'y0': -0.5, 'x1': 0.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 1.5, 'y0': -0.5, 'x1': 1.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 2.5, 'y0': -0.5, 'x1': 2.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 5.5, 'y0': -0.5, 'x1': 5.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 7.5, 'y0': -0.5, 'x1': 7.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 10.5, 'y0': -0.5, 'x1': 10.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 11.5, 'y0': -0.5, 'x1': 11.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 0.5, 'x1': 12.5, 'y1': 0.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 1.5, 'x1': 12.5, 'y1': 1.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 2.5, 'x1': 12.5, 'y1': 2.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 5.5, 'x1': 12.5, 'y1': 5.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 7.5, 'x1': 12.5, 'y1': 7.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 10.5, 'x1': 12.5, 'y1': 10.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 11.5, 'x1': 12.5, 'y1': 11.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}}
                    ]
        )
    }


@app.callback(
    Output('miniScatter', 'figure'),
    [Input('intermediate-value-2', 'children'),
     Input('heatmap', 'clickData')])
def display_miniscatter(jsonified_data, clickData):
    potX = ['e-tron', 'Bolt', 'I-Pace', 'Niro EV', 'Niro Plg', 'Optima', 'Leaf S', 'Leaf S+', 'Model 3', 'Model S',
            'Model X', 'Prius', 'e-Golf']
    color_dict_seg = {'Millenial': 'darkgreen', 'Gen X': 'darkgoldenrod', 'Baby Boomer': 'dimgrey'}

    offerMatching = pd.read_json(jsonified_data, orient='split')
    indChoice = offerMatching['LiveProduct'] - 1

    if clickData is None:
        indX = 9  # starting slected co-ordinates on heatmap
        indY = 4
    else:
        indX = potX.index(clickData['points'][0]['x'])
        indY = potX.index(clickData['points'][0]['y'])

    indArr = list(np.where(indChoice == (indY + 0))[0])
    yVal = (offerMatching.iloc[indArr, (indX + 6)] / offerMatching.iloc[indArr, (indY + 6)])
    xVal = offerMatching.iloc[indArr, (indY + 6)]

    coloursUse = offerMatching.iloc[indArr, :]['Segment'].map(color_dict_seg)
    if indX == indY:
        returnFigure = ff.create_distplot([list(xVal)], ['distplot'], show_hist=False, show_rug=True)
        returnFigure['layout'].update(
            showlegend=False,
            xaxis=dict(title='Prob.(Choose Product)', tickformat=',.2%'),
            yaxis=dict(title='Density', tickformat=',.2f'),
        )
    else:
        returnFigure = {
            'data': [go.Scatter(
                x=xVal,
                y=yVal,
                text="Respondent ID: " + offerMatching['ID'].astype(str),
                customdata=offerMatching.iloc[indArr, :]['Segment'],
                mode='markers',
                marker={
                    'size': 15,
                    'opacity': 0.5,
                    'color': coloursUse,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )],
            'layout': go.Layout(
                xaxis={
                    'title': "Prob.(Choose Predicted Product)",
                    'tickformat': ',.1%'
                },
                yaxis={
                    'title': "Ratio of Probabilities (Column Product to Predicted Product)",
                    'tickformat': ',.1%'
                },
                margin={'l': 80, 'b': 50, 't': 10, 'r': 0},
                height=600,
                hovermode='closest'
            )}
    return returnFigure


@app.callback(
    [Output('brandFig', 'figure'),
     Output('brandWar', 'figure')],
    [Input('intermediate-value-2', 'children'),
     Input('brandBeauty', 'value'),
     Input('brandAttrition', 'value')])
def display_BrandKDE(jsonified_data, typeBrand, whichOne):
    offerMatching = pd.read_json(jsonified_data, orient='split')
    mat = offerMatching.iloc[:, 6:].values
    indChoice = offerMatching['LiveProduct'] - 1
    providerInd = np.array([0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7])
    brandInd = [0, 1, 2, 3, 4, 5, 6, 7]
    brandChoice = providerInd[indChoice]

    labelsSheet = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    colors = ['mediumblue', 'gold', 'teal', 'deepskyblue', 'red', 'purple', 'orange', 'deeppink']
    labels = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']

    if typeBrand == "Major Brands":
        majorData = [None for x in range(4)]
        iBr = 0
        clrs = [None for x in range(4)]
        for i in [1, 4, 5, 6]:
            matBrand = mat[brandChoice == i, :].T[providerInd == i, :].T
            matProb = matBrand.sum(axis=1)
            # logProb = np.log10(matProb / (1 - matProb))
            logProb = matProb
            majorData[iBr] = list(logProb)
            clrs[iBr] = colors[i]
            iBr += 1

        brandFigure = ff.create_distplot(majorData, ['Chevrolet', 'Nissan', 'Tesla', 'Toyota'], show_rug=True,
                                         show_hist=False, colors=clrs)
        brandFigure['layout'].update(
            xaxis=dict(title='Prob(Choose Brand in Legend | Predict Person choose Brand in Legend)', tickformat=',.2%'),
            yaxis=dict(title='Density', tickformat=',.3f'),
            title="Major Brands Attraction"
        )
    elif typeBrand == 'Minor Brands':
        minorData = [None for x in range(4)]
        iBr = 0
        clrs = [None for x in range(4)]
        for i in [0, 2, 3, 7]:
            matBrand = mat[brandChoice == i, :].T[providerInd == i, :].T
            matProb = matBrand.sum(axis=1)
            # logProb = np.log10(matProb / (1 - matProb))
            logProb = matProb
            minorData[iBr] = list(logProb)
            clrs[iBr] = colors[i]
            iBr += 1

        brandFigure = ff.create_distplot(minorData, ['Audi', 'Jaguar', 'Kia', 'VW'], show_rug=True,
                                         show_hist=False, colors=clrs)
        brandFigure['layout'].update(
            xaxis=dict(title='Prob.(Choose Brand)', tickformat=',.2%'),
            yaxis=dict(title='Density', tickformat=',.3f'),
            title="Minor Brands Attraction"
        )

    i = labelsSheet.index(whichOne)
    jRange = brandInd
    brandData = [None for x in jRange]
    clrs = [None for x in jRange]
    jBr = 0
    for j in jRange:
        matBrand = mat[brandChoice == i, :].T[providerInd == j, :].T
        matProb = matBrand.sum(axis=1)
        #logProb = np.log10(matProb / (1 - matProb))
        logProb = matProb
        brandData[jBr] = list(logProb)
        clrs[jBr] = colors[j]
        jBr += 1

    scavengeFigure = ff.create_distplot(brandData, labels, show_rug=True, show_hist=False, colors=clrs)
    scavengeFigure['layout'].update(
        xaxis=dict(title='Probability(Person chooses Brand in Legend | Predict Person chooses Brand in Title)',
                   tickformat=',.2%'),
        yaxis=dict(title='Density', tickformat=',.3f'),
        title=labelsSheet[i] + ' Attrition'
    )

    return brandFigure, scavengeFigure


@app.callback(
    [Output('intermediate-value-3', 'children'),
     Output('intermediate-value-4', 'children'),
     Output('intermediate-value-5', 'children'),
     Output('intermediate-value-6', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('analysisOptions2', 'value'),
     State('adding-rows-table', 'data'),
     State('adding-rows-table', 'columns'),
     State('analysis-type', 'value')])
def calculateOffer(nclicks, attribute, rows, columns, sweepType):
    fullCols = ['Monthly Begin', 'Monthly End', 'Monthly Step', 'Monthly Baseline',
                'Upfront Begin', 'Upfront End', 'Upfront Step', 'Upfront Baseline']
    if nclicks is None:
        raise PreventUpdate
    else:
        df = pd.DataFrame([[int(row.get(c['id'], None)) for c in columns] for row in rows],
                          columns=[c['name'] for c in columns])
        dfNew = pd.DataFrame(df['Product ID'])
        dfNew.columns = ['Product ID']
        for k in range(len(fullCols)):
            if fullCols[k] in df.columns:
                dfNew[fullCols[k]] = df[fullCols[k]]
            else:
                dfNew[fullCols[k]] = np.ones(df.shape[0])

        df = dfNew.copy()
        useProductsDf = ProductsDf.copy()
        provider = useProductsDf[useProductsDf['id'] == int(df['Product ID'].iloc[0])]['brand'].values[0]

        nProd = df.shape[0]

        if (attribute == 'Monthly') or (attribute == 'Monthly-Upfront'):
            gridN = len(np.arange(df.loc[0, 'Monthly Begin'], df.loc[0, 'Monthly End'] + df.loc[0, 'Monthly Step'],
                                  df.loc[0, 'Monthly Step']))
            gridX = np.zeros((nProd, gridN))
            for product in df['Product ID'].index:
                gridX[product, ] = np.arange(df.loc[product, 'Monthly Begin'], df.loc[product, 'Monthly End']
                                             + df.loc[product, 'Monthly Step'], df.loc[product, 'Monthly Step'])

            baseInds = np.array([np.where(gridX[k, :] == df.loc[k, 'Monthly Baseline'])[0][0]
                                 for k in range(nProd)])
        else:
            gridN = len(np.arange(df.loc[0, 'Upfront Begin'], df.loc[0, 'Upfront End'] + df.loc[0, 'Upfront Step'],
                                  df.loc[0, 'Upfront Step']))
            gridX = np.zeros((nProd, gridN))
            for product in df['Product ID'].index:
                gridX[product, ] = np.arange(df.loc[product, 'Upfront Begin'], df.loc[product, 'Upfront End']
                                             + df.loc[product, 'Upfront Step'], df.loc[product, 'Upfront Step'])

            baseInds = np.array([np.where(gridX[k, :] == df.loc[k, 'Upfront Baseline'])[0][0]
                                 for k in range(nProd)])

        if sweepType == "Lockstep":
            returnDf, dfFlow = sweepCalculate(df, attribute, sweepType)
            augVar = dfFlow.to_json(date_format='iso', orient='split')
        elif sweepType == "Attribute Grid":
            returnDf, baseVar = sweepCalculate(df, attribute, sweepType)
            augVar = baseVar

        return returnDf.to_json(date_format='iso', orient='split'), provider, baseInds, augVar


@app.callback(
    [Output('SIOgraph', 'figure'),
     Output('ARPUgraph', 'figure'),
     Output('RevenueGraph', 'figure'),
     Output('EBITgraph', 'figure'),
     Output('flowA', 'figure'),
     Output('flowB', 'figure'),
     Output('flowC', 'figure'),
     Output('flowAll', 'figure')],
    [Input('intermediate-value-3', 'children'),
     Input('intermediate-value-4', 'children'),
     Input('intermediate-value-5', 'children'),
     Input('intermediate-value-6', 'children'),
     Input('analysisOptions', 'value'),
     Input('attributeVal', 'value')],
    [State('analysisOptions2', 'value'),
     State('analysis-type', 'value')])
def display_output_Lockstep(jsonified_data, provider, baseInds, jsonified_flow, viewType, val, attribute, analysisType):
    if jsonified_flow is None:
        raise PreventUpdate
    else:

        if type(jsonified_flow) == list:
            raise PreventUpdate

        returnDf = pd.read_json(jsonified_data, orient='split')
        dfFlow = pd.read_json(jsonified_flow, orient='split')
        if (viewType == "Consolidated + Defined Baseline") or (viewType == "Consolidated + Initial Baseline"):
            aggView = True
        elif (viewType == "Individual + Defined Baseline") or (viewType == "Individual + Initial Baseline"):
            aggView = False

        if analysisType == "Attribute Grid":
            aggView = False

        figData = plotFinancials(returnDf, dfFlow, provider, attribute, val, viewType, aggView, baseInds)
        if figData[0] is None:
            raise PreventUpdate

        return figData


@app.callback(
    [Output('SIOgrid', 'figure'),
     Output('ARPUgrid', 'figure'),
     Output('RevenueGrid', 'figure'),
     Output('EBITgrid', 'figure')],
    [Input('intermediate-value-3', 'children'),
     Input('intermediate-value-6', 'children'),
     Input('analysisOptions', 'value')])
def display_grid(jsonified_data, baseVar, viewType):
    if jsonified_data is None:
        raise PreventUpdate
    else:
        returnDf = pd.read_json(jsonified_data, orient='split')
        yVal = returnDf.index

        SIOcols = [col for col in returnDf.columns if "SIO" in col]

        if not is_number(SIOcols[0][4:]):
            raise PreventUpdate

        if (baseVar is None) or (type(baseVar) != list):
            raise PreventUpdate

        xSIO = [int(x[4:]) for x in SIOcols]
        ARPUcols = [col for col in returnDf.columns if "ARPU" in col]
        xARPU = [int(x[5:]) for x in ARPUcols]
        RevenueCols = [col for col in returnDf.columns if "Revenue" in col]
        xRevenue = [int(x[8:]) for x in RevenueCols]
        EBITcols = [col for col in returnDf.columns if "EBIT" in col]
        xEBIT = [int(x[5:]) for x in EBITcols]

        if viewType == "Nominal":
            evalMatSIO = np.round(returnDf[SIOcols].values - baseVar[0], 4)
            evalMatARPU = np.round(returnDf[ARPUcols].values - baseVar[1], 4)
            evalMatRevenue = np.round(returnDf[RevenueCols].values - baseVar[2], 3)
            evalMatEBIT = np.round(returnDf[EBITcols].values - baseVar[3], 3)

            strSIO = "SIO Change (Thousands)"
            strARPU = "ARPU Change ($)"
            strRevenue = "Revenue Change ($ thousands per month)"
            strEBIT = "EBIT Change ($ thousands per month)"

            hoverSIO = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: %{z}K<extra></extra>'
            hoverARPU = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: $%{z}<extra></extra>'
            hoverRevenue = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: $%{z}K<extra></extra>'
            hoverEBIT = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: $%{z}K<extra></extra>'
        elif viewType == "Percentage":
            evalMatSIO = np.round(((returnDf[SIOcols].values - baseVar[0]) / baseVar[0]) * 100, 2)
            evalMatARPU = np.round(((returnDf[ARPUcols].values - baseVar[1]) / baseVar[1]) * 100, 2)
            evalMatRevenue = np.round(((returnDf[RevenueCols].values - baseVar[2]) / baseVar[2]) * 100, 2)
            evalMatEBIT = np.round(((returnDf[EBITcols].values - baseVar[3]) / baseVar[3]) * 100, 2)

            strSIO = "SIO Relative Change (Percentage)"
            strARPU = "ARPU Relative Change (Percentage)"
            strRevenue = "Revenue Relative Change (Percentage)"
            strEBIT = "EBIT Relative Change (Percentage)"

            hoverSIO = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: %{z}%<extra></extra>'
            hoverARPU = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: %{z}%<extra></extra>'
            hoverRevenue = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: %{z}%<extra></extra>'
            hoverEBIT = '<i>y</i>: %{y}<br>' + '<i>x</i>: %{x}<br>' + '<i>z</i>: %{z}%<extra></extra>'

        heatSIO = {
            'data': [go.Heatmap(z=list(evalMatSIO), x=xSIO, y=yVal, colorscale='Viridis', hovertemplate=hoverSIO)],
            'layout': go.Layout(
                title=strSIO,
                hovermode="closest",
                xaxis={'title': "Monthly Cost (on First Model)", 'titlefont': {'color': 'black', 'size': 14},
                       'tickprefix': '$', 'ticksuffix': 'p/m', 'tickfont': {'size': 9, 'color': 'black'}},
                yaxis={'title': "Upfront Cost (on First Product)",
                       'tickprefix': '$',
                       'titlefont': {'color': 'black', 'size': 14},
                       'tickfont': {'size': 9, 'color': 'black'}},
                #coloraxis={'ticksuffix': 'M'}
            )
        }

        heatARPU = {
            'data': [go.Heatmap(z=list(evalMatARPU), x=xARPU, y=yVal, colorscale='Viridis', hovertemplate=hoverARPU)],
            'layout': go.Layout(
                title=strARPU,
                hovermode="closest",
                xaxis={'title': "Monthly Cost (on First Product)", 'titlefont': {'color': 'black', 'size': 14},
                       'tickprefix': '$', 'ticksuffix': 'p/m', 'tickfont': {'size': 9, 'color': 'black'}},
                yaxis={'title': "Upfront Cost (on First Product)",
                       'tickprefix': '$',
                       'titlefont': {'color': 'black', 'size': 14},
                       'tickfont': {'size': 9, 'color': 'black'}}
            )
        }

        heatRevenue = {
            'data': [go.Heatmap(z=list(evalMatRevenue), x=xRevenue, y=yVal,
                                colorscale='Viridis', hovertemplate=hoverRevenue)],
            'layout': go.Layout(
                title=strRevenue,
                hovermode="closest",
                xaxis={'title': "Monthly Cost (on First Product)", 'titlefont': {'color': 'black', 'size': 14},
                       'tickprefix': '$', 'ticksuffix': 'p/m', 'tickfont': {'size': 9, 'color': 'black'}},
                yaxis={'title': "Upfront Cost (on First Product)",
                       'tickprefix': '$',
                       'titlefont': {'color': 'black', 'size': 14},
                       'tickfont': {'size': 9, 'color': 'black'}}
            )
        }

        heatEBIT = {
            'data': [go.Heatmap(z=list(evalMatEBIT), x=xEBIT, y=yVal, colorscale='Viridis', hovertemplate=hoverEBIT)],
            'layout': go.Layout(
                title=strEBIT,
                hovermode="closest",
                xaxis={'title': "Monthly Cost (on First Product)", 'titlefont': {'color': 'black', 'size': 14},
                       'tickprefix': '$', 'ticksuffix': 'p/m', 'tickfont': {'size': 9, 'color': 'black'}},
                yaxis={'title': "Upfront Cost (on First Product)",
                       'tickprefix': '$',
                       'titlefont': {'color': 'black', 'size': 14},
                       'tickfont': {'size': 9, 'color': 'black'}}
            )
        }

        return heatSIO, heatARPU, heatRevenue, heatEBIT


if __name__ == '__main__':
    app.run_server(debug=True)