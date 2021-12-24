import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# import various AL strategies
from Classes.active_learner import ActiveLearnerRandom
from Classes.active_learner import ActiveLearnerUncertainty
from Classes.active_learner import ActiveLearnerLAL
# import the dataset class
from Classes.dataset import DatasetMW
# import the model for LAL strategy
from Classes.lal_model import LALmodel
# import Experiment and Result classes that will be responsible for running AL and saving the results
from Classes.experiment import Experiment
from Classes.results import Results


# -----------------------------------------------
# model 1
if 'model 1':
    fn = 'LAL-randomtree-simulatedunbalanced-big.npz'
    # we found these parameters by cross-validating the regressor and now we reuse these expreiments
    parameters = {'est': 2000, 'depth': 40, 'feat': 6 }
    filename = './lal datasets/'+fn
    regression_data = np.load(filename)
    regression_features = regression_data['arr_0']
    regression_labels = regression_data['arr_1']

    print('Building lal regression model..')
    lalModel1 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
                                    max_features=parameters['feat'], oob_score=True, n_jobs=8)

    lalModel1.fit(regression_features, np.ravel(regression_labels))    

    print('Done!')
    print('Oob score = ', lalModel1.oob_score_)


# model 2
if 'model 2':
    fn = 'LAL-iterativetree-simulatedunbalanced-big.npz'
    # we found these parameters by cross-validating the regressor and now we reuse these expreiments
    parameters = {'est': 1000, 'depth': 40, 'feat': 6 }
    filename = './lal datasets/'+fn
    regression_data = np.load(filename)
    regression_features = regression_data['arr_0']
    regression_labels = regression_data['arr_1']

    print('Building lal regression model..')
    lalModel2 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
                                    max_features=parameters['feat'], oob_score=True, n_jobs=8)

    lalModel2.fit(regression_features, np.ravel(regression_labels))    

    print('Done!')
    print('Oob score = ', lalModel2.oob_score_)


'''
nExperiments: 实验次数
nStart: 初始化标签数（默认2，0、1类型各取1个）
nIterations: 每次实验迭代次数（每轮增加1个标签样本数）
'''
# number of experiment repeats
nExperiments = 5
# number of estimators (random trees) in the classifier
nEstimators = 50
# number of labeled points at the beginning of the AL experiment
nStart = 2
# number of iterations in AL experiment
nIterations = 2998
# the quality metrics computed on the test set to evaluate active learners
quality_metrics = ['TP', 'TN', 'FP', 'FN', 'accuracy', 'roc_auc', 'prauc', 'recall', 'precision']

# load dataset
dtst = DatasetMW()
# set the starting point
dtst.setStartState(nStart)
# Active learning strategies
alR = ActiveLearnerRandom(dtst, nEstimators, 'random')
alU = ActiveLearnerUncertainty(dtst, nEstimators, 'uncertainty')
alLALindepend = ActiveLearnerLAL(dtst, nEstimators, 'lal-rand', lalModel1)
alLALiterative = ActiveLearnerLAL(dtst, nEstimators, 'lal-iter', lalModel2)
als = [alR, alU, alLALindepend, alLALiterative]

exp = Experiment(nIterations, nEstimators, quality_metrics, dtst, als, 
                 'here we can put a comment about the current experiments')
# the Results class helps to add, save and plot results of the experiments
res = Results(exp, nExperiments)

print('Starting nExperiments' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for i in range(nExperiments):
    print('\nExperiments #' + str(i+1) + ': ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # run an experiment
    performance = exp.run()
    res.addPerformance(performance)
    # reset the experiment (including sampling a new starting state for the dataset)
    exp.reset()
print('\nEnding nExperiments' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

res.saveResults('mw_exp')

res2plot = Results()
res2plot.readResult('mw_exp')
res2plot.plotResults(metrics = ['accuracy'])
res2plot.plotResults(metrics = ['roc_auc'])
res2plot.plotResults(metrics = ['prauc'])
res2plot.plotResults(metrics = ['IoU'])
