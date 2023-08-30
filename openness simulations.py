# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:14:57 2022

@author: Jake Barrett
"""
pd.options.mode.chained_assignment = None  # default='warn'
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# library installs
import os
import numpy as np
import pandas as pd
import random
from random import seed
seed(1)
import matplotlib.pyplot as plt
#random.Random(3)

# SIMULATION STEPS

# 1 initialise parameters

# number of strong nodes - fixed at 20%
n_i = 100
n_j=10
#power_prop = 0.2

# allocation paradigm - fixed, random, opt
allocation = 'fixed'
allocation = 'random'
allocation = 'opt'

# parameters for rules - O, G, p
O = np.round(n_j/2) # group sway
# rho: parameter for importance of diversity
rho = 0.5

# number of rounds
T = 10

# set global neighbourhood for bounded confidence
nhood = 2.5

# initial opinions - currently random, adjusting this is a big to-do

opinions_0 = np.repeat(range(10),np.repeat(10,10),axis=0)
random.Random(3).shuffle(opinions_0)
opinions_base = pd.DataFrame(opinions_0,columns=["opinions_0"])

# initial openness distribution
opm_trigger = 0.8
clm_trigger = 0.2
    # modelling: once openness reaches a threshold, bucket is full (can also subtract from openness)
    # is there a reasonable distribution we can use for OPM? Then can simply adjust parameters

# we want a distribution where: some individuals can have value already above the threshold, but most 
# participants are closed-minded
np.random.seed(0)
# opm_0 = np.random.beta(2,4,n_i) # CONTINUOUS CASE
prop_opm = 0.1

# 2 create dataset from parameters (using code from IP - add in randomised OPM attribute)
    # 2a read in optimised protocol
os.chdir("C:\\Users\\Jake\\Documents\\PhD\\OPM")
opt_protocol = pd.read_csv('100_20_200_protocol.csv')
# Data generated through table allocation code - single demographic (a1 or a2), with proportions 0.2 and 0.8


data = create_data(opt_protocol,5,prop_opm,opinions_base,'opt',['a'],False,0)
data = create_data(opt_protocol,200,0.1,opinions_base,'random',['a'],False,0)

# 3 create update rules: function taking a given table grouping and updating status

opinions_update = True
exp_included = True
# initialise expert - always presents evidence for 7 (to be updated)
exp_opinion = 7
exp_opinion_array = np.round(np.random.uniform(low=6,high=8,size=200))
exp_weight = 0.1
movement_speed = 0.1
mode = "DeGroot" # "bounded_confidence"
OPM_to_CM=True
prop_opm=0.1
O=7
rho=0.2
R=0.005
pt=0.01
O2=8.5
demog_cols=['a']
# define possible number of demographic subgroups
C = 2
table = opm_update(data,1,0,demog_cols,rho,C,O,
                   opinions_update,exp_included,exp_opinion,exp_weight,movement_speed,mode,
                   OPM_to_CM,pt,O2)



r1 = round_update(data,1,demog_cols,rho,C,O,opinions_update,exp_included,exp_opinion_array[0],exp_weight,movement_speed,mode,OPM_to_CM,pt,O2)
r2 = round_update(r1,2,demog_cols,rho,C,O,opinions_update,exp_included,exp_opinion_array[1],exp_weight,movement_speed,mode,OPM_to_CM,pt,O2)

# allocation paradigm - fixed, random, opt
a = sim_trial(opt_protocol,50,prop_opm,opinions_base,'opt',demog_cols,rho,C,O,
              opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
              OPM_to_CM,pt,O2,False,0)

double_plot(a,50,'opt',1,True,True,exp_asymptote)

extremists = True
extreme_index = random.sample(range(n_i),20)

n_iterations = 5
a1 = avg_over_trials(opt_protocol,20,prop_opm,opinions_base,allocation,demog_cols,rho,C,O,n_iterations,
                    opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
                    OPM_to_CM,pt,O2,
                    False,0)

double_plot(a1,50,'opt',n_iterations,True,True,exp_asymptote)

# 4 visualise output

# for each round, want info on: (a) how many are OPM, (b) how many because of prob, (c) how many because of G,
# (d) how many becaose of O

input_frame = a
T=20
allocation='opt'
n_iterations=1
scale_to_complete=True
exp_asymptote = [6,8]

double_plot(a1,20,'opt',5,True,True,exp_asymptote)

# Experts: random, central, extreme, switching between extremes periodically, none
n_iterations = 10
extreme_index = random.sample(range(n_i),20)
experts_random = np.random.choice(np.arange(0,10),size=200)
experts_central = np.array(random.choices([3,4,5,6],[0.125,0.375,0.375,0.125],k=200))
experts_extreme = np.repeat(8,200)
experts_periodic = np.tile(np.concatenate((np.repeat(1,5),np.repeat(8,5))),20)
# need to account for randomisation in CAS trials
for i in range(n_iterations): # a row for each iteration
    if i == 0:
        experts_random_update = [list(experts_random)]
        experts_extreme_update = [list(experts_extreme)]
        experts_periodic_update = [list(experts_periodic)]
    else:
        experts_random_update.append(list(experts_random))
        experts_extreme_update.append(list(experts_extreme))
        experts_periodic_update.append(list(experts_periodic))
      
# Participants: random opinions, bipartisan opinions,commited extremists
opinions_bimodal = opinions_base.copy()
opinions_bimodal['opinions_0'] = np.array(random.choices([0,1,2,7,8,9],k=n_i))
opinions_extreme = opinions_base.copy()
opinions_extreme.loc[extreme_index[0:10]]=0
opinions_extreme.loc[extreme_index[11:20]]=9

# initialise with sensible parameters
input_data = opt_protocol
prop_opm=0.1
T=200
rho=0.1
demog_cols = ['a']
# variable: prop_opm=0.1 (OPM True) or 1 (OPM False),
# variable: opinions_base=opinions_base, opinions_bimodal, opinions_extreme
allocation='opt'
R=0.005
pt=0.001
O=7
C=2 # two demographic levels in this example
n_iterations=10
opinions_update=True
exp_included = True
# variable: exp_weight = 0.1 (x_x_low_x_x) or 0.25 (x_x_high_x_x)
# variable: movement_speed = 0.1 (x_x_low_x_x) or 0.25 (x_x_high_x_x)
# variable: OPM_to_CM=True (x_x_x_x_true) or False (x_x_x_x_false)
O2=7
# variable: extremists = True (x_x_x_commit_x) or False otherwise
# variable: extreme_index = range(20) (x_x_x_commit_x) or 0 otherwise
# naming convention: opinion model_experts_exp_weight_initial opinions_opm used

paradigm_values = {'dg':"DeGroot",'bc':"bounded confidence"}
exp_values = {'random':experts_random_update,'extrem':experts_extreme_update,'periodic':experts_periodic_update}
participant_values = {'random':[opinions_base,False,0],'bipart':[opinions_bimodal,False,0],'extrem':[opinions_extreme,True,extreme_index]}
speed_values = {'slower':[0.075,0.075]}
opm_values = {'opm':True,'nopm':False}



trial = 0
for mode in paradigm_values:
    for exp in exp_values:
        for participant in participant_values:
            for speed in speed_values:
                for opm in opm_values:
                    string = str(mode)+"_"+str(exp)+"_"+str(participant)+"_"+str(speed)+"_"+str(opm)
                    print("test for trial " + str(trial) + ": " + str(string))
                    output = avg_over_trials(input_data,T,prop_opm,
                                             participant_values[participant][0],
                                             allocation,demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                                             exp_values[exp],speed_values[speed][0],speed_values[speed][1],paradigm_values[mode],
                                             opm_values[opm],
                                             pt,O2,
                                             participant_values[participant][1],participant_values[participant][2])
                    values = opinion_analysis(output,True,np.array(exp_values[exp][0]),n_iterations,participant_values[participant][2])
                    if trial == 0:
                        final_data = {string:values}
                    else:
                        final_data[string]=values
                    trial += 1

output_2 = pd.DataFrame(final_data).transpose().reset_index()

# why is there bimodality in periodic, bipart?
bimodes = avg_over_trials(input_data,T,prop_opm,opinions_bimodal,allocation,demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                          experts_periodic_update,0.075,0.075,'DeGroot',True,pt,O2,False,0)
double_plot(bimodes,T,'opt',n_iterations,True,True,experts_periodic_update[0])
opinion_analysis(bimodes,True,experts_periodic,n_iterations,extreme_index)

# investigation of multimodality
multi_modes = avg_over_trials(input_data,T,prop_opm,opinions_bimodal,allocation,demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                              experts_extreme,0.1,0.1,'DeGroot',True,0.02,O2,False,0)
double_plot(multi_modes,T,'opt',n_iterations,True,True,0)
opinion_analysis(multi_modes,True,experts_random,n_iterations,extreme_index)

# FIGURES FOR LATEX
# one bad trial - BC, extreme, bipartisan
latex_plot_bad = avg_over_trials(opt_protocol,200,0.1,opinions_bimodal,'opt',['a'],rho,C,O,n_iterations,True,True,
                                 experts_extreme_update,0.075,0.075,'bounded confidence',False,0.001,7,False,0)
double_plot(latex_plot_bad,200,'opt',n_iterations,True,True,experts_extreme_update[1])
# also want to see multimodality - run through opinion_analysis code
latex_plot_good = avg_over_trials(opt_protocol,200,0.1,opinions_bimodal,'opt',['a'],rho,C,O,n_iterations,True,True,
                                 experts_extreme_update,0.075,0.075,'DeGroot',True,0.001,7,False,0)
double_plot(latex_plot_good,200,'opt',n_iterations,True,True,experts_extreme_update[1])

#######
#NEW PLOTTING FUNCTION - want to display opening and closing distributions

# cycle through relevant parameters of interest
output = avg_over_trials(input_data,T,prop_opm,
                         participant_values['bipart'][0],
                         allocation,demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                         exp_values['periodic'],speed_values['slower'][0],speed_values['slower'][1],'bounded confidence',
                         True,
                         pt,O2,
                         participant_values['bipart'][1],participant_values['bipart'][2])

# want to visualise opinions at first and last stage
# need dataset of opinions at first and last stage

kde_plot(output,"Opinion Density Plot",True)

######
# Showing random allocation's effects

exp_values = {'random':experts_random_update,'extrem':experts_extreme_update,'periodic':experts_periodic_update}
participant_values = {'random':[opinions_base,False,0],'bipart':[opinions_bimodal,False,0],'extrem':[opinions_extreme,True,extreme_index]}

trial = 0
for exp in exp_values:
    for participant in participant_values:
        string = str(exp)+"_"+str(participant)
        print("test for trial " + str(trial) + ": " + str(string))
        output = avg_over_trials(input_data,T,prop_opm,
                                 participant_values[participant][0],
                                 'random',demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                                 exp_values[exp],0.075,0.075,'DeGroot',
                                 True,
                                 pt,O2,
                                 participant_values[participant][1],participant_values[participant][2])
        values = opinion_analysis(output,True,np.array(exp_values[exp][0]),n_iterations,participant_values[participant][2])
        if trial == 0:
            final_data_3 = {string:values}
        else:
            final_data_3[string]=values
        trial += 1

output_3 = pd.DataFrame(final_data_3).transpose().reset_index()

#######################
###### FIT TO CAS
#######################
# initial opinions come from population data - assume sampled accurately
os.chdir(".")
os.chdir('opinion formation and OPM/UKDA-8992-excel/excel')
os.listdir()
# factor analysis: identify latent variables in the data

# read in all data and label the weekend

# only need data that asks questions that are also in the panel data, and are tracked over time
pre_3=pd.read_csv('member_survey_10_shared_dataset_weekend_3_pre.csv')
pre_3['t']=4
post_3=pd.read_csv('member_survey_10_shared_dataset_weekend_3_post.csv')
post_3['t']=5
pre_4=pd.read_csv('member_survey_10_shared_dataset_weekend_4_pre.csv')
pre_4['t']=6
post_4=pd.read_csv('member_survey_10_shared_dataset_weekend_4_post.csv')
post_4['t']=7
popn_data_1 = pd.read_csv('population_survey_6_data_wave_1_csv_file.csv')
popn_data_2 = pd.read_csv('population_survey_6_data_wave_2_csv_file.csv')
demogs = pd.read_csv("member_survey_10_shared_dataset_demographics.csv")
expert_mapping = pd.read_csv('CAS expert mapping.csv')


# PART 1: create expert input - don't know when each presented, so randomise across weekend (subject to maintaining correct days)
# each of size n_iterations so can iterate through randomness

# reset column headers on expert_mapping
expert_mapping.columns = expert_mapping.iloc[4]
expert_mapping = expert_mapping.iloc[5:]
# collect cols
expert_use = pd.melt(expert_mapping, id_vars = expert_mapping.columns[0:5])
expert_use = expert_use.drop('t',axis=1)
expert_use=expert_use.rename(columns = {4:'t','value':'values'})
expert_values_dict = {}

#expert_value="att_01_w3"
for expert_value in np.unique(expert_use.Question):
    print(expert_value)
    # want two dictionary entries for each Q: variable for participants, and n_iterations*n_experts randomised arrays
    expert_redux = expert_use.loc[expert_use.Question==expert_value]
    expert_split = pd.concat([expert_redux['values'], expert_redux['t'].str.split(',', expand=True)], axis=1)
    expert_melt = pd.melt(expert_split,id_vars = 'values')[['values','value']]
    expert_melt.columns=['opinion','round']
    expert_melt = expert_melt.apply(pd.to_numeric)
    expert_melt = expert_melt.dropna(0)
    # create an array for each randomisation
    for iteration in range(n_iterations):
        opinion_array = []
        for t in range(int(max(expert_melt['round']))):
            t_use = t+1
            # randomly sample any opinion that could have been used in the round
            expert_t = expert_melt.loc[expert_melt['round']==t_use]
            #if expert_t.shape[0]==1:
            #    opinion_array.append(expert_t['opinion'][0])
            rand_array = expert_t['opinion']
            opinion_array.append(rand_array.sample(1).reset_index()['opinion'][0])
        # bind all arrays for final usage
        if iteration == 0:
            opinion_array_bind = [opinion_array]
        else:
            opinion_array_bind.append(opinion_array)
    expert_values_dict[expert_value]=opinion_array_bind
    
# PART 2: collect opinion data - target data for fitting

pre_3_opinion_data = pre_3[['ID_Code','att_01_W3', 'att_02_W3', 'att_03_W3', 'att_04_W3']]
pre_3_opinion_data.columns = ['ID_Code','att_01_W3', 'att_02_W3', 'att_03_W3', 'att_04_W3']
opinion_data = pd.melt(pre_3_opinion_data,id_vars='ID_Code')
opinion_data['t'] = 8

post_3_opinion_data = post_3[['ID_Code','att_01_W3', 'att_02_W3', 'att_03_W3', 'att_04_W3']]
post_3_opinion_data.columns = ['ID_Code','att_01_W3', 'att_02_W3', 'att_03_W3', 'att_04_W3']
post_3_opinion_data = pd.melt(post_3_opinion_data,id_vars='ID_Code')
post_3_opinion_data['t'] = 12
opinion_data = pd.concat([opinion_data,post_3_opinion_data])

pre_4_opinion_data = pre_4[['ID_Code', 'att_01_W4','att_02_W4', 'att_04_W4', 'att_05_W4']]
pre_4_opinion_data.columns = ['ID_Code', 'att_01_W4','att_02_W4', 'att_04_W4', 'att_05_W4']
pre_4_opinion_data = pd.melt(pre_4_opinion_data,id_vars='ID_Code')
pre_4_opinion_data['t'] = 12
opinion_data = pd.concat([opinion_data,pre_4_opinion_data])

post_4_opinion_data = post_4[['ID_Code', 'att_01_W4','att_02_W4', 'att_04_W4', 'att_05_W4']]
post_4_opinion_data.columns = ['ID_Code', 'att_01_W4','att_02_W4', 'att_04_W4', 'att_05_W4']
post_4_opinion_data = pd.melt(post_4_opinion_data,id_vars='ID_Code')
post_4_opinion_data['t'] = 16
opinion_data = pd.concat([opinion_data,post_4_opinion_data])

# initial opinions on questions - no demographic averages, so just randomise
question_lookup = expert_mapping[expert_mapping.columns[[0,2,3]]]

# want: for each question asked, what were the average and variance in opinions at time 0 and each respective t
all_responses = popn_data_1[question_lookup['Variable in panel data']]
all_responses.columns = question_lookup['Question']

processed_responses = pd.melt(all_responses)
# filter out 99s - should also default remove NAs
processed_responses = processed_responses.loc[processed_responses.value<99]
# map to 0-9 scale
processed_responses.value = 9/5*processed_responses.value
processed_responses = processed_responses.groupby('Question')['value'].agg(['mean','var']).reset_index()
processed_responses['t'] = 0
# reorder cols to commute
processed_responses.columns=['variable','mean','var','t']
processed_responses = processed_responses[['t','variable','mean','var']]

# also want processed opinions of participants
opinion_data['value'] = 9/5*opinion_data['value']
# remove 99s
opinion_data = opinion_data.loc[opinion_data.value<99]
processed_opinions = opinion_data.groupby(['t','variable'])['value'].agg(['mean','var']).reset_index()

opinion_comparisons = pd.concat([processed_responses,processed_opinions])


# PART 3: create an allocations schedule
# need to run the algorithm for these participants for T rounds to get the schedule
# stratify on age and gender (in member_survey_10_shared_dataset_demographics)
allocation = 'opt'
# stratification done via GroupSelect app - need to format output (already removed summaries from csvs)
for i in range(1,17):
    gs_out = pd.read_csv("CAS GS temp files/New File Results " + str(i) +".csv")
    group_format = gs_out.dropna(0).melt(id_vars = 'Person/Group')[['variable','value']]
    group_format.columns=['allocation_'+str(i),'id']
    # drop rows with no value
    group_format = group_format.loc[group_format['id']!="(empty)"]
    if i == 1:
        input_data = group_format
    else:
        input_data = pd.merge(input_data,group_format,how = "left",on="id")
    # merge with demographic_data
input_data = pd.merge(input_data,demogs[['ID_Code','age_WK1','gender_WK1']],how="left",left_on = "id",right_on="ID_Code")
# ensure consistent naming
input_data=input_data.rename(columns = {'age_WK1':'age','gender_WK1':'gender'})
allocation_cols = [col for col in input_data if col.startswith('allocation')]
input_data[allocation_cols] = input_data[allocation_cols].apply(pd.to_numeric)
# reindex table numbers from 0
input_data[allocation_cols] = input_data[allocation_cols].sub(1)


n_i = input_data.shape[0]
n_j = 12  # from CAS data

# PART 4: initialise opinions
# for each question, randomly sample from the relevant first survey answers
# remember to standardise opinions
opinions_dict = {}
for question in  np.unique(expert_use.Question):
    sample_data = all_responses[question]
    # standardise to 0-9 scale
    sample_data = sample_data[sample_data<99]
    sample_data = 9/5*sample_data
    # want to sample n_i of these
    sampled_data = np.random.choice(sample_data,size=n_i,replace=True)
    opinions_dict[question] = sampled_data


# PART 5: run the model TODO
# initialise standard parameters: 4 reallocations per weekend so 16 timesteps total
T = 16
demog_cols = ['age','gender']
opinions_update = True
exp_included = True
C = 12 # in data, 5 age groups and 2 genders so 10 distinct subgroups
n_iterations = 10
# parameters to be optimised
prop_opm = 0.1 # REVISIT THIS WITH EVIDENCE

# iterate through grid search and see which is closest
rho_list = [0.1] #0.2
O_list = [7] #7
movement_list = [0.075,0.1,0.25] #0.25
exp_list = [0.075,0.1,0.25] #0.25
pt_list = [0.001,0.01,0.05] #0.05
O2_list = [7] #8.5


#exp_value = 'att_04_W4'
test_trial = avg_over_trials(input_data,T,prop_opm,opinions_dict[exp_value],'opt',demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                         expert_values_dict[exp_value],exp_weight,movement_speed,'DeGroot',True,pt,O2,
                         False,0)
double_plot(test_trial,T,'opt',n_iterations,True,True,0)

grid_search_3 = {}
meta_trial = 0
for rho in rho_list:
    for O in O_list:
        for movement_speed in movement_list:
            for exp_weight in exp_list:
                for pt in pt_list:
                    for O2 in O2_list:
                        # iterate through grid search
                        save_string = str(rho)+"_"+str(O)+"_"+str(movement_speed)+"_"+str(exp_weight)+"_"+str(pt)+"_"+str(O2)
                        print("META TRIAL # " + str(meta_trial)+": "+str(save_string))
                        trial = 0
                        for exp_value in expert_values_dict:
                            string = str(exp_value)
                            print("test for trial " + str(trial) + ": " + str(string))
                            output = avg_over_trials(input_data,T,prop_opm,opinions_dict[exp_value],allocation,demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                                                         expert_values_dict[exp_value],exp_weight,movement_speed,'DeGroot',True,pt,O2,
                                                         False,0)
                            if trial == 0:
                                sim_data = {string:output}
                            else:
                                sim_data[string]=output
                            trial += 1
                        
                        # check no na
                        #double_plot(sim_data[string],T,'opt',n_iterations,True,True,0)
                        
                        # PART 6: model comparisons TODO
                        times = []
                        questions = []
                        means = []
                        variances = []
                        opinion_comparisons
                        for r in range(opinion_comparisons.shape[0]):
                            row = opinion_comparisons.iloc[r]
                            if row['t']!=0:
                                time = row['t']
                                question = row['variable']
                                sim_data_row = sim_data[question]
                                opinions_col = sim_data_row['opinions_'+str(time)]
                                times.append(time)
                                questions.append(question)
                                means.append(np.mean(opinions_col))
                                variances.append(np.var(opinions_col))
                        
                        simulated_comparisons = pd.DataFrame({'times':times,'questions':questions,'means':means,'variances':variances})
                        # how far away are the totals? save in a meta-frame
                        
                        # PART 7: parameter tuning TODO
                        # tune, investigate, report differences
                        grid_search_3[save_string]=simulated_comparisons
                        meta_trial+=1

trial = []
avg_abs_mean_diff = []
avg_abs_var_diff = []
for string in grid_search_3:
    trial.append(string)
    test_trial = grid_search_3[string]
    # how far is this from the averages?
    result_compare = pd.merge(processed_opinions,test_trial,left_on = ['t','variable'],right_on = ['times','questions'])[['t','variable','mean','means','var','variances']]
    result_compare['abs_mean_diff'] = abs(result_compare['mean']-result_compare['means'])
    result_compare['abs_var_diff'] = abs(result_compare['var']-result_compare['variances'])
    avg_abs_mean_diff.append(np.mean(result_compare['abs_mean_diff']))
    avg_abs_var_diff.append(np.mean(result_compare['abs_var_diff']))
    
grid_results_3 = pd.DataFrame({"trial":trial,"mean_diff":avg_abs_mean_diff,"var_diff":avg_abs_var_diff})
# split on trial
grid_results_3[['rho', 'O','movement','expert','pt','O2']] = grid_results_3['trial'].str.split('_', 5, expand=True)
# averages by each value?
grid_results_3.groupby('rho')['mean_diff','var_diff'].mean()
grid_results_3.groupby('O')['mean_diff','var_diff'].mean()
grid_results_3.groupby('movement')['mean_diff','var_diff'].mean()
grid_results_3.groupby('expert')['mean_diff','var_diff'].mean()
grid_results_3.groupby('pt')['mean_diff','var_diff'].mean()
grid_results_3.groupby('O2')['mean_diff','var_diff'].mean()


for i in range(n_iterations): # a row for each iteration
    if i == 0:
        experts_random_update = [list(experts_random)]
    else:
        experts_random_update.append(list(experts_random))
  

parameter_test = avg_over_trials(opt_protocol,200,prop_opm,
                         opinions_bimodal,
                         allocation,['a'],0.1,2,7,n_iterations,opinions_update,exp_included,
                         experts_random_update,0.075,0.075,'DeGroot',
                         True,
                         0.001,7,
                         False,0)

double_plot(parameter_test,T,'opt',n_iterations,True,True,0)
opinion_analysis(parameter_test,True,experts_random,n_iterations,0)

