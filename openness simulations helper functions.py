# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:17:53 2022

@author: Jake Barrett
"""
import os
import numpy as np
import pandas as pd
import random
from random import seed
seed(1)
import matplotlib.pyplot as plt
import seaborn as sns


#!pip install unidip
from unidip import UniDip
import unidip.dip as dip


def opm_status(row):
    if row['opm_0']<opm_trigger:
        val = 0
    else: 
        val = 1
    return val

def create_data(input_data,T,prop_opm,opinions_base,allocation,demog_cols,extremists,extreme_index):
    opt_rounds = input_data[[col for col in input_data if col.startswith('allocation')]]
        # 2b create base for fixed and random allocations - retain the same strong nodes
    base_data = input_data[['id']+demog_cols].copy()
        # 2c create table allocations
    # fixed allocations: need to partition once, then replicate - can just take first allocation of optimised
    for round in range(T):
        round_name = "allocation_"+str(round+1)
        if(allocation == "fixed"):
            base_data = pd.concat([base_data,opt_rounds['allocation_1']],axis=1)
            base_data.columns = [*base_data.columns[:-1], round_name]
        if(allocation == "random"):
            # want to shuffle vector of tables from allocation_1 in opt_protocol
            shuffle_data = opt_rounds['allocation_1'].copy()
            random.shuffle(shuffle_data)
            base_data = pd.concat([base_data,shuffle_data],axis=1)
            base_data.columns = [*base_data.columns[:-1], round_name]
        if(allocation == "opt"):
            base_data = pd.concat([base_data,opt_rounds[round_name]],axis=1)
        # 2d attach new variables based on set parameters - initial parameters, initial openness

    #base_data = pd.concat([base_data,opinions_base],axis=1)
    base_data['opinions_0'] = opinions_base
    opm_0 = random.choices([0,1],weights=[1-prop_opm,prop_opm],k=n_i)
    opm_base = pd.DataFrame(opm_0,columns=["opm_0"])
    base_data = pd.concat([base_data,opm_base],axis=1)
    # participants can have status 0 (closed, always been closed), 1 (open), and 2 (closed, having previously been open)
    
    #base_data['opm_status_0'] = base_data.apply(opm_status, axis=1) # CONTINUOUS CASE
    base_data['opm_status_0'] = base_data['opm_0']

    # also want to track when specific rules are being triggered
    base_data['first_O'] = 0
    base_data['first_rho'] = 0
    base_data['first_pt'] = 0
    base_data['first_O2'] = 0
    base_data['OPM_to_CM_trigger'] = 0
    if(extremists==True):
        # set the corresponding index of 'opm_to_cm_trigger' to 1: no change of opm/opinion
        base_data.loc[extreme_index,'OPM_to_CM_trigger'] = 1
    
    return base_data

# data = strict_reg_change, round_no=11,table_no=1,R=0.05,B=5,O=7,opinions_update=True,exp_included=True,exp_opinion=4.5,exp_weight=0.25,movement_speed=0.25

# need a diversity calculator for second OPM rule. For our simple example, demog_cols is simply ['a'], and as the allocation is optimised there
# is identical diversity per table per round
def diverse_calc(table,demog_cols,rho,C):
    # want to return a scalar value based on the diversity of the table
    P_values = table.groupby(demog_cols).count().reset_index()['prior_open']/table.shape[0] # prior_open used as an index - could use any col
    # using NGV diversity metric
    NGV = C*(1-sum([x**2 for x in P_values]))/(C-1)
    return rho*NGV


# testing pt rule
#data = sim_trial(opt_protocol,5,prop_opm,opinions_base,'opt',demog_cols,rho,C,O,
#              opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
#              OPM_to_CM,pt,O2,False,0)
#round_no = 1
#table_no = 1

def opm_update(data,round_no,table_no,demog_cols,rho,C,O,opinions_update,
               exp_included,exp_opinion,exp_weight,movement_speed,mode,
               OPM_to_CM,pt,O2):
    # Here we can define and re-define rules of OPM updating
    # function takes current OPM levels for a table, runs through rules, and outputs new OPM
    opm_status_col = 'opm_status_'+str(round_no-1)
    new_opm_status_col = 'opm_status_'+str(round_no)
    table_col = 'allocation_'+str(round_no)
    opinion_col = 'opinions_'+str(round_no-1)
    new_opinion_col = 'opinions_'+str(round_no)
    table=data.loc[data[table_col]==table_no][['id',table_col,opm_status_col,opinion_col,'first_rho','first_O','first_pt','first_O2','OPM_to_CM_trigger'] + demog_cols]
    table[new_opm_status_col] = table[opm_status_col]
    table['prior_open'] = table[opm_status_col]
    table[new_opinion_col] = table[opinion_col]
    # CRITERIA 1: encounter more than O who are already open
    # 1b: if already open, encounter more than O who have become closed (later)
    if OPM_to_CM: # used now for any OPM
        no_open = sum(table['prior_open'])
        if(no_open>=O):
            # those with status 0 reach status 1
            table.loc[(table[opm_status_col]==0)&(table['OPM_to_CM_trigger']==0),new_opm_status_col]=1
            table.loc[table['first_O']==0,'first_O']=round_no
        # CRITERIA 2: if on a diverse table, higher probability of becoming OPM
        rho_NGV = diverse_calc(table,demog_cols,rho,C)
        table['temp_prob']=random.choices([0,1],weights=[1-rho_NGV,rho_NGV],k=table.shape[0])
        # need to track if already triggered
        table.loc[table['first_rho']==0,'first_rho']=np.multiply(table.loc[table['first_rho']==0,'temp_prob'],round_no)
        table.loc[(table[opm_status_col]==0) & (table['temp_prob']==1)&(table['OPM_to_CM_trigger']==0),new_opm_status_col]=1
        
    if OPM_to_CM:
        # add 'become CM' rules
        no_open_to_closed = sum(table['OPM_to_CM_trigger'])
        # if they become opm, opm parameter needs a value - jump to max of [average of those opm, opm_trigger]
        if(no_open_to_closed >=O2):
            # those with status 0 reach status 1
            table.loc[table[opm_status_col]==1,new_opm_status_col]=0
            table.loc[(table[opm_status_col]==1)&(table['first_O2']==0),'first_O2']=round_no
        # CRITERIA 2: far from the average opinion
        # 2b: close to average opinion - need to exclude themselves from average
        # if they become closed, what influences their level of opm? TODO revisit this, currently set to trigger
        #table['avg_excl'] = (table[opinion_col].sum()-table[opinion_col])/(table.shape[0]-1)
        #table['dist_from_avg'] = abs(table[opinion_col]-table['avg_excl'])
        #table.loc[(table[opm_status_col]==1) & (table['dist_from_avg']<B),new_opm_status_col]=0
        #table.loc[(table[opm_status_col]==1) & (table['first_B']==0) & (table['dist_from_avg']<B),'first_B']=round_no
        # CRITERIA 2: small constant probability of becoming open IF NOT ALREADY CM (additive?)
        # how long have they been OPM?
        OPM_triggers = table[['first_O','first_rho']]
        table['OPM_time'] = round_no-np.where(OPM_triggers>0,OPM_triggers,round_no).min(axis=1)
        table['pt'] = np.array([1-(1/(1+pt*x)) for x in table['OPM_time']])
        table['temp_prob']=np.array([float(random.choices([1,0],weights=[1-pt,pt],k=1)[0]) for pt in table['pt']])
        # need to keep track if already triggered pt
        table.loc[(table[opm_status_col]==1) & (table['first_pt']==0),'first_pt']=np.multiply(1-table.loc[(table[opm_status_col]==1) & (table['first_pt']==0),'temp_prob'],round_no)
        # only transition if haven't already
        table.loc[(table[new_opm_status_col]==1)&(table[opm_status_col]==1),new_opm_status_col]=table.loc[(table[new_opm_status_col]==1)&(table[opm_status_col]==1),'temp_prob']
        # add flag if OPM_to_CM triggered
        table.loc[(table[opm_status_col]==1)&(table[new_opm_status_col]==0),'OPM_to_CM_trigger']=1
    if opinions_update:
        # update opinions according to rules
        table_new_opinions=opinion_update(table,round_no,table_no,exp_included,exp_opinion,exp_weight,movement_speed,mode)
        table = table[['id','first_rho','first_O','first_pt','first_O2',opm_status_col,new_opm_status_col,'OPM_to_CM_trigger']].merge(table_new_opinions,how='left',on='id')
    # return relevant cols
    return table[['id','first_rho','first_O','first_pt','first_O2',new_opm_status_col,new_opinion_col,'OPM_to_CM_trigger']]

# table_data = table
# table_data = r1.loc[r1.allocation_1==0]
# table_data = run_data_0.loc[(run_data_0.allocation_58==4)&(run_data_0.iteration==0)], exp_included=False,exp_weight=0.1,movement_speed=0.1
    
def opinion_update(table_data,round_no,table_no,exp_included,exp_opinion,exp_weight,movement_speed,mode):
    #print("updating opinions for expert opinion "+str(exp_opinion)) # these are randomly too high
    new_opm_status_col = 'opm_status_'+str(round_no)
    table_col = 'allocation_'+str(round_no)
    opinion_col = 'opinions_'+str(round_no-1)
    new_opinion_col = 'opinions_'+str(round_no)
    table=table_data[['id',table_col,new_opm_status_col,opinion_col]].copy()
    table[new_opinion_col] = table[opinion_col]
    
    # mix with expert - mixing parameter?
    # ADD IN MODE
    if mode=="DeGroot":
        # calculate average opinion (excluding agent)
        table['total_excl'] = table[opinion_col].sum()-table[opinion_col]
        table['avg_excl'] = table['total_excl']/(table.shape[0]-1)
        if exp_included:
            table['avg_with_exp']=exp_weight*exp_opinion+(1-exp_weight)*table['avg_excl']
            table.loc[table[new_opm_status_col]==1,new_opinion_col] = movement_speed*table.loc[table[new_opm_status_col]==1,'avg_with_exp']+(1-movement_speed)*table.loc[table[new_opm_status_col]==1,opinion_col]
        else:
            table.loc[table[new_opm_status_col]==1,new_opinion_col] = movement_speed*table.loc[table[new_opm_status_col]==1,'avg_excl']+(1-movement_speed)*table.loc[table[new_opm_status_col]==1,opinion_col]
    elif mode=="bounded confidence":
        nhood_sums = []
        nhood_counts = []
        # only takes average of weights in neighbourhood - defined globally
        # DEBUGGING: opinion = 1.058609
        for opinion in table[opinion_col]:
            nhood_frame = table.loc[(table[opinion_col]>=(opinion-nhood))&(table[opinion_col]<=(opinion+nhood)),opinion_col]
            nhood_sum = sum(nhood_frame)
            nhood_count = len(nhood_frame)
            # don't weight own opinion, unless there are no others in the neighbourhood
            if nhood_count >1:
                nhood_count-=1
                nhood_sum-=opinion
            nhood_sums.append(nhood_sum)
            nhood_counts.append(nhood_count)
        table['total_nhood_excl'] = nhood_sums
        table['n_nhood'] = nhood_counts
        table['avg_excl'] = table['total_nhood_excl']/(table['n_nhood'])
        if exp_included:
            # also need to modify expert - only listen if in neighbourhood
            table['expert_within_nhood']=np.where((table[opinion_col]>=(exp_opinion-nhood))&(table[opinion_col]<=(exp_opinion+nhood)),1,0)
            table['avg_with_exp']=table['expert_within_nhood']*(exp_weight*exp_opinion+(1-exp_weight)*table['avg_excl'])+(1-table['expert_within_nhood'])*table['avg_excl']
            table.loc[table[new_opm_status_col]==1,new_opinion_col] = movement_speed*table.loc[table[new_opm_status_col]==1,'avg_with_exp']+(1-movement_speed)*table.loc[table[new_opm_status_col]==1,opinion_col]
        else:
            table.loc[table[new_opm_status_col]==1,new_opinion_col] = movement_speed*table.loc[table[new_opm_status_col]==1,'avg_excl']+(1-movement_speed)*table.loc[table[new_opm_status_col]==1,opinion_col]
    else:
        print("error: mode not specified within DeGroot or bounded confidence (mode = "+str(mode)+")")
    return table[['id', new_opinion_col]]


round_no=1
table_no=0
def round_update(data,round_no,demog_cols,rho,C,O,
                 opinions_update,exp_included,exp_opinion,exp_weight,movement_speed,mode,
                 OPM_to_CM,pt,O2):
    #print("generating for round "+str(round_no))
    for table_no in range(n_j):
        #print("table "+str(table_no))
        new_opm = opm_update(data,round_no,table_no,demog_cols,rho,C,O,
                             opinions_update, exp_included, exp_opinion, exp_weight, movement_speed,mode,
                             OPM_to_CM,pt,O2)
        if(table_no == 0):
            new_frame = new_opm
        else:
            new_frame = pd.concat([new_frame,new_opm])
        #print("new table added "+str(table_no))
    # merge new info onto original info
    prior_data = data.drop(['first_rho','first_O','first_pt','first_O2','OPM_to_CM_trigger'], axis=1)
    post_data = pd.merge(prior_data,new_frame,how='left',on='id')
    #print("# OPM_to_CM_transitions in round "+str(round_no)+ " = " +str(sum(post_data.OPM_to_CM_trigger)))
    return post_data




def sim_trial(input_data,T,prop_opm,opinions_base,allocation,demog_cols,rho,C,O,
              opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
              OPM_to_CM,pt,O2,
              extremists,extreme_index):
    # simulate the run T times to get average behaviour
    data_update = create_data(input_data,T,prop_opm,opinions_base,allocation,demog_cols,
                              extremists,extreme_index)
    #print(data_update.columns)
    for t in range(T):
        #print(t)
        
        data_update = round_update(data_update,t+1,demog_cols,rho,C,O,
                                   opinions_update,exp_included,exp_opinion_array[t],exp_weight,movement_speed,mode,
                                   OPM_to_CM,pt,O2)
        if(sum(data_update.OPM_to_CM_trigger)==n_i):
            #print(sum(data_update.OPM_to_CM_trigger))
            break
    return(data_update)
    
def avg_over_trials(input_data,T,prop_opm,opinions_base,allocation,demog_cols,rho,C,O,n_iterations,
                    opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
                    OPM_to_CM,pt,O2,
                    extremists,extreme_index):
    for i in range(n_iterations):
        #print("Iteration: "+str(i))
        temp_data = sim_trial(input_data,T,prop_opm,opinions_base,allocation,demog_cols,rho,C,O,
                              opinions_update,exp_included,exp_opinion_array[i],exp_weight,movement_speed,mode,
                              OPM_to_CM,pt,O2,
                              extremists,extreme_index)
        temp_data['iteration']=i
        if(i==0):
            output = temp_data
        else:
            output = pd.concat([output,temp_data])
    return(output)
    
    
# openness_simulations.py line 104: input_frame = latex_plot_good, T=200, allocation = 'opt', scale_to_complete=True, n_iterations=5, importance = True
# exp_asmyptote = experts_extreme_update[0]

def double_plot(input_frame,T,allocation,n_iterations,scale_to_complete,importance,exp_asymptote):
    # add in: if importance, need to only look at when the rules triggered a change
    # 4a: each round, how many opm?
    if n_iterations==1:
        input_frame['iteration']=1
    opm_overview = pd.melt(input_frame[[col for col in input_frame if col.startswith(('opm_status_','iteration'))]],id_vars=['iteration'])
    # need to replace NaNs in value
    opm_overview = opm_overview.replace(np.nan,0)
    opm_overview['variable'] = opm_overview.variable.str.replace('opm_status_?' , '')
    # want to remove outliers, e.g. if only one or two are left to become OPM - after last iteration has converged
    opm_min = opm_overview.groupby(['variable','iteration']).sum().reset_index()
    opm_min.variable = opm_min.variable.astype(float)
    if len(opm_min.loc[opm_min.value==0,].groupby('iteration')['variable']) == 0:
        first_0 = T
    else:
        first_0 = max(opm_min.loc[opm_min.value==0,].groupby('iteration')['variable'].min()) # this will never be 0, as always some start with OPM=1
    opm_grouped = opm_overview.groupby('variable').sum().reset_index()
    opm_grouped.variable = opm_grouped.variable.astype(float)
    late_entrants = sum(opm_grouped.loc[opm_grouped.variable>=first_0,"value"])
    time_to_consensus = max(opm_grouped.loc[opm_grouped.value!=0,"variable"])+1
    if time_to_consensus!=(first_0):
        print("Note: not all individuals were OPM by the time we reached 0 aggregate (" + str(late_entrants) + " individuals remaining CM)")
        time_to_consensus=first_0
    #print("NOTE: CONSENSUS NOT REACHED IN " + str(T) + " ROUNDS")
    if scale_to_complete:
        opm_grouped = opm_grouped.loc[opm_grouped['variable']<=time_to_consensus,]
        opm_grouped.variable = 100*opm_grouped['variable']/time_to_consensus
    opm_grouped = opm_grouped.sort_values('variable').reset_index()[['variable','value']]
    # standardise based on number of iterations we're averaging over
    opm_grouped.value = opm_grouped['value']/n_iterations
    # NEED TO ADD ERROR BARS - one standard error away
    if n_iterations>1:
        opm_round_values = opm_overview.groupby(['iteration','variable']).sum().reset_index()
        opm_round_values['variable'] = opm_round_values.variable.str.replace('opm_status_?' , '')
        opm_round_values.variable = opm_round_values.variable.astype(float)
        if scale_to_complete:
            #opm_round_values = opm_round_values.loc[opm_round_values['variable']<=time_to_consensus,]
            opm_round_values.variable = 100*opm_round_values['variable']/time_to_consensus
        #https://matplotlib.org/stable/gallery/lines_bars_and_markers/errorbar_limits_simple.html#sphx-glr-gallery-lines-bars-and-markers-errorbar-limits-simple-py
        opm_errors = pd.merge(opm_round_values,opm_grouped,on='variable')
        opm_errors['num']=(opm_errors['value_x']-opm_errors['value_y'])**2
        # want a standard error for every 'variable' (i.e. round number)
        opm_error_values = opm_errors.groupby('variable').sum().reset_index()[['variable','num']]
        opm_error_values['num'] = np.sqrt(opm_error_values['num']/n_iterations)
    opm_grouped.columns=['Round','Cumulative OPM']
    plt.plot(opm_grouped['Round'],opm_grouped['Cumulative OPM'])
    if n_iterations>1:
        plt.errorbar(opm_grouped['Round'],opm_grouped['Cumulative OPM'],yerr=opm_error_values['num'])
    #plt.title('Cumulative OPM by round for table allocation setting: '+str(allocation))
    plt.ylim(0,n_i)
    plt.ylabel('cumulative # agents with OPM status 1')
    if scale_to_complete:
        x_lim = 100
    else:
        x_lim = T+1
    plt.xlim(0,x_lim)
    if(scale_to_complete):
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.show()
    
    # 4b: how many because of prob
    if importance:
        # for each individual, only need to keep their lowest first
        prob_inspect = input_frame[['first_O','first_rho']]
        # 0 out any entries that are not the lowest - NEED TO INCLUDE pi_0
        prob_inspect['lowest']=np.where(prob_inspect>0,prob_inspect,np.inf).min(axis=1)
        prob_inspect.loc[prob_inspect.first_O!=prob_inspect.lowest,'first_O'] = 0
        prob_inspect.loc[prob_inspect.first_rho!=prob_inspect.lowest,'first_rho'] = 0
        prob_bind=pd.concat([input_frame['opm_0'],prob_inspect],axis=1)
        # 0 out those who were initialised OPM
        prob_bind.loc[prob_bind['opm_0']==1,'first_O']=0
        prob_bind.loc[prob_bind['opm_0']==1,'first_rho']=0
        prob_melt = pd.melt(prob_bind[['first_rho','first_O']])
        # OPM to CM
        prob_inspect_2 = input_frame[['first_pt','first_O2']]
        # 0 out any entries that are not the lowest - NEED TO INCLUDE pi_0
        prob_inspect_2['lowest']=np.where(prob_inspect_2>0,prob_inspect_2,np.inf).min(axis=1)
        prob_inspect_2.loc[prob_inspect_2.first_pt!=prob_inspect_2.lowest,'first_pt'] = 0
        prob_inspect_2.loc[prob_inspect_2.first_O2!=prob_inspect_2.lowest,'first_O2'] = 0
        prob_melt_2 = pd.melt(prob_inspect_2[['first_pt','first_O2']])
        prob_melt_all = pd.concat([prob_melt,prob_melt_2])
    else:
        prob_melt = pd.melt(input_frame[['first_rho','first_O']])
        prob_melt_2 = pd.melt(input_frame[['first_pt','first_O2']])
        prob_melt_all = pd.concat([prob_melt,prob_melt_2])
    prob_grouped = prob_melt_all.groupby(['variable','value']).size().reset_index()
    prob_grouped = prob_grouped[prob_grouped.value!=0]
    prob_grouped = prob_grouped.pivot(index='value',columns='variable',values=0).reset_index()
    # replace na values
    prob_grouped = prob_grouped.fillna(0)
    # need to factor in for when a variable has no influence
    if 'first_rho' not in prob_grouped.columns:
        prob_grouped['first_rho']=0
    if 'first_O' not in prob_grouped.columns:
        prob_grouped['first_O']=0
    if 'first_pt' not in prob_grouped.columns:
        prob_grouped['first_pt']=0
    if 'first_O2' not in prob_grouped.columns:
        prob_grouped['first_O2']=0
    prob_grouped['rho_csum'] = prob_grouped.first_rho.cumsum()
    prob_grouped['O_csum'] = prob_grouped.first_O.cumsum()
    prob_grouped['pt_csum'] = prob_grouped.first_pt.cumsum()
    prob_grouped['O2_csum'] = prob_grouped.first_O2.cumsum()
    prob_grouped = prob_grouped.rename(columns = {'value':'Round'})
    if scale_to_complete:
        prob_grouped.Round = prob_grouped.Round.astype(float)
        #prob_grouped = prob_grouped.loc[prob_grouped['Round']<=time_to_consensus,]
        prob_grouped.Round = 100*prob_grouped['Round']/time_to_consensus
    
    plt.plot(prob_grouped['Round'], prob_grouped['rho_csum']/n_iterations, label = "rho", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O_csum']/n_iterations, label = "O",color="orange")
    plt.legend()
    #plt.title('Influence of different OPM rules for table allocation setting: '+str(allocation))
    plt.ylim(0, n_i)
    if(importance):
        plt.ylabel('cumulative # agents becoming OPM due to each rule')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if(scale_to_complete):
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0,x_lim)
    plt.show()
    plt.plot(prob_grouped['Round'], prob_grouped['pt_csum']/n_iterations, '--', label = "R", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O2_csum']/n_iterations, '--', label = "O'",color="orange")
    #plt.plot(prob_grouped['Round'], prob_grouped['B_csum']/n_iterations, '--', label = "B",color="blue")
    plt.legend()
    #plt.title('Influence of different CM rules for table allocation setting: '+str(allocation))
    plt.ylim(0, n_i)
    if(importance):
        plt.ylabel('cumulative # agents becoming CPM due to each rule')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if(scale_to_complete):
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0,x_lim)
    plt.show()
    
    # 4c: add in opinion shift - just average opinions? Don't want to truncate by time step
    opinion_overview = pd.melt(input_frame[[col for col in input_frame if col.startswith(('opinions_','iteration','id'))]],id_vars=['iteration','id'])
    # need to replace final NaNs with final opinions for each individual
    final_opinions = opinion_overview.groupby(['iteration','id']).last().reset_index()[['iteration','id','value']]
    final_opinions.columns=['iteration','id','value_fill']
    opinion_overview = pd.merge(opinion_overview,final_opinions,on=['iteration','id'],how='left')
    opinion_overview.value.fillna(opinion_overview.value_fill,inplace=True)
    opinion_overview = opinion_overview.drop('value_fill',axis=1)
    opinion_overview['variable'] = opinion_overview.variable.str.replace('opinions_?' , '')
    opinion_grouped = opinion_overview.groupby('variable').mean().reset_index()
    opinion_grouped.variable = opinion_grouped.variable.astype(float)
    if n_iterations>1:
        opinion_round_values = opinion_overview.groupby(['iteration','variable']).mean().reset_index()
        opinion_round_values.variable = opinion_round_values.variable.astype(float)
        opinion_errors = pd.merge(opinion_round_values,opinion_grouped[['variable','value']],on='variable')
        opinion_errors['num']=(opinion_errors['value_x']-opinion_errors['value_y'])**2
        # want a standard error for every 'variable' (i.e. round number)
        opinion_error_values = opinion_errors.groupby('variable').sum().reset_index()[['variable','num']]
        opinion_error_values['num'] = np.sqrt(opinion_error_values['num']/n_iterations)
    opinion_grouped = opinion_grouped.sort_values('variable').reset_index()[['variable','value']]
    opinion_grouped.columns=['Round','Average opinion']
    
    # add 75% bars
    opinion_75 = pd.melt(input_frame[[col for col in input_frame if col.startswith(('opinions_','iteration','id'))]],id_vars=['iteration','id'])
    opinion_75 = pd.merge(opinion_75 ,final_opinions,on=['iteration','id'],how='left')
    opinion_75.value.fillna(opinion_75.value_fill,inplace=True)
    opinion_75 = opinion_75.drop('value_fill',axis=1)
    opinion_75['variable'] = opinion_75.variable.str.replace('opinions_?' , '')
    opinion_75.variable = opinion_75.variable.astype(float)
    # want to group by variable, order by value, and find 25 and 75 values
    #lq=opinion_75.groupby('variable').quantile(0.25).reset_index().sort_values('variable')
    #uq=opinion_75.groupby('variable').quantile(0.75).reset_index().sort_values('variable')
    # moving to a single standard deviation away
    sd = opinion_75.groupby(['variable','iteration']).agg("var").reset_index()
    sd['value'] = np.sqrt(sd['value'])
    sd = sd.groupby('variable')['value'].mean()
    opinion_grouped['sd'] = sd
    opinion_grouped['lq'] = opinion_grouped['Average opinion']-opinion_grouped['sd']
    opinion_grouped['uq'] = opinion_grouped['Average opinion']+opinion_grouped['sd']
    if len(exp_asymptote)!=opinion_grouped.shape[0]:
        exp_asymptote.append(exp_asymptote[len(exp_asymptote)-1])
    opinion_grouped['expert']=exp_asymptote[0:opinion_grouped.shape[0]]
    
    if scale_to_complete:
        opinion_grouped.Round = 100*opinion_grouped['Round']/time_to_consensus
        
    plt.plot(opinion_grouped['Round'],opinion_grouped['Average opinion'],color='red')
    # need to add 75% error bars
    plt.plot(opinion_grouped['Round'],opinion_grouped['lq'],'--',color='blue')
    plt.plot(opinion_grouped['Round'],opinion_grouped['uq'],'--',color='blue')
    if exp_asymptote != "-":
        plt.plot(opinion_grouped['Round'],opinion_grouped['expert'],linestyle="",marker = "o")
    if n_iterations>1:
        plt.errorbar(opinion_grouped['Round'],opinion_grouped['Average opinion'],yerr=opinion_error_values['num'],color="red")
    #plt.title('Average opinion by round for table allocation setting: '+str(allocation))
    plt.ylim(0,10)
    # add expert asymptotes
    #if exp_asymptote != "-":
        # want to add a line for expert presentations
        
        #if type(exp_asymptote)==int:
        #    plt.axhline(y = exp_asymptote, color = 'r', linestyle = '--')
        #else:
        #    for hline in exp_asymptote:
        #        plt.axhline(y = hline, color = 'r', linestyle = '--')
    plt.ylabel('average opinion (with LQ+UQ)')
    if(scale_to_complete):
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('t')
    plt.xlim(0,x_lim)
    plt.show()
    
    # final plot - all OPM on a single view
    # a
    plt.plot(opm_grouped['Round'],opm_grouped['Cumulative OPM'])
    if n_iterations>1:
        plt.errorbar(opm_grouped['Round'],opm_grouped['Cumulative OPM'],yerr=opm_error_values['num'],color='orange',label='# agents OPM')
    #plt.title('Cumulative OPM by round for table allocation setting: '+str(allocation))
    plt.legend()
    plt.ylim(0,n_i)
    if scale_to_complete:
        x_lim = 100
    else:
        x_lim = T+1
    plt.xlim(0,x_lim)
    if(scale_to_complete):
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    
    
    plt.plot(prob_grouped['Round'], prob_grouped['rho_csum']/n_iterations, label = "rho", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O_csum']/n_iterations, label = "O",color="orange")
    plt.legend()
    #plt.title('Influence of different OPM rules for table allocation setting: '+str(allocation))
    plt.ylim(0, n_i)
    if(importance):
        plt.ylabel('cumulative # agents becoming OPM due to each rule')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if(scale_to_complete):
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0,x_lim)
    plt.plot(prob_grouped['Round'], prob_grouped['pt_csum']/n_iterations, '--', label = "p", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O2_csum']/n_iterations, '--', label = "O'",color="orange")
    #plt.plot(prob_grouped['Round'], prob_grouped['B_csum']/n_iterations, '--', label = "B",color="blue")
    plt.legend()
    #plt.title('Influence of different CM rules for table allocation setting: '+str(allocation))
    plt.ylim(0, n_i)
    if(importance):
        plt.ylabel('cumulative # agents')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if(scale_to_complete):
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0,x_lim)
    plt.show()
    
    
    
    
    
    
# data = isolate_fit_opt
    
from scipy.optimize import curve_fit
    
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

# data = curve_fit_data_2
pad_0=False
scale_to_100 =False
method = 'opt'
importance=True
def fit_exp_to_O(data,pad_0,scale_to_100,method,importance):
    #https://rowannicholls.github.io/python/curve_fitting/exponential.html
    if importance:
        # for each individual, only count O if it is their first trigger
        prob_inspect = data[['first_pt','first_O']]
        # 0 out any entries that are not the lowest - NEED TO INCLUDE pi_0
        prob_inspect['lowest']=np.where(prob_inspect>0,prob_inspect,np.inf).min(axis=1)
        prob_inspect.loc[prob_inspect.first_O!=prob_inspect.lowest,'first_O'] = 0
        prob_bind=pd.concat([data[['iteration','trial','opm_0']],prob_inspect['first_O']],axis=1)
        # 0 out those who were initialised OPM
        prob_bind.loc[prob_bind['opm_0']==1,'first_O']=0
        prob_grouped = prob_bind[prob_bind.first_O!=0].groupby(['iteration','trial','first_O']).size().reset_index()
    else:
        prob_grouped = data[data.first_O!=0].groupby(['iteration','trial','first_O']).size().reset_index()
    prob_grouped.columns = ['iteration','trial','Round','ct']
    # add in max for each iteration
    prob_grouped_max_round = prob_grouped.groupby(['iteration','trial']).agg(max).reset_index()[['iteration','trial','Round']]
    prob_grouped_max_round.columns = ['iteration','trial','max_round']
    prob_grouped_min_round = prob_grouped.groupby(['iteration','trial']).agg(min).reset_index()[['iteration','trial','Round']]
    prob_grouped_min_round.columns = ['iteration','trial','min_round']
    prob_grouped = prob_grouped.merge(prob_grouped_max_round,on=['iteration','trial'],how='left')
    prob_grouped = prob_grouped.merge(prob_grouped_min_round,on=['iteration','trial'],how='left')
    prob_grouped = prob_grouped[['iteration','trial','Round','ct','max_round']]
    # pad 0s for each iteration
    if(pad_0):
        # add a row for each value under min_round
        # create dictionary of unique iteration/trial/min_rounds
        for i in range(prob_grouped_min_round.shape[0]):
            # add min_round rows, if min round is not 0
            n_reps = prob_grouped_min_round.min_round[i]
            if n_reps>=1:
                new_frame = pd.DataFrame(index=range(n_reps),columns=['iteration','trial','Round','ct','max_round'])
                new_frame.iteration = prob_grouped_min_round.iteration[i]
                new_frame.trial = prob_grouped_min_round.trial[i]
                new_frame.Round = range(n_reps)
                new_frame.ct = 0
                new_frame.max_round = prob_grouped_max_round.max_round[i]
                if i==0:
                    pad_frame = new_frame
                else:
                    pad_frame = pd.concat([pad_frame,new_frame])
        prob_grouped = pd.concat([prob_grouped,pad_frame]).sort_values(["iteration","trial","Round"])
    prob_grouped['O_csum'] = prob_grouped.groupby(['iteration','trial']).ct.cumsum()
    if scale_to_100==True:
        prob_scale = prob_grouped.groupby(['iteration','trial']).agg(max).reset_index()[['iteration','trial','O_csum']]
        prob_scale.columns=['iteration','trial','agent_scale']
        prob_grouped = prob_grouped.merge(prob_scale,on=['iteration','trial'],how="left")
    else:
        prob_grouped['agent_scale']=100
    prob_grouped['O_csum']=100*prob_grouped.O_csum/prob_grouped.agent_scale
    # need to normalise over 0-1 scale
    prob_grouped.Round = 100*prob_grouped.Round/prob_grouped.max_round
    prob_grouped = prob_grouped[['Round','ct','O_csum']].sort_values(['Round'])
    # fit polynomial curve to data: 
    x = prob_grouped.Round
    y = prob_grouped.O_csum
    p_exp = np.polyfit(x,np.log(y),1)
    a = np.exp(p_exp[1])
    b = p_exp[0]
    x_fitted = x.sort_values()
    y_fitted_exp = a*np.exp(b*x_fitted)
    # fit sigmoid
    p0 = [max(y), np.median(x),1,min(y)] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x, y,p0, method='dogbox',maxfev=5000)
    y_fitted_sig = sigmoid(x_fitted, *popt)
    ax = plt.axes()
    ax.scatter(x, y, label='Raw data')
    ax.plot(x_fitted, y_fitted_exp, 'k', label='Fitted exponential curve')
    ax.plot(x_fitted, y_fitted_sig, 'k', label='Fitted sigmoid curve',color='red')
    if(importance):
        ylab="% agents becoming OPM by triggering O rule"
    else:
        ylab="% of agents triggering O rule"
    ax.set_title('Convergence to OPM for various parameter settings: '+str(method))
    ax.set_ylabel(str(ylab))
    ax.set_ylim(0, 100)
    ax.set_xlabel('% of time to convergence')
    ax.legend()








def opinion_analysis(data,experts,expert_input,n_iterations,extreme_index):
    opinion_overview = pd.melt(data[[col for col in data if col.startswith(('opinions_','iteration','id'))]],id_vars=['iteration','id'])
    opinion_overview['variable'] = opinion_overview.variable.str.replace('opinions_?' , '')
    opinion_overview.variable = opinion_overview.variable.astype(float)
    # what is the maximal round number?
    max_round = max(opinion_overview.variable)
    # drop na rows
    opinion_overview = opinion_overview[opinion_overview['value'].notna()].sort_values(by='variable')
    opinion_grouped = opinion_overview.groupby(['iteration','id'])
    bound_opinions = pd.concat([opinion_grouped.head(1),opinion_grouped.tail(1)]).drop_duplicates().sort_values(['iteration','id']).reset_index(drop=True)
    bound_opinions['type'] = np.where(bound_opinions['variable']==0,"First","Last")
    # also want to add round where they reached final opinion - within 0.01 of last opinion
    last_opinions = bound_opinions.loc[bound_opinions.type=="Last",['iteration','id','value']]
    last_opinions = last_opinions.rename(columns={'value':'last_value'})
    last_opinion_reached = pd.merge(opinion_overview,last_opinions,on=['iteration','id'],how='left')
    last_opinion_reached['within'] = np.where((last_opinion_reached['value']-last_opinion_reached['last_value']<=0.01)&(last_opinion_reached['value']-last_opinion_reached['last_value']>=-0.01),1,0)
    # want last round where within = 0, then take following result
    last_opinions_merge = last_opinion_reached.loc[last_opinion_reached['within']==0].groupby(['iteration','id']).last().reset_index()[['iteration','id','variable']]
    last_opinions_merge.variable = np.where(last_opinions_merge['variable']==max_round,last_opinions_merge['variable'],last_opinions_merge['variable']+1)
    last_opinions_merge['type'] = 'Mid'
    # no rows for those whose opinions never changed - need to treat differently if commited minority
    # merge on actual opinion
    last_opinions_merge = pd.merge(last_opinions_merge,last_opinion_reached[['iteration','id','variable','value']],on=['iteration','id','variable'],how='left')[['iteration','id','variable','value','type']]
    bound_opinions = bound_opinions.append(last_opinions_merge,ignore_index=True)
    # add rows for those whose opinions never changed
    no_change = bound_opinions.groupby(['id','iteration']).count().reset_index()
    no_change_ind = no_change.loc[no_change.type==2][['id','iteration']]
    # check these cases exist
    if no_change_ind.shape[0] > 0:
        no_change_ind = pd.merge(no_change_ind,bound_opinions,on=['id','iteration'],how='left')
        no_change_ind = no_change_ind.loc[no_change_ind['type']=="First"]
        no_change_ind['type']="Mid"
        bound_opinions = bound_opinions.append(no_change_ind,ignore_index=True)
    # also want to add first opinion shift - when did they start moving?
    first_opinions = bound_opinions.loc[bound_opinions.type=="First",['iteration','id','value']]
    first_opinions = first_opinions.rename(columns={'value':'first_value'})
    first_opinion_reached = pd.merge(opinion_overview,first_opinions,on=['iteration','id'],how='left')
    first_opinion_reached['within'] = np.where(first_opinion_reached['value']-first_opinion_reached['first_value']==0,1,0)
    # want last round where within = 1, then take following result
    first_opinions_merge = first_opinion_reached.loc[first_opinion_reached['within']==1].groupby(['iteration','id']).last().reset_index()[['iteration','id','variable']]
    #first_opinions_merge.variable = np.where(first_opinions_merge['variable']==max_round,first_opinions_merge['variable'],first_opinions_merge['variable']+1)
    first_opinions_merge['type'] = 'Start'
    # convert those who never change back to 0
    first_opinions_check = pd.merge(first_opinions_merge,bound_opinions.loc[bound_opinions.type=="Last"][['iteration','id','variable']].rename(columns={"variable":"last_vbl"}),on=['iteration','id'],how='left')
    first_opinions_check.loc[first_opinions_check.variable == first_opinions_check.last_vbl,"variable"] = 0
    # merge on actual opinion
    first_opinions_merge = pd.merge(first_opinions_check[['iteration','id','variable','type']],first_opinion_reached[['iteration','id','variable','value']],on=['iteration','id','variable'],how='left')[['iteration','id','variable','value','type']]
    bound_opinions = bound_opinions.append(first_opinions_merge,ignore_index=True)
    # add rows for those whose opinions never changed
    no_change = bound_opinions.groupby(['id','iteration']).count().reset_index()
    no_change_ind = no_change.loc[no_change.type==3][['id','iteration']]
    # check these cases exist
    if no_change_ind.shape[0] > 0:
        no_change_ind = pd.merge(no_change_ind,bound_opinions,on=['id','iteration'],how='left')
        no_change_ind = no_change_ind.loc[no_change_ind['type']=="First"]
        no_change_ind['type']="Start"
        bound_opinions = bound_opinions.append(no_change_ind,ignore_index=True)    
    # now we have data, want to extract meaning:
    # a) mean final opinion
    mean_final = bound_opinions.loc[bound_opinions['type']=="Last"]['value'].mean()
    # b) mean initial opinion
    mean_initial = bound_opinions.loc[bound_opinions['type']=="First"]['value'].mean()
    # c) mean expert opinion
    expert_redux = expert_input[0:int(max_round)]
    mean_expert = expert_redux.mean()
    # R1 and R2
    R1 = mean_final-mean_initial
    #R2a = abs(mean_initial-mean_expert)
    R2a = np.mean(abs(bound_opinions.loc[bound_opinions['type']=="First"]['value']-mean_expert))
    R2b = np.mean(abs(bound_opinions.loc[bound_opinions['type']=="Last"]['value']-mean_expert))
    R2 = R2a-R2b
    # d) N8,0-N8,T
    N8_0 = bound_opinions.loc[(bound_opinions.type=='First') & (bound_opinions.value>=8)].shape[0]
    N8_T = bound_opinions.loc[(bound_opinions.type=='Last') & (bound_opinions.value>=8)].shape[0]
    # e) N9,0-N9,T
    N9_0 = bound_opinions.loc[(bound_opinions.type=='First') & (bound_opinions.value>=8.5)].shape[0]
    N9_T = bound_opinions.loc[(bound_opinions.type=='Last') & (bound_opinions.value>=8.5)].shape[0]
    # f) N0,0-N0,T
    N1_0 = bound_opinions.loc[(bound_opinions.type=='First') & (bound_opinions.value<=1)].shape[0]
    N1_T = bound_opinions.loc[(bound_opinions.type=='Last') & (bound_opinions.value<=1)].shape[0]
    # g) N1,0-N1,T
    N0_0 = bound_opinions.loc[(bound_opinions.type=='First') & (bound_opinions.value<=0.5)].shape[0]
    N0_T = bound_opinions.loc[(bound_opinions.type=='Last') & (bound_opinions.value<=0.5)].shape[0]
    # R3
    R3a = (N8_0-N8_T)/n_iterations
    R3b = (N9_0-N9_T)/n_iterations
    R3c = (N1_0-N1_T)/n_iterations
    R3d = (N0_0-N0_T)/n_iterations
    # h) MT
    # want mode for each iteration - number of peaks in KDE
    # if extremists, need to exclude from analysis
    if extreme_index == 0:
        modal_data = bound_opinions.loc[bound_opinions.type=='Last']
    else:
        modal_data = bound_opinions.loc[(bound_opinions.type=='Last') & (~bound_opinions.id.isin(extreme_index))]
    # plot modal data as sense check if needed
    #sns.distplot(modal_data['value'], hist=False, kde=True,bins=int(1800/5), color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
    data = np.msort(modal_data['value'])
    intervals = UniDip(data).run()
    # the bounds of each peak
    R4 = len(intervals)
    # is this always the mode?
    modes = []
    for iteration in range(n_iterations):
        iter_modal_data = np.msort(modal_data.loc[modal_data.iteration==iteration]['value'])
        intervals = UniDip(iter_modal_data).run()
        modes.append(len(intervals))
    individual_mode_count = len([x for x in modes if x==R4]) 
    # i) Var(X)T
    R5 = np.mean((bound_opinions.loc[bound_opinions['type']=="Last"].groupby('iteration').std()['value'])**2)
    # j) Average runtime
    avg_T = bound_opinions.loc[bound_opinions['type']=="Last"][['variable','iteration']].groupby('iteration').max().mean()[0]
    # k) Average flux start time
    avg_t_opm = bound_opinions.loc[bound_opinions['type']=="Start"]['variable'].mean()
    # l) Average flux end time
    avg_t_cm = bound_opinions.loc[bound_opinions['type']=="Mid"]['variable'].mean()
    # m) Average flux time
    avg_flux = avg_t_cm-avg_t_opm
    
    return {"R1":R1,
            "R2a":R2a,
            "R2b":R2b,
            "R2":R2,
            "R3a (8)":R3a,
            "R3b (9)":R3b,
            "R3c (1)":R3c,
            "R3d (0)":R3d,
            "R4":R4,
            "R4 uniqueness":individual_mode_count/n_iterations,
            "R5":R5,
            "Average runtime":avg_T,
            "Average t OPM":avg_t_opm,
            "Average t CM":avg_t_cm,
            "Average time in flux":avg_flux}




def kde_plot(opinions,title,legend_status):
    long_data = pd.wide_to_long(opinions,stubnames='opinions',i=['iteration','id'],j='round',sep="_").reset_index()[['iteration','id','round','opinions']]
    # only want first and last opinions per id and iterations
    long_data = long_data.dropna()
    long_data_grouped = long_data.groupby(['iteration','id'])
    filter_data = pd.concat([long_data_grouped.head(1),long_data_grouped.tail(1)])
    filter_data['round'].loc[filter_data['round']==0]="First"
    filter_data['round'].loc[filter_data['round']!="First"]="Last"
    for value in ['First','Last']:
        subset = filter_data[filter_data['round']==value]
        sns.distplot(subset['opinions'],hist=False,kde=True,kde_kws={'linewidth':3,'clip': (0.0, 9.0)},label=value)
    if legend_status == True:
        plt.legend(prop={'size': 12}, title = 'Round')
    #plt.title(title)
    plt.xlabel('Opinion')
    plt.ylabel('Density')
    plt.ylim(0,0.5)
    # fix ylim at 0, 0.5
    



