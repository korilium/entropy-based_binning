import math 
import pandas as pd 
import numpy as np 


def entropy2(train_data, classifier , base = 2):

  count = [len(train_data[train_data[classifier]==1]), len(train_data[train_data[classifier]==0]), len(train_data)]
  prob_1 =count[2] and count[0]/count[2] or 0
  prob_2 = count[2] and count[1]/count[2] or 0
  entropy = (prob_1 and  - prob_1 * math.log(prob_1,2) or 0 )- ( prob_2 and   prob_2* math.log(prob_2, base) or 0 ) 



  return entropy


def entropy_split(train_data, label, classifier,  bins=100, base = 2):
    total_entropy = entropy2(train_data, classifier,base)
    bins = list(np.linspace(
        math.floor(train_data[label].min()),
        math.ceil(train_data[label].max()),
        bins))
    x= []
    result=[]
    for bin in bins : 
        upper, lower = train_data[train_data[label]> bin], train_data[train_data[label]< bin]
        count= [len(upper[upper[classifier] == 1]), len(upper[upper[classifier] == 0]), len(upper),len(lower[lower[classifier] == 1]),len(lower[lower[classifier] == 0]),len(lower), len(train_data)]
        prob_u1 = count[2] and count[0]/count[2] or 0 # if divide by zero set probability to zero 
        prob_u2 = count[2] and count[1]/count[2] or 0
        prob_l1 = count[5] and count[3]/count[5] or 0
        prob_l2 = count[5] and count[4]/count[5] or 0
        
        entropy_u =  ( prob_u1 and - prob_u1*math.log(prob_u1, base) or  0 )+ ( prob_u2 and - prob_u2*math.log(prob_u2, base) or  0 ) # probability zero set to zero 

        entropy_l = ( prob_l1 and - prob_l1*math.log(prob_l1, base) or  0 )+ ( prob_l2 and - prob_l2*math.log(prob_l2, base) or  0 )
        entropy_gain = total_entropy  - (entropy_u*count[2]/count[6]+ entropy_l*count[5]/count[6])
        entropy = [bin, entropy_gain, entropy_u, entropy_l,count[2], count[5]]
        x.append(entropy)
        results  = pd.DataFrame(x, columns= ['bin_point','entropy_gain', 'entropy_u', 'entropy_l', 'count_u', 'count_l'])
        result = results[results['entropy_gain'] == results.entropy_gain.max()][:1]
        # y.append(entropy_l)
        # z.append(entropy_u)
        # w.append(count)
        # q.append(entropy)
   # fig =  pd_result.plot.line(x='bin_point', y='entropy_gain')
    return result#, y, z, w, q


from functools import total_ordering
from itertools import count


def entropy_binning(train_data, label, classifier, split_bins=100,  minimum_gain=0.0001):
    total_entropy = entropy2(train_data, classifier=classifier)
    results= entropy_split(train_data, label=label, classifier = classifier, bins=split_bins)

    n = 0
    reg_entropy = pd.DataFrame({ 'splits':[0], 'split_side' : [None], 'bin_point':[None], 'next_optimal_bin': None, 'bin_entropy_gain':0,
                                'used': [0], 'entropy': total_entropy,'entropy_gain':1 })
    data= {n : {'data_u': train_data,
                'data_l': train_data,
                'entropy_u' :results,
                'entropy_l' : results,
                'used_split': 0 }}


    entropy = [[results['count_u'].values[0],results['entropy_u'].values[0],n, 'upper'], [results['count_l'].values[0], results['entropy_l'].values[0], n, 'lower']]

    entropy = pd.DataFrame(entropy, columns=['count','entropy', 'split', 'side'])

    prev_entropy  = total_entropy
    total_count= len(train_data)
    entropy['prec'] = entropy['count']/total_count
    now_entropy = np.sum(entropy['prec']*entropy['entropy'])
    entropy_gain = prev_entropy - now_entropy
    while reg_entropy['entropy_gain'].min() > minimum_gain : 
        if len(reg_entropy) < 2 : 
            #chosen parameters 
            chosen_bin = results['bin_point'].values[0]
            reg_entropy['next_optimal_bin'] = chosen_bin
            n = n+1
            data_upper = train_data.loc[train_data[label]> chosen_bin]
            data_lower = train_data.loc[train_data[label]< chosen_bin]
            entropy_u = entropy_split(data_upper, label=label, classifier=classifier )
            entropy_l = entropy_split(data_lower, label=label, classifier=classifier )
            data[n] =  {'data_u'  : data_upper,
                            'data_l' : data_lower,
                            'entropy_u': entropy_u, 
                            'entropy_l': entropy_l,  
                            'used_split': n}
            
        else: 
            #next step 
            n = n+1
            #find the optimal bin which has not yet been used based on the entropy gain in its region 
            bin_option  = reg_entropy[ (reg_entropy.used==0) ]
            optimal_bin =  bin_option[bin_option.bin_entropy_gain ==  bin_option.bin_entropy_gain.max()]
            #chosen parameters 
            chosen_bin = optimal_bin['next_optimal_bin'].values[0]
            chosen_split = optimal_bin['splits'].values[0]
            chosen_split_side = optimal_bin['split_side'].values[0]
            #set the bin to used 

            #remove the branch that has been split in two in the entropy calculation
            entropy = entropy.drop(entropy[(entropy['split'] == chosen_split-1) & (entropy["side"] == chosen_split_side)].index)

            # select the used branch to retrieve the data and calculate the entropy gain from the split within the branch
            if chosen_split_side == 'upper': 
                entropy_addu  = pd.Series({ 'count': data[chosen_split]['entropy_u']['count_u'].values[0] ,
                                            'entropy' : data[chosen_split]['entropy_u']['entropy_u'].values[0], 
                                            'split' : n-1, 
                                            'side': 'upper'})
                entropy_addl  = pd.Series({'count':  data[chosen_split]['entropy_u']['count_l'].values[0],
                                            'entropy' : data[chosen_split]['entropy_u']['entropy_l'].values[0],
                                            'split' : n-1, 
                                            'side': 'lower'})
                entropy = pd.concat([entropy, entropy_addu.to_frame().T, entropy_addl.to_frame().T], ignore_index=True)
                data_upper = data[chosen_split]['data_u'][       data[chosen_split]['data_u'][label]> chosen_bin            ]
                data_lower = data[chosen_split]['data_u'][       data[chosen_split]['data_u'][label]< chosen_bin            ]
                entropy_u = entropy_split(data_upper, label=label, classifier=classifier)
                entropy_l = entropy_split(data_lower, label=label, classifier=classifier)
                data[n] = {'data_u'  : data_upper,
                            'data_l' : data_lower,
                            'entropy_u' :entropy_u, 
                            'entropy_l' : entropy_l, 
                            'used_split': [chosen_split]}
                        

            else: 
                entropy_addu  = pd.Series({ 'count' : data[chosen_split]['entropy_l']['count_u'].values[0],
                                            'entropy' : data[chosen_split]['entropy_l']['entropy_u'].values[0], 
                                            'split' : n-1, 
                                            'side': 'upper'})
                entropy_addl  = pd.Series({'count':  data[chosen_split]['entropy_l']['count_l'].values[0],
                                            'entropy' : data[chosen_split]['entropy_l']['entropy_l'].values[0],
                                            'split' : n-1, 
                                            'side': 'lower'})
                entropy = pd.concat([entropy, entropy_addu.to_frame().T, entropy_addl.to_frame().T], ignore_index=True)
                data_upper = data[chosen_split]['data_l'][       data[chosen_split]['data_l'][label]> chosen_bin          ]
                data_lower = data[chosen_split]['data_l'][       data[chosen_split]['data_l'][label]< chosen_bin          ]

                entropy_u = entropy_split(data_upper, label=label, classifier=classifier)
                entropy_l = entropy_split(data_lower, label=label, classifier=classifier)
                data[n] = {'data_u'  : data_upper,
                            'data_l' : data_lower,
                            'entropy_u' :entropy_u, 
                            'entropy_l' : entropy_l, 
                            'used_split': [chosen_split]}

        entropy['prec'] = entropy['count']/total_count
        now_entropy = np.sum(entropy['prec']*entropy['entropy'])
        entropy_gain = prev_entropy - now_entropy


        #add all the results together 
        pd_entropy1 =  [ chosen_bin,n,entropy_u['bin_point'].values[0], 'upper', 0, now_entropy, entropy_gain, entropy_u['entropy_gain'].values[0]]
        pd_entropy2 = [chosen_bin,n,entropy_l['bin_point'].values[0], 'lower', 0, now_entropy, entropy_gain, entropy_l['entropy_gain'].values[0]]
        add_entropy = pd.DataFrame([pd_entropy1, pd_entropy2], 
                                    columns= ['bin_point', 'splits', 'next_optimal_bin','split_side', 'used', 'entropy','entropy_gain', 'bin_entropy_gain'])

        reg_entropy = pd.concat([reg_entropy, add_entropy], ignore_index=True)


        used_rule = chosen_bin== reg_entropy.next_optimal_bin
        reg_entropy.loc[used_rule, 'used'] = 1 
        prev_entropy=now_entropy
    return reg_entropy
