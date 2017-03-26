from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from __future__ import division
from sklearn import metrics
import string
import math
import csv
import random
import time
import pandas as pd
from sklearn import tree

# GLOBALS:
BUDGET = float(6250000)


def get_std_slotprice(path,column="slotprice"):
    df = pd.read_csv(path, skipinitialspace=True, usecols=[column])
    return int(df.slotprice.values.std())

def get_LRS_params(path):
    df=pd.read_csv(path)
    avgCTR=(df.click.sum()/df.shape[0])*100
    base_bid=df.payprice.mean()
    return avgCTR,base_bid

def load_data(filepath,training=True):
    data = defaultdict(list)
    labels = defaultdict(list)
    # std of slotprice for nomalization
    STD_SLOTPRICE = get_std_slotprice(filepath,column="slotprice")
    print "std stored"
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # pass header:
        next(reader)
        # Iterate:
        for row in reader:
            instance=process_event(row,STD_SLOTPRICE)
            data[row[24]].append(instance)
            labels[row[24]].append(int(row[0]))
    print "data and labels loaded"
    return data,labels


def trainCTRmodel(training_events, training_labels):
    models = {}
    for key in training_events.keys():
        data=training_events[key]
        labels= training_labels[key]


        label_encoder = LabelEncoder()
        vectorizer = DictVectorizer(sparse=True)

        train_event_x = vectorizer.fit_transform(data)
        train_event_y = label_encoder.fit_transform(labels)

        # Getting the class weight to rebalance data:
        #neg_weight = sum(labels) / len(labels)
        #pos_weight = 1 - neg_weight

        # Create and train the model.
        #p = 0.34
        #model = LogisticRegression(C=p, class_weight={1: pos_weight, 0: neg_weight})
        #model=LogisticRegressionCV(class_weight={1: pos_weight, 0: neg_weight})
        model = SGDClassifier(class_weight="balanced",learning_rate='optimal', \
                           n_iter=20,loss="log",penalty='elasticnet')
        #model=tree.DecisionTreeClassifier(class_weight="balanced",max_features="sqrt")
        #print "labels: "+str(labels)
        model.fit(train_event_x, train_event_y)
        models[key] = (model, label_encoder, vectorizer)
        print('Training model for advertiser %s done')%(key)
    return models


def process_event(row,STD_SLOTPRICE,data_type="Training"):
    # Initilize instance:
    if data_type=="Training":
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17], 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}
        instance = update_usertag_os_browser(instance,row)
        return instance

    elif data_type=="Validation":
        instance = {'weekday': row[1], 'hour': row[2], 'region': row[8], \
                    'city': row[9], 'adexchange': row[10], 'slotwidth': row[15], 'slotheight': row[16], \
                    'slotvisibility': row[17],'payprice':int(row[22]), 'slotformat': row[18], 'slotprice': float(row[19]) / STD_SLOTPRICE, \
                    'advertiser': row[24]}
        instance = update_usertag_os_browser(instance, row)
        return instance
    else:
        instance = {'weekday': row[0], 'hour': row[1], 'region': row[7], \
                    'city': row[8], 'adexchange': row[9], 'slotwidth': row[14], 'slotheight': row[15], \
                    'slotvisibility': row[16],'slotformat': row[17], 'slotprice': float(row[18]) / STD_SLOTPRICE, \
                    'advertiser': row[21]}
        instance = update_usertag_os_browser(instance, row,col_usertag=22,col_os_brow=5)
        return instance


def update_usertag_os_browser(instance,row,col_usertag=25,col_os_brow=6):
    usertags = row[col_usertag].split(',')
    temp_dict = {}
    for tag in usertags:
        temp_dict["tag " + tag] = True
    instance.update(temp_dict)
    # add OS and browser:
    op_sys, browser = row[col_os_brow].split('_')
    instance.update({op_sys: True, browser: True})
    return instance


def predict_event_CTR(instance,advertiser, model): # models:dict
    #print "model adv " +str(model[advertiser])
    lr = model[advertiser][0]
    # Transform event:
    label_encoder = model[advertiser][1]
    vectorizer = model[advertiser][2]
    event = [instance]
    event_x = vectorizer.transform(event)
    #event_y = label_encoder.inverse_transform(lr.predict(event_x))
    event_y = lr.predict_proba(event_x)
    return event_y[0][1]


def testCTRmodel(models_dictio,validation_path):
    performance=defaultdict(lambda : defaultdict(float))
    y_pred=[]
    y_test=[]
    adv_y_pred = defaultdict(list)
    adv_y_true = defaultdict(list)
    impressions = 0
    clicks = 0

    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)

    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)

        for row in reader:
            advertiser=row[24]
            # parsing event:
            instance = process_event(row, STD_SLOTPRICE, data_type="Validation")
            # Predicting CTR, in this case this is our b(theta):
            pCTR = predict_event_CTR(instance,advertiser, models_dictio)
            true_click = int(row[0])
            y_test.append(true_click)
            pred_click= int(pCTR>=0.5)
            y_pred.append(pred_click)

            # adding to the dictio:
            adv_y_pred[advertiser].append(pred_click)
            adv_y_true[advertiser].append(true_click)


            if pCTR > 0.5:
                impressions += 1
                performance[advertiser]['impressions'] += 1
                if true_click == 1:
                    clicks += 1
                    performance[advertiser]['clicks'] += 1

            #performance[advertiser]['impressions']+=1

            if performance[advertiser]['impressions']>0:
                performance[advertiser]['CTR'] = performance[advertiser]['clicks']/performance[advertiser]['impressions']


    print("Impressions:{0}".format(impressions))
    print("Clicks:{0}".format(clicks))
    return performance, y_test, y_pred,adv_y_true,adv_y_pred




def RTB_simulation(model, validation_path, start_budget = BUDGET, lambda_const=5.5e-4, c=60,
                   winning_rate="w1",data_type="Validation"):  # param is the dictionary with the bidprice per advertiser

    performance = defaultdict(lambda: defaultdict(float))
    y_pred = []
    y_test = []
    adv_y_pred = defaultdict(list)
    adv_y_true = defaultdict(list)
    final_results=[]
    impressions = 0
    clicks = 0
    spend = 0
    budget=start_budget
    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)
    results=[]

    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            advertiser=row[24]
            # parsing event:
            instance = process_event(row, STD_SLOTPRICE, data_type = data_type)

            # Predicting CTR, in this case this is our b(theta):
            pCTR = predict_event_CTR(instance,advertiser, model)
            true_click = int(row[0])
            y_test.append(true_click)
            pred_click= int(pCTR>=0.5)
            y_pred.append(pred_click)
            # adding to the dictio:
            adv_y_pred[advertiser].append(pred_click)
            adv_y_true[advertiser].append(true_click)

            # Calculate the bid based on ORTB:
            if winning_rate=="w1":
                current_bid = np.sqrt((c*pCTR)/lambda_const+np.power(c,2))-c
            else:
            # Second way:
                current_bid=np.power((pCTR+np.square((c**2*lambda_const**2+pCTR**2))/(c*lambda_const)),(1/3))- \
                                np.power(((c*lambda_const)/(pCTR+np.square((c**2*lambda_const**2+pCTR**2)))),(1/3))

            # Check if we still have budget:
            if budget > current_bid:

                # Get the market price:

                payprice = float(instance['payprice'])

                # Check if we win the bid:
                if current_bid > payprice:
                    impressions += 1
                    budget -= payprice
                    performance[advertiser]['impressions'] += 1
                    performance[advertiser]['spend'] += payprice
                    spend += payprice
                    # Check if the person clicks:
                    if row[0] == "1":
                        print "current bid : %d , payprice: %d, click? : %s, pCTR: %0.3f" % (int(current_bid), int(payprice), row[0],pCTR)
                        clicks += 1
                        performance[advertiser]['clicks'] += 1

    print("Impressions:{0}".format(impressions))
    print("Clicks:{0}".format(clicks))
    print("Reamining Budget:{0}".format(budget))

    for key in performance.keys():
        if performance[key]['clicks']>0:
            results.append((key,performance[key]['impressions'],performance[key]['spend'],\
                   performance[key]['clicks'],c,lambda_const))
        else:
            results.append((key,performance[key]['impressions'],performance[key]['spend'],\
                   0,c,lambda_const))

    if impressions > 0:
        #print "Best bid CTR: " + str((clicks / impressions) * 100)
        finalCTR=(clicks / impressions) * 100
        final_results.append((finalCTR,clicks,impressions,spend,c,lambda_const))
        return final_results,results,performance
    else:
        return -1,results,performance



def RTB_simulation_test(model, validation_path, start_budget = BUDGET,
                        lambda_const=5.5e-4, c=60, data_type="test"):  # param is the dictionary with the bidprice per advertiser

    # Calculate the standard deviation for slotprice
    STD_SLOTPRICE = get_std_slotprice(validation_path)
    print "got std, start simulation "
    results=[]

    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            advertiser=row[21]
            # parsing event:
            instance = process_event(row, STD_SLOTPRICE, data_type = data_type)

            # Predicting CTR, in this case this is our b(theta):
            pCTR = predict_event_CTR(instance,advertiser, model)

            # Calculate the bid based on ORTB:
            current_bid = np.sqrt((c*pCTR)/lambda_const+np.power(c,2))-c

            # Second way:
            #current_bid=np.power((pCTR+np.square((c**2*lambda_const**2+pCTR**2))/(c*lambda_const)),(1/3))- \
            #                    np.power(((c*lambda_const)/(pCTR+np.square((c**2*lambda_const**2+pCTR**2)))),(1/3))
            results.append((row[2],current_bid))

        return results


def find_optimal_c(model, validation_path, training_path):  # param is the dictionary with the bidprice per advertiser:
    c_range=np.arange(20,200,15)
    val_c_CTR=[]
    for c in c_range:
        predCTR=RTB_simulation(model, validation_path, training_path,c=c)
        val_c_CTR.append((c,predCTR))
        print "c: %d, CTR: %0.5f"%(c,predCTR)
    return val_c_CTR


def find_optimal_lambda(model, validation_path, training_path):  # param is the dictionary with the bidprice per advertiser:
    lambda_range=np.logspace(-7,1,10)
    val_c_CTR=[]
    for lamb in lambda_range:
        predCTR=RTB_simulation(model, validation_path, training_path, lambda_const=lamb, c=140)
        val_c_CTR.append((lamb,predCTR))
        print "lambda: %d, CTR: %0.5f"%(lamb,predCTR)
    return val_c_CTR


def find_optimal_params(model, validation_path, training_path):  # param is the dictionary with the bidprice per advertiser:
    c_range = np.arange(20, 200, 15)
    lambda_range=np.logspace(-7,1,10)
    params_CTR=[]
    for lamb in lambda_range:
        for c in c_range:
            predCTR=RTB_simulation(model, validation_path, training_path, lambda_const=lamb, c=c)
            params_CTR.append((c,lamb,predCTR))
            print "lambda: %0.2f, c: %d, CTR: %0.5f"%(lamb,c,predCTR)
    return params_CTR


def results_to_csv(results):
    results=np.array(results)
    results=pd.DataFrame(results,columns=["bidid","bidpice"])
    return results





if __name__=="__main__":
    # MAIN:
    st=time.time()
    training_path = r"../dataset/train.csv"
    validation_path = r"../dataset/validation.csv"
    test_path = r"../dataset/test.csv"

    # Extracting data:
    training_events, training_labels = load_data(training_path)
    #validation_events, validation_labels = load_data(validation_path)
    # training model
    models_CTR= trainCTRmodel(training_events, training_labels)
    # Evaluating this model:
    perf, y_true, y_pred, adv_true, adv_pred = testCTRmodel(models_CTR, validation_path)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    print auc

    # Evaluating each model:
    adv_auc={}
    for key in adv_pred.keys():
        fpr, tpr, thresholds = metrics.roc_curve(adv_true[key], adv_pred[key])
        auc = metrics.auc(fpr, tpr)
        adv_auc[key]=auc
        print ("adv: %s, auc: %0.3f" %(key,auc))

    # Tuning the parameters:
    all_res_3=[]
    #for c in np.arange(10,30,10):
    for c in np.arange(15, 30, 5):
        for lamb in [1e-10,5e-10,1e-9   ]:
            final_res,_,_=RTB_simulation(models_CTR,validation_path, winning_rate="w2",c=c ,lambda_const=lamb,data_type="Validation")
            all_res_3.append(final_res)

    final_ctr, results, res_perf=RTB_simulation(models_CTR, validation_path,winning_rate="w1", lambda_const=5e-4, c=60, data_type="Validation")


    # Best: lambda_const=7e-4, c=85
        # optimal: [5.2e-4, 10e-4, 50e-4, 1e-3]:


    #optimal_c=find_optimal_c(models_best_CTR,validation_path,training_path)
    # optimal_lambda = find_optimal_lambda(models_best_CTR, validation_path, training_path)
    #optimal_params = find_optimal_params(models_best_CTR, validation_path, training_path)


    #results_test = RTB_simulation_test(models_best_CTR, test_path, c=60,lambda_const=5.2e-3,data_type="Test")

    #optimal_c=find_optimal_c(models_best_CTR,validation_path,training_path)
    # optimal_lambda = find_optimal_lambda(models_best_CTR, validation_path, training_path)
    #optimal_params = find_optimal_params(models_best_CTR, validation_path, training_path)

    #results_df=results_to_csv(results_test)
    #x = pd.DataFrame.from_dict(perf)

    # perf_cl={}
    # for key in dict_res.keys():
    #     perf_cl[key]=dict_res[key][2]
    # ctr_cl = {}
    # for key in dict_res.keys():
    #     ctr_cl[key] = dict_res[key][0]
    #
    #
    # for i in dict_res.keys():
    #     x=pd.DataFrame.from_dict(dict_res[i][2]).T
    #     x.to_csv(str(i)+".csv")

