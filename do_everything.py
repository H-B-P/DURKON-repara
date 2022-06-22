import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,2], "cat1":['a','a','a','a','b','b','b','b'], "cat2":['c','c','d','d','c','d','e','d'], "y":[0,0,0,1,0,0,0,1]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = prep.prep_model(df, "y", ["cat1","cat2"],["cont1","cont2"], defaultValue=0)

#Classifier-ify model
model["BASE_VALUE"] = calculus.Logit_delink(model["BASE_VALUE"])
model["featcomb"] = "addl"

print(model)

model = actual_modelling.train_model(df, "y", 500, 0.1, model, link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0)

print(model)

df["PredComb"]=misc.predict(df,model)

sugImps=[]
sugFeats=[]
sugTypes=[]

for i in range(len(cats)):
 for j in range(i+1, len(cats)):
  trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
  prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=0)
  trialmodel = actual_modelling.train_model(df, "y", 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0)
  
  print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0), trialmodel)
  
  sugFeats.append(cats[i]+" X "+cats[j])
  sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
  sugTypes.append("catcat")

for i in range(len(cats)):
 for j in range(len(conts)):
  trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
  prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=0)
  trialmodel = actual_modelling.train_model(df, "y", 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0)
  
  print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0), trialmodel)
  
  sugFeats.append(cats[i]+" X "+conts[j])
  sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
  sugTypes.append("catcont")

for i in range(len(conts)):
 for j in range(i+1, len(conts)):
  trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
  prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=0)
  trialmodel = actual_modelling.train_model(df, "y", 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0)
  
  print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0), trialmodel)
  
  sugFeats.append(conts[i]+" X "+conts[j])
  sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
  sugTypes.append("contcont")

sugDf = pd.DataFrame({"feat":sugFeats,"imp":sugImps})
sugDf = sugDf.sort_values(['imp'], ascending=False)
print(sugDf)