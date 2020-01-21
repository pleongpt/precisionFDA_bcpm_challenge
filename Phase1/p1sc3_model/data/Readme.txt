p1sc3_model
===========
$ ./p1sc3_model.py -h
usage: p1sc3_model.py [-h] [-block] [-phenotype {0,1}]
                      [-m {log_reg,elastic,lin_SVM,rbf_SVM,RF,xgboost,classifiers_only,soft_voting,all}]
                      [-prefilter {0,1}] [-bagging {0,1}] [-smote {0,1}]
                      [-predict] [-rna_dna_CN_predict RNA_DNA_CN_PREDICT]
                      [-pheno_predict PHENO_PREDICT]

optional arguments:
  -h, --help            show this help message and exit
  -block                Enable a plot to block the running process until the
                        plot is closed.
  -phenotype {0,1}      Specify whether to include the phenotype data in the
                        modeling. (0: Do not include (default); 1: include).
  -m {log_reg,elastic,lin_SVM,rbf_SVM,RF,xgboost,classifiers_only,soft_voting,all}, --model {log_reg,elastic,lin_SVM,rbf_SVM,RF,xgboost,classifiers_only,soft_voting,all}
                        Specify a specific classifier model. Default
                        "soft_voting".
  -prefilter {0,1}      Specify to include the attribute prefilters from P1SC1
                        and P1SC2 (0: No; 1: Yes (default)).
  -bagging {0,1}        Specify whether bagging should be applied to each
                        classifier (0: No; 1: Yes (default)).
  -smote {0,1}          Specify whether SMOTE should be used to increase
                        minority sample instances (0: No; 1: Yes (default)).
  -predict              Enable prediction based on a trained model.
  -rna_dna_CN_predict RNA_DNA_CN_PREDICT
                        Specify the input RNA DNA CN filename for generating a
                        prediction.
  -pheno_predict PHENO_PREDICT
                        Specify the phenotype filename for generating a
                        prediction.

-----------------------------------------------------------------------------------------------------
The following are commands for various purposes:

# To generate models for log_reg, elastic, lin_SVM, rbf_SVM, RF and xgboost.
# Bagging and SMOTE are enabled.  Phenotype data is included. Attribute
# prefilters from P1SC1 and P1SC2 xgboost are enabled.
# Log: p1sc3_classifiers_only_bagging_1_smote_1_phenotype1_prefilter1.txt

p1sc3_model.py -m classifiers_only -bagging 1 -smote 1 -phenotype 1 -prefilter 1

# To generate model for soft_voting.
# Bagging and SMOTE are enabled.  Phenotype data is included. Attribute
# prefilters from P1SC1 and P1SC2 xgboost are enabled.
# Log: p1sc3_soft_voting_bagging_1_smote_1_phenotype1_prefilter1.txt

p1sc3_model.py -m soft_voting -bagging 1 -smote 1 -phenotype 1 -prefilter 1

# Use the soft_voting model to predict the survival status based on a
# set of RNA-DNA-CN and phenotype test data. For example: Test data files 
# rna_DNA_CN_test.tsv and pheno_test.tsv under the data folder.
# Log: p1sc3_soft_voting_predict.txt

p1sc3_model.py -m soft_voting -prefilter 1 -predict -rna_dna_CN_predict ./data/rna_dna_CN_test.tsv -pheno_predict ./data/pheno_test.tsv

# Use the xgboost model to predict the survival status based on a
# set of RNA and phenotype test data. For example: Test data files 
# rna_dna_CN_test.tsv and pheno_test.tsv under the data folder.
# Log: p1sc3_xgboost_predict.txt

p1sc3_model.py -m xgboost -prefilter 1 -predict -rna_dna_CN_predict ./data/rna_dna_CN_test.tsv -pheno_predict ./data/pheno_test.tsv





