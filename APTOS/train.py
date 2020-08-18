from model import BaseCNNModel
from torch_trainer import Trainer
import pandas as pd 
import numpy as np 
import config

if __name__ == "__main__":

    num_classes = config.NUM_CLASSES
    model = BaseCNNModel(num_classes)

    lr = config.LR
    num_epochs = config.NUM_EPOCHS
    log_steps = config.LOG_STEPS
    is_cv = config.isCV
    n_folds = config.N_FOLDS
    nrows = config.NROWS
    
    datafile = config.TRAIN_CSV_FILE

    trainer = Trainer(model, 
                      datafile,
                      lr=lr,
                      max_epochs=num_epochs,
                      log_steps=log_steps,
                      validation=False,
                      n_folds=n_folds, 
                      cv=is_cv,
                      nrows=nrows)

    # cv_perf_df, cv_results = trainer.cv_fit()
    # print(cv_perf_df)
    # print(cv_results)
    results = trainer.simple_fit()
    # trainer.plot_training_performance(results)
    
    model_path = 'model_2020-08-18.pt'
    model_scores = trainer.score_model(model_path, config.TEST_CSV_FILE)
    
    model_out = {}
    for j in model_scores.keys():
        model_out[j] = np.argmax(model_scores[j])
        
    test_df  = pd.DataFrame(model_out.keys(), columns=['id_code'])
    test_df['diagnosis_pred'] = model_out.values()
    
    # load submission
    sub_df = pd.read_csv(config.SUB_FILE)
    sub_df = sub_df.merge(test_df, how='left', on=['id_code'])
    sub_df = sub_df.drop(['diagnosis'], axis=1)
    sub_df.rename(columns={"diagnosis_pred": "diagnosis"})
    
    # save submission
    sub_fname = config.SUB_FOLDER + 'naive_sub.csv'
    sub_df.to_csv(sub_fname, index=False)
    
    print("done...")
    