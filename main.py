import pickle
import json
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from flask import Flask, render_template, request
from flask_restx import Api, Resource
from flask_cors import CORS
import sys
import os
sys.path.append(os.getcwd())

# with open('kmeans_500_model_final.pkl', 'rb') as file1:
#     kmeans_model = pickle.load(file1)
    
with open('data/ols_model_final.pkl', 'rb') as file2:
   ols_model = pickle.load(file2)

with open('data/result_bygroup_final.json', 'r') as file3:
    json_data = json.load(file3)
    
with open('data/knn_2_model_final.pkl', 'rb') as file4:
    knn_model = pickle.load(file4)
    
df = pd.read_csv('data/df_formodel_final.csv', index_col=0)
df = df[['year', 'condition', 'sellingprice', 'odometer', 'mmr']]

def pca(df):
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data=principalComponents, columns = ['pc1'])
    return principalDf

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

@app.route('/', methods=['GET', 'POST']) 

def make_ml_answer(*args,**kwargs):
    if request.method == 'POST':
        # HTML에서 POST 요청으로 받은 데이터 처리
        year = request.form.get('year')
        condition = request.form.get('condition')
        odometer = request.form.get('odometer')
        mmr = request.form.get('mmr')
        sellingprice = request.form.get('sellingprice')
        
        check_year = int(round(float(year)))
        check_condition = int(round(float(condition)))
        check_odometer = int(round(float(odometer)))
        check_mmr = int(round(float(mmr)))
        check_sellingprice = int(round(float(sellingprice)))

        new_data = {'year': check_year, 'condition': check_condition, 'sellingprice': check_sellingprice, 'odometer': check_odometer, 'mmr': check_mmr}

        # 정규화 진행
        df.loc[len(df)] = new_data
        minmax_scaler = MinMaxScaler() # MinMaxScaler 
        minmax_scaler.fit(df) # 정규화를 수행 함수 생성
        transformed = minmax_scaler.transform(df) 
        df_scaled= pd.DataFrame(transformed)
        df_scaled.columns = df.columns
        
        # pca 진행
        df_pca = pca(pd.DataFrame(df_scaled.loc[len(df_scaled)-1]).T.reset_index(drop=True))
        x_train_final = np.array(df_pca['pc1']).reshape(-1, 1)

        # knn 모델 예측
        predict_group = str(int(knn_model.predict(x_train_final)))
        # ols 모델 예측
        predict_price = ols_model.predict(df.loc[(len(df)-1):,['year', 'condition', 'mmr', 'odometer']].reset_index(drop=True))
        result_predict_price = str(round(predict_price[0],2))
        
        result_year = json_data[predict_group]['count_year']
        result_condition = json_data[predict_group]['mean_condition']
        result_make = json_data[predict_group]['count_make']
        result_odometer = json_data[predict_group]['mean_odometer']
        result_mmr = json_data[predict_group]['mean_mmr']
        result_sellingprice = json_data[predict_group]['mean_sellingprice']
        
        
        output = {
            'year' : 2023 - result_year,
            'make' : result_make,
            'condition' : round(result_condition, 2),
            'odometer' : round(result_odometer, 2) ,
            'mmr' : round(result_mmr, 2),
            'sellingprice' :round(result_sellingprice, 2),
            'predict_price' : result_predict_price
        }                        
    
        return  output
    else:
        return  "Bad Request"

if __name__ == '__main__':
    app.run(debug=True)



