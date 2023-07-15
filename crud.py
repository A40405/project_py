from fastapi import status 
from fastapi import HTTPException
import database
import numpy as np
import pandas as pd 
from datetime import datetime
from sqlalchemy.orm import Session
from model import DecisionTree
from typing import Optional
import pickle
import json



#def error
def is_csv_file(df_train,df_test):
    if not df_train.filename.endswith(('.csv', '.xlsx')):
        return "File df_train phải có định dạng csv"
    if not df_test.filename.endswith(('.csv', '.xlsx')):
        return "File df_test phải có định dạng csv"
    return True
def check_data_sizes(df_train, df_test):
    if df_train.shape[1] == df_test.shape[1]:
        return True
    return '2 data cỡ không hợp lệ!'
def checkin_dataframe(df, file_name):
    if df.isnull().values.any():
        return f'File {file_name} chứa giá trị rỗng'
    for i in df.dtypes:
        if not np.issubdtype(i, np.number):
            print(i)
            return f'File {file_name} có kiểu dữ liệu không hợp lệ'
    return True

# def use 
def create_user(db: Session, user_id: str):
    db_user = db.query(database.User).filter(database.User.id == user_id).first()
    if db_user:
        suggested_name = np.sort(np.random.choice(np.arange(100, 10000), size=5, replace=False)).astype(np.str_)
        result = "Gợi ý một số User khả dụng: " + ", ".join([user_id + name for name in suggested_name]) + " ..."
        return f"User: {user_id} đã tồn tại! " + result
    else:
        db_user = database.User(id=user_id)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    return f"Tạo User: {user_id} thành công"

# def train 
def create_training(db: Session, user_id: str, file_train: str, file_test: str):
    # Check if the user exists
    user = db.query(database.User).filter(database.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with id={user_id} not found")
    
    # check: not .csv
    if is_csv_file(file_train,file_test) is not True:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,detail=is_csv_file(file_train,file_test))
    
    df_train = pd.read_csv(file_train.file)
    df_test = pd.read_csv(file_test.file)   
    
    # Check shape 
    if check_data_sizes(df_train, df_test) is not True:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,detail=check_data_sizes(df_train, df_test))
    # Check type and missing_values
    if checkin_dataframe(df_train, 'df_train') is not True:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,detail=checkin_dataframe(df_train, 'df_train'))
    if checkin_dataframe(df_test, 'df_test') is not True:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,detail=checkin_dataframe(df_test, 'df_test'))

    
    # Fit the DecisionTree model
    X_train = df_train.iloc[:,:-1].values
    y_train = df_train.iloc[:,-1].values
    model = DecisionTree()
    model.fit(X_train, y_train)
    
    X_test = df_test.iloc[:,:-1].values
    y_test = df_test.iloc[:,-1].values
    # Create a new Training instance
    new_training = database.TrainingUser(
        id_best = user.count + 1,
        user_id=user_id,
        train_time=datetime.now(),
        train_score=model.score(X_train, y_train),
        test_score=model.score(X_test, y_test),
        column_data = str(df_train.columns.values),
        trained_model= pickle.dumps(model))
    
    # model_new = pickle.loads(new_training.trained_model)
    # Save the Training instance to the database
    db.add(new_training)
    db.commit()
    db.refresh(new_training)
    
    # Update the User instance with the new Training count
    user.count += 1
    db.add(user)
    db.commit()  
    return new_training

# def get database 
def get_users(db: Session, skip: Optional[int] = 0, limit: Optional[int] = 100):
    users = db.query(database.User).offset(skip).limit(limit).all()
    if not users:
        raise HTTPException(status_code=status.HTTP_200_OK, detail = f"Cơ sở dữ liệu không có nổi {skip} người dùng")
    return users

def get_train(db: Session, user_id: str):
    db_user = db.query(database.User).filter(database.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User with id={user_id} not found")
    train = db_user.trainings
    if not train:
        raise HTTPException(status_code=status.HTTP_200_OK, detail=f"User with id={user_id} never trained")
    return train 
def get_weight_train(db: Session, user_id: str,trained_id: str,file_predict: str):
    db_trained_id = db.query(database.TrainingUser).filter(
        database.TrainingUser.user_id == user_id,
        database.TrainingUser.id_best == trained_id
    ).first()

    if not db_trained_id:
        raise HTTPException(status_code=status.HTTP_200_OK, detail=f"User with id={user_id} has no {trained_id}th trained")
    if not file_predict.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File df_predict phải có định dạng csv hoặc xlsx")
    df_predict = pd.read_csv(file_predict.file)
    
    if checkin_dataframe(df_predict, 'df_predict') is not True:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,detail=checkin_dataframe(df_predict, 'df_predict'))
    
    columns_string = db_trained_id.column_data
    columns_data = columns_string[1:-1].replace("\n","").split("' ")
    columns_data = [item.replace("'", "") for item in columns_data]
    if (df_predict.columns.tolist()) != (columns_data[:-1]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File df_predict có dữ liệu không khớp với dữ liệu được training trước đó")
    
    model = pickle.loads(db_trained_id.trained_model)
    return model.predict(df_predict.values)
    
    