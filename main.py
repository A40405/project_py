import pandas as pd
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.responses import HTMLResponse
from typing import Optional
from database import Base,SessionLocal,engine
from crud import create_user, create_training, get_users, get_train, get_weight_train


# Tạo bảng
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create user
@app.post("/users/{user_id}")
async def create_user_route(user_id: str, db: Session = Depends(get_db)):
    return create_user(db=db, user_id=user_id)

# Train model
@app.post("/users/{user_id}/trainings")
async def train_model(
    user_id: str, 
    db: Session = Depends(get_db),
    file_train: UploadFile = File(..., description=".csv or .xlsx"),
    file_test: UploadFile = File(..., description=".csv or .xlsx")):
    training = create_training(db=db, user_id=user_id, file_train=file_train, file_test=file_test)
    return {
        "id": training.id_best, 
        'user_id': training.user_id,
        'train_time': training.train_time,
        'train_score': training.train_score,
        'test_score': training.test_score,
        'column_data': training.column_data,
    }
    

# Get all user as a dataframe
@app.get("/users", response_class=HTMLResponse)
async def read_users(skip: Optional[int] = 0, limit: Optional[int] = 100, db: Session = Depends(get_db)):
    users = get_users(db=db, skip=skip, limit=limit)
    users_df = pd.concat([u.to_dataframe() for u in users])
    return users_df.to_html(index=False)


 
# Get all trainings of a user as a dataframe
@app.get("/users/{user_id}/trained", response_class=HTMLResponse)
async def read_train_history(user_id: str, db: Session = Depends(get_db)):
    train = get_train(db=db, user_id=user_id)
    trainings_df = pd.concat([t.to_dataframe() for t in train])
    return trainings_df.to_html(index=False)

    
# Predict model trained
@app.post("/users/{user_id}/{trained_id}/predict")
async def read_weight(user_id: str, trained_id: str,
                      file_predict: UploadFile = File(..., description=".csv or .xlsx"),
                      db: Session = Depends(get_db)):
    train = get_weight_train(db=db, user_id=user_id,trained_id=trained_id,file_predict = file_predict)
    return {"target":str(train)}
        
@app.get("/")
async def hello():
    html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>My App</title>
            <style>
                h1 {
                    font-size: 36px;
                }
                h2 {
                    
                }
                a {
                    font-size: 25px
                }
                .box {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 10px;
                    margin-bottom:20px;
                    height: 80px;
                    text-align: center;
                }
                .box:hover {
                    background-color: #7FFF00 !important;
                }
            </style>
        </head>
        <body>
            <center>
            <h1>Bài cuối kì</h1>
            <h1>Môn: Lập Trình Python</h1>
            <h2 style = "margin-bottom:20px; margin-top:200px; text-align: right;">Sinh viên thực hiện:</h2>
            <h2 style = " text-align: right;">A40405 Bùi Hữu Huấn</h2>
            <div  style=" display: inline-block; padding: 10px;">
                    <div class="box" style="background-color: #f4a70e; text-align: center; display: flex; justify-content: center;min-width: 120px;">
                        <a href="/docs" target="_blank" style="font-weight: bold; color: #ffffff;">Link app</a>
                    </div>
            </div>
            </center>
        </body>
        </html>
    """
    return HTMLResponse(content=html_content, status_code=200)