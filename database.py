import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime,LargeBinary

SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    count = Column(Integer, default=0)

    trainings = relationship("TrainingUser", back_populates="user")
    
    def to_dataframe(self) -> pd.DataFrame:
        data = {
        "id": [self.id],
        "count": [self.count]}
        return pd.DataFrame(data)
    
class TrainingUser(Base):
    __tablename__ = "trainings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True,)
    id_best = Column(Integer)
    user_id = Column(String, ForeignKey("users.id"))
    
    train_time = Column(DateTime, default=datetime.utcnow)
    train_score = Column(Float)
    test_score = Column(Float)
    column_data = Column(String)
    trained_model = Column(LargeBinary)
    user = relationship("User", back_populates="trainings")
    def to_dataframe(self) -> pd.DataFrame:
        data = {
        "id": [self.id_best],
        "user_id": [self.user_id],
        "train_time": [self.train_time],
        "train_score": [self.train_score],
        "test_score": [self.test_score],
        "column_data": [self.column_data],
        "trained_model": [self.trained_model]}
        return pd.DataFrame(data)
