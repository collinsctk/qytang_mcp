# https://www.cnblogs.com/lsdb/p/9835894.html

from sqlalchemy import create_engine, orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import datetime

engine = create_engine('postgresql+psycopg2://qytangdbuser:Cisc0123@196.21.5.218/qytangdb')


Base = orm.declarative_base()


class User(Base):
    __tablename__ = 'users'
    # PSQL的Data Type
    # https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#postgresql-data-types
    id = Column(Integer, primary_key=True)
    username = Column(String(64), nullable=False, index=True)
    password = Column(String(64), nullable=False)
    realname = Column(String(64), nullable=True)
    email = Column(String(64), nullable=False, index=True)

    def __repr__(self):
        return f"{self.__class__.__name__}(username: {self.username} | email: {self.email})"


if __name__ == '__main__':
    # checkfirst=True，表示创建表前先检查该表是否存在，如同名表已存在则不再创建。其实默认就是True
    Base.metadata.create_all(engine, checkfirst=True)