from sqlalchemy.orm import sessionmaker
from create_table import User, engine


Session = sessionmaker(bind=engine)
session = Session()

# 插入一个条目
user1 = User(username='qinke1',
             password='12334',
             email='collinsctk@qytang.com')
session.add(user1)
session.commit()



# 一次性插入多个条目
# session.add_all([new_user, user2])
# session.commit()
