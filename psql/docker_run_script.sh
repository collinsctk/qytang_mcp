docker run --name qytpg \
-e POSTGRES_USER=qytangdbuser \
-e POSTGRES_PASSWORD=Cisc0123 \
-e POSTGRES_DB=qytangdb \
-p 5432:5432 \
-v $PWD/pg_hba.conf:/etc/postgresql/pg_hba.conf \
-v $PWD/postgresql.conf:/etc/postgresql/postgresql.conf \
-d postgres:13.7 \
-c config_file=/etc/postgresql/postgresql.conf \
-c hba_file=/etc/postgresql/pg_hba.conf