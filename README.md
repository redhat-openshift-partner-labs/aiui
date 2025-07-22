setup local mysql for testing

```shell
podman run -d -it --replace --name mysql-container -e MYSQL_ROOT_PASSWORD=rootpass -e MYSQL_USER=mcpuser -e MYSQL_PASSWORD=mcpuser --network host -p 3306:3306 docker.io/library/mysql:latest \
&& sleep 30 \
&& podman exec mysql-container mysql -uroot -prootpass -e "GRANT ALL PRIVILEGES ON *.* TO 'mcpuser'@'%' WITH GRANT OPTION; FLUSH PRIVILEGES;
```

import the mysql_dummy_data.sql file (you'll need mysql client)
```shell
mysql -h localhost --protocol tcp -u mcpuser -pmcpuser < mysql_dummy_data.sql
```

create virtual env
```shell
python -m venv .venv
source .venv/bin/activate

# ensure you are using the .venv pip
which pip

pip install -r requirements.txt
```

```shell
# run the google genai-toolbox mcp server
toolbox --log-level DEBUG --tools-file "tools.yaml"
```

update the config.yaml ollama section to point to your local ollama instance and available model

run the application
```shell
streamlit run app.py
```