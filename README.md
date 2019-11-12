### Vanilla

- Install the requirements and setup the prod environment.

	`make install && make prod`

- Create the database.

	`python manage.py initdb`

- Run the application.

	`python manage.py runserver`

- Navigate to `localhost:5000`.

Deploying in Heroku

$ git clone https://github.com/kapilrajput22/prediction_model.git 
$ cd prediction_model

$ python3 -m venv getting-started
$ pip install -r requirements.txt

$ createdb prediction_model

$ python manage.py migrate
$ python manage.py collectstatic

$ heroku local
Your app should now be running on localhost:5000.

Deploying to Heroku
$ heroku create
$ git push heroku master

$ heroku run python manage.py migrate
$ heroku open
