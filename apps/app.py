from pathlib import Path
from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from apps.config import config

db = SQLAlchemy()
csrf = CSRFProtect()

login_manager = LoginManager()
login_manager.login_view = "auth.signup"
login_manager.login_message = ""

def create_app(config_key):
  app = Flask(__name__)

  app.config.from_object(config[config_key])
  app.config.from_mapping(
    SECRET_KEY="2AZSMss3p5QPbcY2hBsJ",
    SQLALCHEMY_DATABASE_URI=
      f"sqlite:///{Path(__file__).parent.parent / 'local.sqlite'}",
    SQLALCHEMY_TRACKMODIFICATIONS=False,
    SQLALCHEMY_ECHO=True,
    WTF_CSRF_SECRET_KEY="AuwzyszU5sugKN7KZs6f",
  )

  csrf.init_app(app)
  db.init_app(app)
  Migrate(app, db)
  login_manager.init_app(app)

  from apps.crud import views as crud_views
  app.register_blueprint(crud_views.crud, url_prefix="/crud")

  from apps.auth import views as auth_views
  app.register_blueprint(auth_views.auth, url_prefix="/auth")

  return app

