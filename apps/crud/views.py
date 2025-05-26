from apps.crud.forms import UserForm
from apps.app import db
from apps.crud.models import User
from flask import Blueprint, render_template, redirect, url_for

crud = Blueprint(
  "crud",
  __name__,
  template_folder="templates",
  static_folder="static",
)

@crud.route("/")
def index():
  return render_template("crud/index.html")

@crud.route("/user/new", methods=["GET", "POST"])
def create_user():
  # UserFormをインスタンス化する
  form = UserForm()
  # フォームの値をバリデートする
  if form.validate_on_submit():
    # ユーザーを作成する
    user = User(
      username=form.username.data,
      email=form.email.data,
      password=form.password.data,
    )
    # ユーザーを追加してコミットする
    db.session.add(user)
    db.session.commit()
    # ユーザーの一覧画面へリダイレクトする
    return redirect(url_for("crud.users"))
  return render_template("crud/create.html", form=form)