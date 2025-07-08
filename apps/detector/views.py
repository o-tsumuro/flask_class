import random
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from apps.app import db
from apps.crud.models import User
from apps.detector.forms import DetectorForm, UploadImageForm
from apps.detector.models import UserImage, UserImageTag
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

dt = Blueprint("detector", __name__, template_folder="templates")


@dt.route("/")
def index():
    user_images = (
        db.session.query(User, UserImage)
        .join(UserImage)
        .filter(User.id == UserImage.user_id)
        .all()
    )

    user_image_tag_dict = {}
    for user_image in user_images:
        user_image_tags = (
            db.session.query(UserImageTag)
            .filter(UserImageTag.user_image_id == user_image.UserImage.id)
            .all()
        )
        user_image_tag_dict[user_image.UserImage.id] = user_image_tags

    detector_form = DetectorForm()

    return render_template(
        "detector/index.html",
        user_images=user_images,
        user_image_tag_dict=user_image_tag_dict,
        detector_form=detector_form,
    )


@dt.route("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@dt.route("/upload", methods=["GET", "POST"])
@login_required
def upload_image():
    form = UploadImageForm()
    if form.validate_on_submit():
        file = form.image.data
        ext = Path(file.filename).suffix
        image_uuid_file_name = str(uuid.uuid4()) + ext

        image_path = Path(current_app.config["UPLOAD_FOLDER"], image_uuid_file_name)
        file.save(image_path)

        user_image = UserImage(user_id=current_user.id, image_path=image_uuid_file_name)
        db.session.add(user_image)
        db.session.commit()

        return redirect(url_for("detector.index"))
    return render_template("detector/upload.html", form=form)


# --- ヘルパー関数の修正 ---

def make_color(labels):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in labels]
    color = random.choice(colors)
    return color


def make_line(result_image):
    line = round(0.002 * max(result_image.shape[0:2])) + 1
    return line


# cv2を返さないように修正
def draw_lines(c1, c2, result_image, line, color):
    cv2.rectangle(result_image, c1, c2, color, thickness=line)


# cv2を引数と返り値から削除
def draw_texts(result_image, line, c1, color, labels, label):
    display_txt = f"{labels[label]}"
    font = max(line - 1, 1)
    t_size = cv2.getTextSize(display_txt, 0, fontScale=line / 3, thickness=font)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(result_image, c1, c2, color, -1)
    cv2.putText(
        result_image,
        display_txt,
        (c1[0], c1[1] - 2),
        0,
        line / 3,
        [225, 225, 225],
        thickness=font,
        lineType=cv2.LINE_AA,
    )


# --- 物体検知のメイン関数の修正 ---

def exec_detect(target_image_path):
    labels = current_app.config["LABELS"]
    
    # 画像を読み込み、RGBに変換する
    image = Image.open(target_image_path).convert("RGB")
    
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # weights_only=Falseを指定してモデルを読み込む
    model = torch.load(
        Path(current_app.root_path, "detector", "model.pt"), weights_only=False
    )
    model = model.eval()
    output = model([image_tensor])[0]

    tags = []
    result_image = np.array(image.copy())
    
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score > 0.5 and labels[label] not in tags:
            color = make_color(labels)
            line = make_line(result_image)
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
            
            # cv2 = を削除
            draw_lines(c1, c2, result_image, line, color)
            # cv2 = と引数のcv2を削除
            draw_texts(result_image, line, c1, color, labels, label)
            
            tags.append(labels[label])

# 検知後の画像ファイル名を生成する
    detected_image_file_name = str(uuid.uuid4()) + ".jpg"

    # --- ★★★ ここから修正 ★★★ ---

    # 画像のフルパスを先に変数に入れる
    detected_image_file_path = str(
        Path(current_app.config["UPLOAD_FOLDER"], detected_image_file_name)
    )
    
    # 保存する画像データ（色空間を変換）
    image_to_save = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # imwriteの返り値を受け取り、成功したかチェックする
    success = cv2.imwrite(detected_image_file_path, image_to_save)

    # コンソールに結果を表示
    if success:
        print(f"【成功】画像が保存されました: {detected_image_file_path}")
    else:
        print(f"【失敗】画像の保存に失敗しました。パス: {detected_image_file_path}")
    
    # --- ★★★ ここまで修正 ★★★ ---

    return tags, detected_image_file_name


def save_detected_image_tags(user_image, tags, detected_image_file_name):
    user_image.image_path = detected_image_file_name
    user_image.is_detected = True
    db.session.add(user_image)

    for tag in tags:
        user_image_tag = UserImageTag(user_image_id=user_image.id, tag_name=tag)
        db.session.add(user_image_tag)

    db.session.commit()


@dt.route("/detect/<string:image_id>", methods=["POST"])
@login_required
def detect(image_id):
    user_image = (
        db.session.query(UserImage).filter(UserImage.id == image_id).first()
    )
    if user_image is None:
        flash("物体検知対象の画像が存在しません。")
        return redirect(url_for("detector.index"))

    target_image_path = Path(current_app.config["UPLOAD_FOLDER"], user_image.image_path)

    tags, detected_image_file_name = exec_detect(target_image_path)

    try:
        save_detected_image_tags(user_image, tags, detected_image_file_name)
    except SQLAlchemyError as e:
        flash("物体検知処理でエラーが発生しました。")
        db.session.rollback()
        current_app.logger.error(e)
        return redirect(url_for("detector.index"))

    return redirect(url_for("detector.index"))
