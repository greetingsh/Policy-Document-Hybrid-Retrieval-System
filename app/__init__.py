from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # 允许跨域，方便前端对接

    # 注册路由
    from app.routes.doc import doc_bp
    app.register_blueprint(doc_bp, url_prefix="/api/doc")

    # 注册AI路由
    from app.routes.ai import ai_bp
    app.register_blueprint(ai_bp, url_prefix="/api/ai")

    return app
