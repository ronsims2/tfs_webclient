from flask import Flask, request, json, abort
from flask_swagger_ui import get_swaggerui_blueprint
from os.path import join
from os import getcwd

from google.protobuf.json_format import MessageToJson
from google.protobuf.service import RpcException
from werkzeug.exceptions import HTTPException

from utils import read_labels, is_valid_upload, make_inference, get_metadata, get_found_labels
from PIL import Image
import numpy as np

# Read labels for model sourced from https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv
LABEL_PATH = join(getcwd(), 'static', 'objectinfo150.csv')
# For small things its just easier to keep thing in memory
labels = read_labels(LABEL_PATH)

# Domain of the tfs server
DOMAIN = '127.0.0.1'
# Port to communicate with grpc
PORT = 8500
MODEL_NAME = 'deeplabv3'
PRED_TIMEOUT = 15.0

# This should come from a common module shared from the model fixer
GRAPH_SIGNATURE_KEYS = {
    'SIGNATURE_DEF_INPUT_NAME': 'input_image_data',
    'SIGNATURE_DEF_OUTPUT_NAME': 'output_data',
    'SIGNATURE_DEF_MAP_NAME': 'segmentation_data'
}

SWAGGER_PATH = '/swagger'
UPLOAD_PATH = join(getcwd(), 'uploads')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
swagger_page = get_swaggerui_blueprint(SWAGGER_PATH, '/static/swagger.yml')
app.register_blueprint(swagger_page)

@app.route('/')
def index():
    return 'hello world'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'upload' not in request.files:
        return 'User did not upload a file', 400

    file = request.files['upload']
    if is_valid_upload(file):
        img = Image.open(file)
        img_data = np.array([np.array(img)])

        pred_result = make_inference(img_data, DOMAIN,
                                     PORT, MODEL_NAME,
                                     GRAPH_SIGNATURE_KEYS['SIGNATURE_DEF_MAP_NAME'],
                                     GRAPH_SIGNATURE_KEYS['SIGNATURE_DEF_INPUT_NAME'],
                                     GRAPH_SIGNATURE_KEYS['SIGNATURE_DEF_OUTPUT_NAME'],
                                     PRED_TIMEOUT)

        pred_labels = get_found_labels(pred_result, labels)

        return {
            'pred_labels': pred_labels
        }
    else:
        abort(403)


    @app.route('/sys/<path:frag>')
    def get_system_info(frag):
        if frag == 'graph-metadata/signature_def':
            res = get_metadata(MODEL_NAME, 'signature_def')
            return MessageToJson(res)
        elif frag == 'graph-metadata/signature_def':
            pass
        else:
            abort(403)


    @app.errorhandler(Exception)
    def handle_errors(e):
        err_desc = 'Big fail!'
        err_code = 403
        show_err = app.debug

        if isinstance(RpcException) and show_err:
            err_desc = e.details()

        if isinstance(HTTPException) and show_err:
            err_desc = e.description

        return json.dumps({
            'description': err_desc
        }), err_code


if __name__ == '__main__':
    app.run()
