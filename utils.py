import csv
import filetype
import grpc
from PIL import Image
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc, get_model_metadata_pb2
import tensorflow as tf
import numpy as np


def read_labels(loc: str):
    item_labels = []
    with open(loc, newline='') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            item_labels.append(row[5])

    return item_labels


def is_valid_upload(f):
    guess = filetype.guess(f.read())
    return guess.mime in ['image/gif', 'image/jpeg', 'image/png']


# sig_name, sig_in, sig_out should come from the model formatted for tsf serving
def make_inference(img_data, domain, port, model_name, sig_name, sig_in, sig_out, pred_timeout):
    proto = tf.make_tensor_proto(img_data)
    rpc_channel = grpc.insecure_channel('{}:{}'.format(domain, port))
    service_stub = prediction_service_pb2_grpc.PredictionServiceStub(rpc_channel)
    rpc_req = predict_pb2.PredictRequest()
    rpc_req.model_spec.name = model_name
    rpc_req.model_spec.signature_name = sig_name
    rpc_req.inputs[sig_in].CopyFrom(proto)

    res = service_stub.Predict(rpc_req, pred_timeout)

    return res.outputs[sig_out]


def get_metadata(domain, port, model_name, field_name):
    rpc_channel = grpc.insecure_channel('{}:{}'.format(domain, port))
    service_stub = prediction_service_pb2_grpc.PredictionServiceStub(rpc_channel)
    rpc_req = get_model_metadata_pb2.GetModelMetadataRequest()
    rpc_req.model_spec.name = model_name
    rpc_req.metadata_field.append(field_name)

    res = service_stub.GetModelMetadata(rpc_req, 5)

    return res


def get_found_labels(pred, labels):
    segmentation_map = tf.make_ndarray(pred).astype(np.uint8)
    img_seg_map = Image.fromarray((segmentation_map[0]))
    found = np.unique(segmentation_map).tolist()
    found_classes = [labels[label] for label in found]

    return found_classes
