# The Big Picture
This project allows you to develop customs services to interface with a model hosted in TensorFlow Serving.

This sample project uses a [Deeplabv3](https://modelzoo.co/model/deeplab) model.
# Get Started
1. Clone tensorFlow Serving (TFS) if you haven't already.

   ```git clone https://github.com/tensorflow/serving.git```

2. Optionally, rename the `serving` directory to something more descriptive like `tf_serving`.
3. Navigate to the TFS directory `cd tf_serving`.
5. [Export a model for TSF](https://www.tensorflow.org/tfx/serving/serving_basic). 
6. Run the TFS container and serve the model.  Note: the exported model must be in a directory with a numeric name. 
    ```
    docker run -p 8500:8500 --mount type=bind,source=/Users/tace105/Documents/tfutils/saves/375d57e9-fd0b-4f08-b517-a15b32df86f2/1/,target=/models/deeplabv3/1/ -e MODEL_NAME=deeplabv3 -t tensorflow/serving &
   ```
7. Start the tfs web client from the root of the project folder:

   ```source venv/bin/activate && python app.py```

8. Visit the swagger page to get started: http://127.0.0.1:5000/swagger.


# Resources
* Tensorflow Serving tutorial to [get a model server up and running](https://www.tensorflow.org/tfx/serving/serving_basic).