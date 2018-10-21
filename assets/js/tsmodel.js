/*
JS version of emotion recognition model from SavedModel
*/

import * as tf from '@tensorflow/tfjs';

const MODEL_URL = 'https://.../web_model/tensorflowjs_model.pb'
const WEIGHTS_URL = 'http://.../web_model/weights_manifest.json';

const model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);

model.predict()