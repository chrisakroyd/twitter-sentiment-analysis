import axios from 'axios';

import config from '../config';

import { examples, examplesSuccess, examplesFailure, classes, classesSuccess, classesFailure, setInputText } from './textActions';

import {
  predict, predictTokens, predictSuccess, predictTokensSuccess,
  predictFailure, predictTokensFailure, clearError,
} from './predictActions';

const demoUrl = `http://localhost:${config.demoPort}`;

function isErrorFixed(error, text) {
  let isFixed = false;
  if (error !== null) {
    const errorText = error.parameters.text;
    isFixed = text !== errorText;
  }
  return isFixed;
}

export function getPrediction() {
  return (dispatch, getState) => {
    const { text } = getState();

    dispatch(predict());

    return axios.post(`${demoUrl}/api/v1/model/predict`, { text: text.text })
      .then(res => dispatch(predictSuccess(res.data)))
      .catch(err => dispatch(predictFailure(err)));
  };
}

export function getExample() {
  return (dispatch) => {
    dispatch(examples());
    return axios.get(`${demoUrl}/api/v1/examples`, { params: { numExamples: 1 } })
      .then(res => dispatch(examplesSuccess(res.data)))
      .then(() => dispatch(getPrediction()))
      .catch(err => dispatch(examplesFailure(err)));
  };
}

export function setText(text) {
  return (dispatch, getState) => {
    // Test if we have an error, if we do and the text has changed, we clear that error as user
    // is taking steps to fix.
    const { predictions } = getState();
    if (isErrorFixed(predictions.error, text)) {
      dispatch(clearError());
    }
    dispatch(setInputText(text));
  };
}

export function predictWithTokens() {
  return (dispatch, getState) => {
    const { text, predictions } = getState();
    const tokens = [];
    predictions.enabled.forEach((flag, i) => {
      if (flag) {
        tokens.push(predictions.tokens[i]);
      } else {
        tokens.push('');
      }
    });

    dispatch(predictTokens());

    return axios.post(`${demoUrl}/api/v1/model/predictTokens`, { text: text.text, tokens })
      .then(res => dispatch(predictTokensSuccess(res.data)))
      .catch(err => dispatch(predictTokensFailure(err)));
  };
}

export function loadClasses() {
  return (dispatch) => {
    dispatch(classes());
    return axios.get(`${demoUrl}/api/v1/classes`)
      .then(res => dispatch(classesSuccess(res.data)))
      .catch(err => dispatch(classesFailure(err)));
  };
}
