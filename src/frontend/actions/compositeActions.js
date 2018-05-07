import axios from 'axios';

import config from '../config';
import { tweets, tweetsSuccess, tweetsFailure } from './twitterActions';
import {
  neural, neuralSuccess, neuralFailure, status, statusSuccess, statusFailure,
} from './neuralActions';

export function getTweets(type) {
  return (dispatch) => {
    dispatch(tweets());

    return axios.get(`${config.siteUrl}/api/v1/tweets/${type}/sample`)
      .then(res => dispatch(tweetsSuccess(res.data)))
      .catch(err => dispatch(tweetsFailure(err)));
  };
}

export function getModelStatus() {
  return (dispatch) => {
    dispatch(status());

    return axios.get(`${config.siteUrl}/api/v1/status/`)
      .then(res => dispatch(statusSuccess(res.data)))
      .catch(err => dispatch(statusFailure(err)));
  };
}

export function getPrediction() {
  return (dispatch) => {
    dispatch(neural());

    return axios.get(`${config.siteUrl}/api/v1/tweets/process`)
      .then(res => dispatch(neuralSuccess(res.data)))
      .catch(err => dispatch(neuralFailure(err)));
  };
}

export function getEmbeddings(type) {
  return (dispatch) => {
    dispatch(tweets());

    return axios.get(`${config.siteUrl}/api/v1/tweets/${type}/sample`)
      .then(res => dispatch(tweetsSuccess(res.data)))
      .catch(err => dispatch(tweetsFailure(err)));
  };
}

export function getDatasets(type) {
  return (dispatch) => {
    dispatch(tweets());

    return axios.get(`${config.siteUrl}/api/v1/tweets/${type}/sample`)
      .then(res => dispatch(tweetsSuccess(res.data)))
      .catch(err => dispatch(tweetsFailure(err)));
  };
}
