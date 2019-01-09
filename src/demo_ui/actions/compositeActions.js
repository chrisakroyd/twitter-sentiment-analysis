import axios from 'axios';

import config from '../config';
import { tweets, tweetsSuccess, tweetsFailure } from './twitterActions';
import { predict, predictSuccess, predictFailure } from './predictActions';

const demoUrl = `http://localhost:${config.demoPort}`;

export function getTweets() {
  return (dispatch) => {
    dispatch(tweets());
    return axios.get(`${demoUrl}/api/v1/examples`, { params: { numExamples: 1 } })
      .then(res => dispatch(tweetsSuccess(res.data)))
      .catch(err => dispatch(tweetsFailure(err)));
  };
}


export function getPrediction() {
  return (dispatch, getState) => {
    const active = getState().activeText;

    dispatch(predict());

    return axios.post(`${demoUrl}/api/v1/model/predict`, { text: active.text })
      .then(res => dispatch(predictSuccess(res.data)))
      .catch(err => dispatch(predictFailure(err)));
  };
}
