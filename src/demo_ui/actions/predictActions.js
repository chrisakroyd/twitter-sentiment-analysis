import { PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE } from '../constants/actions';

export function predict() {
  return {
    type: PREDICT,
  };
}

export function predictSuccess(data) {
  return {
    type: PREDICT_SUCCESS,
    data,
  };
}

export function predictFailure(errorCode) {
  return {
    type: PREDICT_FAILURE,
    errorCode,
  };
}
