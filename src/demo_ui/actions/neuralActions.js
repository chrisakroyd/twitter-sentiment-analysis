import {
  NEURAL, NEURAL_SUCCESS, NEURAL_FAILURE, STATUS, STATUS_FAILURE, STATUS_SUCCESS,
} from '../constants/actions';

export function neural() {
  return {
    type: NEURAL,
  };
}

export function neuralSuccess(data) {
  return {
    type: NEURAL_SUCCESS,
    data,
  };
}

export function neuralFailure(errorCode) {
  return {
    type: NEURAL_FAILURE,
    errorCode,
  };
}

export function status() {
  return {
    type: STATUS,
  };
}

export function statusSuccess(data) {
  return {
    type: STATUS_SUCCESS,
    status: data,
  };
}

export function statusFailure(errorCode) {
  return {
    type: STATUS_FAILURE,
    errorCode,
  };
}
