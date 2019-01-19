import {
  PREDICT, PREDICT_TOKENS, PREDICT_SUCCESS, PREDICT_TOKENS_SUCCESS,
  PREDICT_FAILURE, PREDICT_TOKENS_FAILURE, CLEAR_ERROR, TOGGLE_TOKEN,
} from '../constants/actions';

export function predict() {
  return {
    type: PREDICT,
  };
}

export function predictSuccess(data) {
  const lastExample = data.data[data.numPredictions - 1];
  return {
    type: PREDICT_SUCCESS,
    data: lastExample,
  };
}

export function predictFailure(error) {
  return {
    type: PREDICT_FAILURE,
    error,
  };
}

export function predictTokens() {
  return {
    type: PREDICT_TOKENS,
  };
}

export function predictTokensSuccess(data) {
  const lastExample = data.data[data.numPredictions - 1];
  return {
    type: PREDICT_TOKENS_SUCCESS,
    data: lastExample,
  };
}

export function predictTokensFailure(error) {
  return {
    type: PREDICT_TOKENS_FAILURE,
    error,
  };
}

export function toggleToken(index) {
  return {
    type: TOGGLE_TOKEN,
    index,
  };
}

export function clearError() {
  return {
    type: CLEAR_ERROR,
  };
}
