import { EXAMPLES, EXAMPLES_SUCCESS, EXAMPLES_FAILURE, SET_INPUT_TEXT } from '../constants/actions';

export function setInputText(text) {
  return {
    type: SET_INPUT_TEXT,
    text,
  };
}

export function examples() {
  return {
    type: EXAMPLES,
  };
}

export function examplesSuccess(data) {
  const lastExample = data.data[data.numExamples - 1];
  return {
    type: EXAMPLES_SUCCESS,
    text: lastExample.text,
  };
}

export function examplesFailure(error) {
  return {
    type: EXAMPLES_FAILURE,
    error,
  };
}
