import { SET_INPUT_TEXT } from '../constants/actions';

export function setInputText(text) {
  return {
    type: SET_INPUT_TEXT,
    text,
  };
}
