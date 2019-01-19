import { EXAMPLES, EXAMPLES_SUCCESS, EXAMPLES_FAILURE, SET_INPUT_TEXT } from '../constants/actions';


const text = (state = {}, action) => {
  switch (action.type) {
    case SET_INPUT_TEXT: {
      return Object.assign({}, state, {
        text: action.text,
      });
    }
    case EXAMPLES:
      return {
        text: '',
        loading: true,
      };
    case EXAMPLES_SUCCESS:
      return {
        text: action.text,
        loading: false,
      };
    case EXAMPLES_FAILURE:
      return Object.assign({}, state, {
        loading: false,
        error: action.error,
      });
    default:
      return state;
  }
};

export default text;
